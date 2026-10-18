"""Frozen-checkpoint, paired diagnosis. Oracle paths are evaluator-only."""
import argparse
from functools import partial
import hashlib
from itertools import permutations
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.ode import odeint
from scipy.integrate import solve_ivp

from keycld import util
from keycld.models import predict
from learned_common import load_model
from train_paper import load_observation


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def rmse(a, b):
    # Diagnostic coordinates can be very large under coarse, unstable Euler.
    # Promote before subtraction/squaring so float32 overflow is not a scorer bug.
    difference = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if not np.isfinite(difference).all():
        return None
    return float(np.sqrt(np.mean(difference ** 2)))


def summary(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values):
        return {'n': 0, 'mean': None, 'std': None, 'median': None}
    return {'n': len(values), 'mean': float(values.mean()),
            'std': float(values.std()), 'median': float(np.median(values))}


def paired_difference(rows, key, a, b):
    differences = np.asarray([r[key][a] - r[key][b] for r in rows
                              if r[key][a] is not None and r[key][b] is not None])
    rng = np.random.RandomState(20261010)
    if not len(differences):
        return {'n': 0}
    boot = np.mean(rng.choice(differences, (2000, len(differences))), axis=1)
    return {'n': len(differences), 'mean_a_minus_b': float(differences.mean()),
            'bootstrap_95_percent_interval': np.percentile(boot, [2.5, 97.5]).tolist()}


def fixed_solver(method, substeps):
    def solve(f, initial, times, action):
        def interval(y, inputs):
            t, dt = inputs
            h = dt / substeps
            def step(k, state):
                instant = t + k*h
                k1 = f(state, instant, action)
                if method == 'euler':
                    return state + h*k1
                k2 = f(state + h*k1/2, instant + h/2, action)
                k3 = f(state + h*k2/2, instant + h/2, action)
                k4 = f(state + h*k3, instant + h, action)
                return state + h*(k1 + 2*k2 + 2*k3 + k4)/6
            y = jax.lax.fori_loop(0, substeps, step, y)
            return y, y
        _, history = jax.lax.scan(interval, initial, (times[:-1], jnp.diff(times)))
        return jnp.concatenate([initial[None], history])
    return solve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root, args.out
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'source'))
    sys.path.insert(0, str(root/'source/fourbar'))
    from prepare import points
    from mechanics import reduced_acceleration, inverse_kinematics, closure_jacobian
    start = time.monotonic()
    protocol = json.loads((out/'diagnosis_protocol.json').read_text())
    checkpoint_hash = hashlib.sha256((root/'trained_params.p').read_bytes()).hexdigest()
    assert checkpoint_hash == protocol['checkpoint_sha256']
    geometry = json.loads((root/'geometry.json').read_text())
    scale = geometry['half_extent_m']; center = np.asarray(geometry['world_xz_center_m'])
    threshold = json.loads((root/'vpt_thresholds.json').read_text())['independent_test_train_only']
    with (root/'trained_params.p').open('rb') as handle:
        params = pickle.load(handle)
    model = load_model(root)

    def status(stage, **fields):
        write_json(out/'status.json', {'stage': stage, 'elapsed_seconds': time.monotonic()-start, **fields})
        print(stage, json.dumps(fields), flush=True)

    def truth_for(index):
        with np.load(root/'truth'/f'run_{index:04d}.npz') as data:
            truth = {key: data[key] for key in data.files}
        config = json.loads((root/'truth'/f'run_{index:04d}.json').read_text())
        position, velocity, acceleration = points(truth['q'], truth['qdot'], truth['qddot'], config)
        add_midpoint = lambda xy: np.concatenate([xy, xy.mean(axis=1, keepdims=True)], axis=1)
        return truth, config, (add_midpoint(position)-center)/scale, add_midpoint(velocity)/scale, add_midpoint(acceleration)/scale

    @jax.jit
    def encode(parameters, images):
        kp, _ = util.map_to_keypoints(model.encoder(parameters, images))
        return kp

    @jax.jit
    def render(parameters, kp):
        return model.renderer(parameters, kp)[0]

    def constraints(kp):
        kp = np.asarray(kp, dtype=np.float64)
        B,C,P = kp[:,0],kp[:,1],kp[:,2]
        A,D = np.asarray(geometry['A_normalized']),np.asarray(geometry['D_normalized'])
        lengths = geometry['link_lengths_normalized']
        distances = np.stack([np.linalg.norm(B-A,axis=1)-lengths[0],
                              np.linalg.norm(C-B,axis=1)-lengths[1],
                              np.linalg.norm(C-D,axis=1)-lengths[2]],axis=1)
        return np.concatenate([distances,P-(B+C)/2],axis=1)
    acceleration = jax.jit(jax.vmap(model.euler_lagrange, in_axes=(None, 0, 0, None)))

    methods = {'odeint': odeint}
    for kind in ['rk4', 'euler']:
        for steps in [1, 4, 16]:
            methods[f'{kind}_{steps}'] = fixed_solver(kind, steps)
    def make_forecast(solver):
        @jax.jit
        def forecast(parameters, images, times, action):
            maps=model.encoder(parameters, images[:3])
            keypoints,_=util.map_to_keypoints(maps)
            box=[]
            def capture(f, state, t, u):
                history=solver(f,state,t,u);box.append(history);return history
            predicted=predict(model,parameters,times,keypoints,action,solver=capture)
            rendered,_=model.renderer(parameters,predicted)
            history=box[0]
            full=jnp.concatenate([jnp.concatenate([keypoints[0].ravel(),history[0,6:]])[None],history])
            a=jax.vmap(lambda y:model.euler_lagrange(parameters,y[:6],y[6:],action))(full)
            return predicted,rendered,full[:,6:].reshape(-1,3,2),a.reshape(-1,3,2)
        return forecast
    # Match the original evaluator's runtime parameter argument, fused encoder /
    # rollout / renderer, prefix batch shape and diagnostic outputs. Constant-
    # closing parameters in separate JITs changes float32 compiler arithmetic.
    forecasts = {name:make_forecast(solver) for name,solver in methods.items()}

    # Permutation is selected using validation only, before reading test data.
    status('semantic_validation', completed=0, total=50)
    candidates = list(permutations(range(3)))
    scores = np.zeros(6)
    for index in range(450, 500):
        run = load_observation(root, index)
        kp = np.asarray(encode(params, run['x']))
        _, _, physical, _, _ = truth_for(index)
        for k, permutation in enumerate(candidates):
            scores[k] += np.mean((kp[:, permutation, :] - physical)**2)
    permutation = candidates[int(np.argmin(scores))]
    semantic_mapping = {'validation_selected_permutation': list(permutation), 'selection_indices': [450, 499],
                        'validation_rmse_m_by_permutation': {str(p):float(np.sqrt(s/50)*scale) for p,s in zip(candidates,scores)},
                        'affine_fit': False, 'used_for_rollout': False}
    mean_train = np.mean([load_observation(root, i)['x'].mean(axis=0) for i in range(450)], axis=0)
    # Fixed oracle state mean gives an honest position-estimation null control.
    mean_physical = np.mean([truth_for(i)[2].mean(axis=0) for i in range(450)], axis=0)

    @jax.jit
    def local_rollouts(parameters, kp, t, action):
        flat = kp.reshape(50, 6)
        x, v = util.finite_difference(flat, t[1]-t[0])
        v = jax.vmap(partial(util.project_velocity, model.constraint_fn))(x, v)
        initial = jnp.concatenate([x, v], axis=1)[:44]
        relative = jnp.arange(5)*(t[1]-t[0])
        result = {}
        for name in ['euler_1', 'rk4_4']:
            histories = jax.vmap(methods[name], in_axes=(None, 0, None, None))(
                partial(model.ode, parameters), initial, relative, action)
            result[name] = histories[:, :, :6].reshape(44, 5, 3, 2)
        return result

    rows = []
    for index in range(500, 550):
        run = load_observation(root, index)
        truth, config, physical, physical_v, physical_a = truth_for(index)
        kp = np.asarray(encode(params, run['x']))
        reconstructed = np.asarray(render(params, kp))
        persistent = np.repeat(run['x'][2:3], 50, axis=0); persistent[:3] = run['x'][:3]
        average = np.repeat(mean_train[None], 50, axis=0); average[:3] = run['x'][:3]
        images = {'framewise_oracle': reconstructed, 'persistence': persistent, 'training_mean': average}
        latent = {}
        for name, forecast in forecasts.items():
            prediction,rendered,velocity,force=jax.device_get(forecast(params,run['x'],run['t'],run['action']))
            latent[name] = np.asarray(prediction)
            images[name] = np.asarray(rendered)
        # Test-only motion mask measures changing pixels; it is not a model input.
        mask = np.mean(np.var(run['x'], axis=0), axis=-1) > 1e-4
        assert mask.any()
        image_mse = {}; foreground_mse = {}; vpt = {}; finite = {}
        for name, rgb in images.items():
            finite[name] = bool(np.isfinite(rgb).all() and
                                (name not in latent or np.isfinite(latent[name]).all()))
            image_mse[name] = float(np.mean((rgb[3:]-run['x'][3:])**2)) if finite[name] else None
            foreground_mse[name] = float(np.mean((rgb[3:, mask]-run['x'][3:, mask])**2)) if finite[name] else None
            vpt[name] = float(util.calculate_vpt(threshold, [run], [{'x_recon':rgb}])[0]) if finite[name] else None
        phi = np.asarray(constraints(kp))*scale
        actual_acceleration = np.asarray(acceleration(params, physical.reshape(50,6).astype(np.float32),
                                                    physical_v.reshape(50,6).astype(np.float32), run['action'])).reshape(50,3,2)
        local = jax.device_get(local_rollouts(params, kp, run['t'], run['action']))
        targets = np.stack([kp[i+1:i+6] for i in range(44)])
        local_errors = {}
        for name, coordinates in local.items():
            local_errors[name] = {key: error*scale if error is not None else None for key,error in
                                  [('one_step_rmse_m',rmse(coordinates[:,1],targets[:,1])),
                                   ('four_step_rmse_m',rmse(coordinates[:,4],targets[:,4]))]}
        local_errors['latent_persistence'] = {'one_step_rmse_m':rmse(targets[:,0],targets[:,1])*scale,
                                              'four_step_rmse_m':rmse(targets[:,0],targets[:,4])*scale}
        row = {'index':index, 'torque_Nm':float(run['action'][0]), 'image_mse':image_mse,
               'foreground_mse':foreground_mse, 'vpt':vpt, 'finite':finite,
               'encoded_rmse_declared_m':rmse(kp,physical)*scale,
               'encoded_rmse_validation_permutation_m':rmse(kp[:,permutation,:],physical)*scale,
               'constant_position_rmse_m':rmse(mean_physical[None],physical)*scale,
               'encoded_constraint_rmse_m':float(np.sqrt(np.mean(phi**2))),
               'encoded_distance_max_error_m':float(np.max(np.abs(phi[:,:3]))),
               'local_latent_prediction':local_errors,
               'solver_position_difference_from_odeint_m':{name:rmse(value[3:],latent['odeint'][3:])*scale
                                                          if np.isfinite(value).all() else None for name,value in latent.items()},
               'solver_constraint_drift_m':{name:rmse(constraints(value)[1:],constraints(value)[1:2])*scale
                                            if np.isfinite(value).all() else None for name,value in latent.items()},
               'learned_acceleration_true_coordinate_relative_error':float(np.linalg.norm(actual_acceleration-physical_a)/np.linalg.norm(physical_a)),
               'learned_acceleration_true_coordinate_rmse_m_s2':rmse(actual_acceleration,physical_a)*scale,
               'true_coordinate_probe_semantics': 'Not a causal dynamics verdict if encoded coordinates do not match physical coordinates.',
               'motion_mask_pixels':int(mask.sum()),
               'max_sample_crank_displacement_rad':float(np.max(np.abs(np.diff(truth['q'][:,0])))),
               'sampling_alias_intervals_above_pi':int(np.sum(np.abs(np.diff(truth['q'][:,0]))>np.pi))}
        rows.append(row)
        if index == 500:
            np.savez_compressed(out/'visual_example.npz', time=run['t'], reference_rgb=np.rint(run['x']*255).astype(np.uint8),
                                reconstructed_rgb=np.rint(reconstructed*255).clip(0,255).astype(np.uint8),
                                predicted_rgb=np.rint(images['odeint']*255).clip(0,255).astype(np.uint8),
                                encoded_normalized=kp, physical_normalized=physical)
        if (index-499)%10==0:
            status('frozen_checkpoint_diagnosis', completed=index-499, total=50)
            write_json(out/'learned_rows.json', rows)

    # Exact-state independent dynamics: isolate integration and velocity sampling.
    oracle_rows = []
    saved_oracle=out/'oracle_rows.json'
    if saved_oracle.exists():
        cached=json.loads(saved_oracle.read_text())
        if [r['index'] for r in cached]==list(range(500,550)):
            oracle_rows=cached
            status('independent_physics_oracle_cached',completed=50,total=50)
    for index in range(500, 550):
        if len(oracle_rows)==50:break
        truth, config, physical, physical_v, physical_a = truth_for(index)
        torque = float(load_observation(root,index)['action'][0])
        t = truth['time'][1:]
        exact = np.array([truth['q'][1,0], truth['qdot'][1,0]])
        B = physical[:3,0]*scale+center
        velocity_fd = (B[2]-B[0])/(truth['time'][2]-truth['time'][0])
        theta = exact[0]; normal = np.array([-np.sin(theta),np.cos(theta)])
        omega_fd = float(normal@velocity_fd/config['lengths_m'][0])
        sampled = np.array([exact[0],omega_fd])
        def equation(instant,state):
            return np.array([state[1],reduced_acceleration(state[0],state[1],torque,config)])
        def reference(initial):
            solution = solve_ivp(equation,(t[0],t[-1]),initial,t_eval=t,
                                 method='DOP853',rtol=1e-10,atol=1e-11)
            assert solution.success and solution.y.shape==(2,49)
            return solution.y.T
        histories = {'dop853_exact':reference(exact), 'dop853_sampled_velocity':reference(sampled)}
        for method,substeps in [('euler',1),('euler',4),('euler',16),('rk4',4)]:
            state=exact.copy();history=[state.copy()]
            with np.errstate(over='ignore',invalid='ignore'):
                for ta,tb in zip(t[:-1],t[1:]):
                    h=(tb-ta)/substeps
                    for k in range(substeps):
                        instant=ta+k*h
                        k1=equation(instant,state)
                        if method=='euler':state=state+h*k1
                        else:
                            k2=equation(instant+h/2,state+h*k1/2)
                            k3=equation(instant+h/2,state+h*k2/2)
                            k4=equation(instant+h,state+h*k3)
                            state=state+h*(k1+2*k2+2*k3+k4)/6
                        if not np.isfinite(state).all():break
                    history.append(state.copy())
                    if not np.isfinite(state).all():break
            histories[f'{method}_{substeps}']=np.asarray(history)
        errors={};phase_errors={};finite={}
        for name,history in histories.items():
            valid=history.shape==(49,2) and bool(np.isfinite(history).all())
            finite[name]=valid
            if valid:
                q=np.array([inverse_kinematics(theta,config) for theta in history[:,0]])
                xy,_,_=points(q,np.zeros_like(q),np.zeros_like(q),config)
                target=physical[1:,:2]*scale+center
                errors[name]=rmse(xy[2:],target[2:])
                phase_errors[name]=rmse(history[2:,0],truth['q'][3:,0])
            else:errors[name]=None;phase_errors[name]=None
        oracle_rows.append({'index':index,'position_rmse_m':errors,'crank_unwrapped_phase_rmse_rad':phase_errors,
                            'finite':finite,'initial_omega_exact_rad_s':float(exact[1]),
                            'initial_omega_fd_rad_s':omega_fd,'initial_omega_abs_error_rad_s':abs(omega_fd-exact[1])})
        if (index-499)%10==0:
            status('independent_physics_oracle',completed=index-499,total=50)
            write_json(out/'oracle_rows.json',oracle_rows)

    original=json.loads((root/'evaluation_report.json').read_text())['conditions']['trained_in_distribution']['runs']
    reproduction=max(abs(a['image_mse']['odeint']-b['future_image_mse']) for a,b in zip(rows,original))
    assert reproduction<1e-6, reproduction
    report={'protocol':protocol,'checkpoint_sha256':checkpoint_hash,'threshold':threshold,'run_count':50,
            'baseline_future_mse_max_abs_difference':reproduction,'semantic_mapping':semantic_mapping,
            'summary':{'image_mse':{name:summary([r['image_mse'][name] for r in rows]) for name in rows[0]['image_mse']},
                       'foreground_mse':{name:summary([r['foreground_mse'][name] for r in rows]) for name in rows[0]['foreground_mse']},
                       'vpt':{name:summary([r['vpt'][name] for r in rows]) for name in rows[0]['vpt']},
                       'visual_state':{key:summary([r[key] for r in rows]) for key in ['encoded_rmse_declared_m','encoded_rmse_validation_permutation_m','constant_position_rmse_m','encoded_constraint_rmse_m','encoded_distance_max_error_m','learned_acceleration_true_coordinate_relative_error','learned_acceleration_true_coordinate_rmse_m_s2','max_sample_crank_displacement_rad']},
                       'oracle_position_rmse_m':{name:summary([r['position_rmse_m'][name] for r in oracle_rows]) for name in oracle_rows[0]['position_rmse_m']},
                       'initial_omega_abs_error_rad_s':summary([r['initial_omega_abs_error_rad_s'] for r in oracle_rows]),
                       'local_latent_prediction':{name:{key:summary([r['local_latent_prediction'][name][key] for r in rows]) for key in ['one_step_rmse_m','four_step_rmse_m']} for name in rows[0]['local_latent_prediction']},
                       'solver_position_difference_from_odeint_m':{name:summary([r['solver_position_difference_from_odeint_m'][name] for r in rows]) for name in methods},
                       'solver_constraint_drift_m':{name:summary([r['solver_constraint_drift_m'][name] for r in rows]) for name in methods},
                       'finite_run_counts':{name:sum(r['finite'][name] for r in rows) for name in rows[0]['finite']},
                       'sampling_alias_intervals_above_pi_total':sum(r['sampling_alias_intervals_above_pi'] for r in rows)},
            'paired':{key:{f'{a}_minus_{b}':paired_difference(rows,key,a,b) for a,b in [('framewise_oracle','odeint'),('rk4_16','odeint'),('euler_1','odeint'),('odeint','persistence')]} for key in ['image_mse','vpt']},
            'elapsed_seconds':time.monotonic()-start,'training_performed':False,'devices':[str(d) for d in jax.devices()]}
    write_json(out/'report.json',report)
    status('complete',run_count=50)


if __name__=='__main__':
    main()
