"""Frozen full-checkpoint evaluation and independent equation/identifiability controls."""
import argparse
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import time

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import numpy as np
from scipy.optimize import lsq_linear
import extended_mechanics as physics

SCALE=.43*63/64
CENTER=np.array([.2,.37])


def write(path,value):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def stats(values):
    a=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    return {'n':len(a),'mean':float(a.mean()) if len(a) else None,'median':float(np.median(a)) if len(a) else None,
            'std':float(a.std()) if len(a) else None,'max':float(a.max()) if len(a) else None}


def rms(array):
    a=np.asarray(array,dtype=float)
    return float(np.sqrt(np.mean(a*a))) if a.size and np.isfinite(a).all() else None


def model_for(c):
    A=jnp.asarray((np.array([0,c['pivot_height_m']])-CENTER)/SCALE)
    D=jnp.asarray((np.array([c['ground_length_m'],c['pivot_height_m']])-CENTER)/SCALE)
    lengths=jnp.asarray(c['lengths_m'])/SCALE;slider=physics.is_slider(c)
    def constraints(x):
        B,C,P=x.reshape(3,2)
        third=C[1]-A[1] if slider else jnp.linalg.norm(C-D)-lengths[2]
        return jnp.r_[jnp.linalg.norm(B-A)-lengths[0],jnp.linalg.norm(C-B)-lengths[1],third,P-(B+C)/2]
    return KeyCLD(3,3,32,True,constraints,image_size=64)


def load(root,name,index):
    with np.load(root/'observations'/name/f'run_{index:04d}.npz') as data:run={key:data[key] for key in data.files}
    run['x']=run['x'].astype(np.float32)/255
    with np.load(root/'truth'/name/f'run_{index:04d}.npz') as data:truth={key:data[key] for key in data.files}
    c=json.loads((root/'truth'/name/f'run_{index:04d}.json').read_text())
    states=[physics.point_states(q,v,a,c) for q,v,a in zip(truth['q'],truth['qdot'],truth['qddot'])]
    return run,truth,c,np.asarray([s[0] for s in states]),np.asarray([s[1] for s in states])


def forecast_functions(model):
    @jax.jit
    def video(parameters,images,t,action):
        kp,_=util.map_to_keypoints(model.encoder(parameters,images[:3]))
        box=[]
        def capture(f,y,t,u):
            history=odeint(f,y,t,u);box.append(history);return history
        x=predict(model,parameters,t,kp,action,solver=capture)
        v=jnp.concatenate([box[0][0:1,6:],box[0][:,6:]],axis=0)
        full=jnp.concatenate([x.reshape(50,6),v],axis=1)
        a=jax.vmap(lambda y:model.euler_lagrange(parameters,y[:6],y[6:],action))(full)
        return x,v.reshape(50,3,2),a.reshape(50,3,2)
    @jax.jit
    def state(parameters,x,v,t,action):
        initial=jnp.r_[x[1].ravel(),util.project_velocity(model.constraint_fn,x[1].ravel(),v[1].ravel())]
        history=odeint(partial(model.ode,parameters),initial,t[1:],action)
        full=jnp.concatenate([jnp.r_[x[0].ravel(),v[0].ravel()][None],history])
        a=jax.vmap(lambda y:model.euler_lagrange(parameters,y[:6],y[6:],action))(full)
        return full[:,:6].reshape(50,3,2),full[:,6:].reshape(50,3,2),a.reshape(50,3,2)
    return video,state


def trajectory_score(x,v,a,action,c,gate):
    finite=all(np.isfinite(y).all() for y in [x,v,a])
    if not finite:return {'finite':False,'geometry_valid_frames':0,'future_frames':47,'motor_torque_error_rms_Nm':None}
    errors=np.asarray([physics.geometry_error(y,c) for y in x]);legal=np.max(np.abs(errors),axis=1)<gate
    residual=[];term_rows=[]
    for i in range(3,50):
        if not legal[i]:continue
        decoded=physics.decode(x[i],v[i],a[i],c)
        if decoded is None:continue
        q,qd,qdd=decoded;terms=physics.terms(q,qd,qdd,action[0],c)
        value=terms['inertia_Nm']+terms['velocity_quadratic_Nm']+terms['gravity_Nm']+terms['friction_Nm']-terms['external_Nm']-terms['motor_Nm']
        residual.append(value);term_rows.append(terms)
    return {'finite':True,'future_frames':47,'geometry_valid_frames':len(residual),
            'geometry_residual_rms_m':rms(errors[3:]),'geometry_residual_max_m':float(np.max(np.abs(errors[3:]))),
            'motor_torque_error_rms_Nm':rms(residual),
            'equation_term_rms_Nm':{key:rms([row[key] for row in term_rows]) for key in term_rows[0]} if term_rows else {},
            'physics_status':'all_future_states_legal' if len(residual)==47 else 'not_a_complete_physical_trajectory'}


def parameter_controls(root,configs):
    result={}
    for name,c in configs.items():
        matrices=[];targets=[];ids=[];truth_parameters=None
        for index in range(1000,1020):
            run,truth,_,_,_=load(root,name,index)
            for q,v,a in zip(truth['q'][3:],truth['qdot'][3:],truth['qddot'][3:]):
                row,rotor,ext,truth_parameters=physics.regressor(q,v,a,c)
                matrices.append(row);targets.append(float(run['action'][0])+ext-rotor);ids.append(index)
        A=np.asarray(matrices);y=np.asarray(targets);cal=np.asarray(ids)<1012
        scale=np.sqrt(np.mean(A[cal]**2,axis=0));scale=np.where(scale>1e-15,scale,1.)
        normalized=A[cal]/scale;singular=np.linalg.svd(normalized,compute_uv=False)
        rank=int(np.sum(singular>singular[0]*1e-10));condition=float(singular[0]/singular[-1])
        fit=lsq_linear(normalized,y[cal],bounds=(0,np.inf),tol=1e-12,max_iter=2000)
        estimate=fit.x/scale
        full_rank=rank==12;well_conditioned=condition<1e8
        result[name]={'uses_ground_truth_states':True,'model_result':False,'fit_seed_count':12,'test_seed_count':8,
                      'regressor_rank':rank,'parameter_count':12,'normalized_condition_number':condition,
                      'individual_parameter_interpretation_valid':bool(full_rank and well_conditioned),
                      'true_parameters':truth_parameters.tolist(),'estimated_parameters':estimate.tolist(),
                      'parameter_order':['m_crank_kg','m_coupler_kg','m_rocker_or_slider_kg','m_payload_kg',
                                         'b_A','b_B','b_C','b_D_or_S','fc_A','fc_B','fc_C','fc_D_or_S'],
                      'calibration_torque_rms_Nm':rms(A[cal]@estimate-y[cal]),
                      'test_torque_rms_Nm':rms(A[~cal]@estimate-y[~cal]),
                      'mass_relative_errors':((estimate[:4]-truth_parameters[:4])/truth_parameters[:4]).tolist() if full_rank and well_conditioned else None,
                      'viscous_absolute_errors':(estimate[4:8]-truth_parameters[4:8]).tolist() if full_rank and well_conditioned else None,
                      'coulomb_absolute_errors':(estimate[8:]-truth_parameters[8:]).tolist() if full_rank and well_conditioned else None,
                      'fit_success':bool(fit.success)}
    return result


def constitutive_probes(model,parameters,c):
    angles=np.linspace(-np.pi,np.pi,73)
    accel=jax.jit(jax.vmap(model.euler_lagrange,in_axes=(None,0,0,None)))
    def query(theta,omega,tau):
        X=[];V=[];N=[];H=[];ideal=[]
        for t,w in zip(theta,omega):
            q=physics.inverse(t,c);gamma,gamma2=physics.derivatives(q,c)
            x,v,a=physics.point_states(q,gamma*w,gamma2*w*w,c)
            _,n,h=physics.point_states(q,gamma,gamma2,c)
            X.append((x-CENTER)/SCALE);V.append(v/SCALE);N.append(n);H.append(h*w*w);ideal.append((q,gamma))
        raw=np.asarray(accel(parameters,np.asarray(X,dtype=np.float32).reshape(-1,6),
                             np.asarray(V,dtype=np.float32).reshape(-1,6),np.array([tau,0.,0.],dtype=np.float32))).reshape(-1,3,2)*SCALE
        N=np.asarray(N);H=np.asarray(H)
        alpha=np.sum(N*(raw-H),axis=(1,2))/np.sum(N*N,axis=(1,2))
        return alpha,ideal
    zero=np.zeros(73);minus,ideal=query(angles,zero,-1.);plus,_=query(angles,zero,1.);offset,_=query(angles,zero,0.)
    gains=(plus-minus)/2;true_I=[];true_G=[];external=[];mass_I=[];mass_G=[];true_masses=None
    for q,gamma in ideal:
        M,C,G=physics.mass_bias(q,np.zeros(3),c);true_I.append(float(gamma@M@gamma));true_G.append(float(gamma@G))
        external.append(float(gamma@physics.external(q,c)))
        true_masses,parts=physics.bodies(q,np.zeros(3),c)
        mass_I.append([float(gamma@part[0]@gamma) for part in parts]);mass_G.append([float(gamma@part[2]) for part in parts])
    true_I=np.asarray(true_I);true_G=np.asarray(true_G);valid=np.isfinite(gains)&(gains>1e-12)
    estimated=np.full(73,np.nan);estimated[valid]=1/gains[valid]
    inertia={'positive_gain_poses':int(valid.sum()),'total_poses':73,
             'equivalent_inertia_relative_error':stats(np.abs((estimated[valid]-true_I[valid])/true_I[valid])),
             'equivalent_inertia_rmse_kg_m2':rms(estimated[valid]-true_I[valid]),
             'interpretation':'Behavior-implied effective SI inertia from a known N*m input, not native network mass parameters.'}
    mass_fit={'native_SI_mass_identifiable':False,'body_mass_interpretation':'Unavailable until all gains are positive and the physical regressor is full rank.'}
    if valid.all():
        implied_G=np.asarray(external)-estimated*offset
        # Fixed unit scales, declared here before observing fit results. Fit never changes weights.
        A=np.vstack([np.asarray(mass_I)/.005,np.asarray(mass_G)/.5])
        y=np.r_[(estimated-c['motor_armature_kg_m2'])/.005,implied_G/.5]
        singular=np.linalg.svd(A,compute_uv=False);rank=int(np.sum(singular>singular[0]*1e-10))
        fit=lsq_linear(A,y,bounds=(0,np.inf),tol=1e-12,max_iter=2000)
        mass_fit.update(behavior_fit_rank=rank,behavior_implied_body_masses_kg=fit.x.tolist() if rank==4 else None,
                        true_body_masses_kg=true_masses.tolist(),
                        relative_errors=((fit.x-true_masses)/true_masses).tolist() if rank==4 else None,
                        physical_profile_fit_rms_scaled=rms(A@fit.x-y),
                        body_mass_interpretation='Evaluator-only fit of the observable inertia/gravity response to known uniform-body shapes; not direct network parameter identification.')
    theta=np.tile(angles,4);speed=np.repeat([.25,1.,4.,12.],73)
    positive,poses=query(theta,speed,1.);negative,_=query(theta,-speed,1.)
    odd=(positive-negative)/2;true_odd=[];features=[];true_torque=[];inferred=[]
    for t,w,(q,gamma),learned_odd in zip(theta,speed,poses,odd):
        plus_a=physics.acceleration(t,w,1.,c);minus_a=physics.acceleration(t,-w,1.,c)
        true_odd.append((plus_a-minus_a)/2)
        M,_,_=physics.mass_bias(q,np.zeros(3),c);I=float(gamma@M@gamma)
        s=physics.rate_map(c)@gamma;rates=s*w
        features.append(np.r_[s*rates,s*np.tanh(rates/c['coulomb_smoothing_rad_s'])])
        true_torque.append(float(-gamma@physics.friction(gamma*w,c)));inferred.append(float(-I*learned_odd))
    true_odd=np.asarray(true_odd);features=np.asarray(features)
    coefficients=np.linalg.lstsq(features,np.asarray(inferred),rcond=1e-10)[0]
    friction={'matched_state_count':292,'learned_odd_acceleration_max_abs_rad_s2':float(np.max(np.abs(odd))),
              'true_odd_acceleration_rms_rad_s2':rms(true_odd),'odd_acceleration_error_rms_rad_s2':rms(odd-true_odd),
              'odd_acceleration_relative_error':float(np.linalg.norm(odd-true_odd)/np.linalg.norm(true_odd)) if np.linalg.norm(true_odd)>1e-12 else None,
              'friction_torque_error_rms_Nm':rms(np.asarray(inferred)-true_torque),
              'friction_basis_rank':int(np.linalg.matrix_rank(features)),
              'behavior_projected_b':coefficients[:4].tolist(),'behavior_projected_fc':coefficients[4:].tolist(),
              'true_b':c['viscous_friction_Nm_s_rad'],'true_fc':c['coulomb_friction_Nm'],
              'interpretation':'Projection of model odd acceleration into the specified friction basis using evaluator-only true inertia; not a native coefficient head.'}
    return {'inertia':inertia,'SI_mass':mass_fit,'friction':friction}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--controls-only',action='store_true',help='Run independent physical parameter controls without a neural runtime')
    args=parser.parse_args();root=args.root;start=time.monotonic()
    protocol=json.loads((root/'extended_protocol.json').read_text());configs=json.loads((root/'condition_configs.json').read_text())
    if args.controls_only:
        controls=parameter_controls(root,configs)
        write(root/'identifiability_controls.json',controls)
        print(json.dumps({name:{k:value[k] for k in ['regressor_rank','normalized_condition_number','test_torque_rms_Nm','mass_relative_errors']} for name,value in controls.items()}),flush=True)
        return
    global jax,jnp,odeint,KeyCLD,predict,util
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
    from keycld.models import KeyCLD,predict
    from keycld import util
    original_report=json.loads((args.baseline/'training_report.json').read_text())
    state_report=json.loads((args.state/'report.json').read_text())
    assert original_report['epochs_completed']==state_report['epochs_completed']==40
    assert original_report['optimizer_updates']==state_report['optimizer_updates']==18000
    files={'video':args.baseline/'trained_params.p','state_RK4':args.state/'params.p'}
    hashes={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in files.items()}
    assert hashes['video']==original_report['trained_checkpoint_sha256']
    assert hashes['state_RK4']==state_report['checkpoint_sha256']
    parameters={}
    for name,path in files.items():
        with path.open('rb') as handle:parameters[name]=pickle.load(handle)
    def status(stage,**fields):
        value={'stage':stage,'elapsed_seconds':time.monotonic()-start,**fields}
        write(root/'evaluation_status.json',value);print(json.dumps(value),flush=True)
    status('identifiability_controls')
    controls=parameter_controls(root,configs);write(root/'identifiability_controls.json',controls)
    status('field_probes')
    probes={};examples={};results={}
    torques=[row['torque_Nm'] for row in json.loads((root/'dataset_manifest.json').read_text())['rows'] if row['condition']=='nominal']
    torque_reference=float(np.sqrt(np.mean(np.asarray(torques)**2)))
    for name,c in configs.items():
        model=model_for(c);probes[name]=constitutive_probes(model,parameters['state_RK4'],c)
        status('field_probes',condition=name)
    # Explicit gauge: masses, potential and action mapping can scale together.
    model=model_for(configs['nominal']);q=physics.inverse(.7,configs['nominal']);gamma,gamma2=physics.derivatives(q,configs['nominal'])
    x,v,_=physics.point_states(q,gamma*2.,gamma2*4.,configs['nominal'])
    x=jnp.asarray(((x-CENTER)/SCALE).ravel(),dtype=jnp.float32);v=jnp.asarray((v/SCALE).ravel(),dtype=jnp.float32)
    reference=np.asarray(model.euler_lagrange(parameters['state_RK4'],x,v,jnp.array([1.,0.,0.])))
    gauge=[]
    for factor in [.25,4.]:
        scaled=jax.tree_util.tree_map(lambda a:a,parameters['state_RK4'])
        scaled['mass_matrix']['params']['point_masses']=scaled['mass_matrix']['params']['point_masses']*np.sqrt(factor)
        for key in ['potential_energy','input_matrix']:
            for leaf in ['kernel','bias']:scaled[key]['params']['Dense_2'][leaf]=scaled[key]['params']['Dense_2'][leaf]*factor
        prediction=np.asarray(model.euler_lagrange(scaled,x,v,jnp.array([1.,0.,0.])))
        gauge.append({'mass_matrix_scale':factor,'max_acceleration_change_m_s2':float(np.max(abs(prediction-reference)))*SCALE})
    write(root/'field_probes.json',{'conditions':probes,'mass_scale_gauge':gauge,'native_SI_parameter_identification':'structurally nonunique while input map and potential are freely learned'})
    for name,c in configs.items():
        model=model_for(c);video,state=forecast_functions(model);groups={'video':[],'state_RK4':[]}
        for index in range(1000,1020):
            run,truth,configuration,xy,velocity=load(root,name,index)
            normalized=((xy-CENTER)/SCALE).astype(np.float32);vnorm=(velocity/SCALE).astype(np.float32)
            output={
                'video':jax.device_get(video(parameters['video'],run['x'],run['t'],run['action'])),
                'state_RK4':jax.device_get(state(parameters['state_RK4'],normalized,vnorm,run['t'],run['action']))}
            for label,arrays in output.items():
                xp,vp,ap=arrays;xp=np.asarray(xp,dtype=float)*SCALE+CENTER;vp=np.asarray(vp,dtype=float)*SCALE;ap=np.asarray(ap,dtype=float)*SCALE
                row={'index':index,**trajectory_score(xp,vp,ap,run['action'],configuration,protocol['geometry_gate_m'])}
                groups[label].append(row)
            if index==1000:
                i=10;theta=float(truth['q'][i,0]);omega=float(truth['qdot'][i,0]);tau=float(run['action'][0])
                ideal_q=physics.inverse(theta,c);g,g2=physics.derivatives(ideal_q,c);alpha=physics.acceleration(theta,omega,tau,c)
                M,C,G=physics.mass_bias(ideal_q,g*omega,c);_,Cunit,_=physics.mass_bias(ideal_q,g,c)
                learned_x,learned_v,_=physics.point_states(ideal_q,g*omega,g2*omega*omega,c)
                _,N,H=physics.point_states(ideal_q,g,g2,c)
                learned_a=np.asarray(model.euler_lagrange(parameters['state_RK4'],jnp.asarray(((learned_x-CENTER)/SCALE).ravel(),dtype=jnp.float32),
                                    jnp.asarray((learned_v/SCALE).ravel(),dtype=jnp.float32),jnp.asarray(run['action']))) .reshape(3,2)*SCALE
                learned_alpha=float(np.sum(N*(learned_a-H*omega**2))/np.sum(N*N))
                I=float(g@M@g);h=float(g@(M@g2+Cunit))
                examples[name]={'seed':1000,'timestamp_s':float(truth['time'][i]),'theta_rad':theta,'omega_rad_s':omega,
                    'I_eff_kg_m2':I,'h_velocity_coefficient':h,'gravity_eff_Nm':float(g@G),
                    'friction_eff_Nm':float(-g@physics.friction(g*omega,c)),'external_eff_Nm':float(g@physics.external(ideal_q,c)),
                    'motor_Nm':tau,'independent_alpha_rad_s2':alpha,'model_alpha_at_same_state_rad_s2':learned_alpha,
                    'matched_state_equivalent_motor_error_Nm':I*(learned_alpha-alpha),
                    'ideal_equation_residual_Nm':I*alpha+h*omega**2+float(g@G)-float(g@physics.friction(g*omega,c))-float(g@physics.external(ideal_q,c))-tau,
                    'source':'Independent ideal-constraint equation evaluated at the measured crank angle/speed; model queried at exactly the same state.'}
            if (index-999)%5==0:status('trajectory_evaluation',condition=name,completed=index-999,total=20)
        results[name]={label:{'run_count':20,'finite_runs':sum(row['finite'] for row in rows),
                      'complete_physical_runs':sum(row['geometry_valid_frames']==47 for row in rows),
                      'valid_physics_frames':sum(row['geometry_valid_frames'] for row in rows),'future_frames':940,
                      'motor_torque_error_rms_Nm':stats([row['motor_torque_error_rms_Nm'] for row in rows if row['geometry_valid_frames']==47]),
                      'geometry_residual_rms_m':stats([row.get('geometry_residual_rms_m') for row in rows]),'rows':rows} for label,rows in groups.items()}
        write(root/'evaluation_report.json',{'protocol':protocol,'torque_reference_Nm':torque_reference,'conditions':results,'checkpoint_sha256':hashes,
                'primary_evidence':'Equivalent motor-torque error on complete legal trajectories only; report geometry failures instead of scoring them as zero.',
                'state_control_scope':'State-supervised source model, exact prefix state, frozen weights; not video-learning performance.',
                'training_performed':False})
    write(root/'equation_examples.json',examples)
    run,_,_,_,_=load(root,'nominal',1000);slider,_,_,_,_=load(root,'crank_slider',1000)
    np.savez_compressed(root/'example_inputs.npz',nominal_rgb=np.rint(run['x']*255).astype(np.uint8),slider_rgb=np.rint(slider['x']*255).astype(np.uint8),time=run['t'])
    source_paths=[Path(__file__),Path(__file__).with_name('extended_mechanics.py'),Path(__file__).with_name('extended_dataset.py')]
    write(root/'execution_manifest.json',{'checkpoint_sha256':hashes,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
        'dataset_manifest_sha256':hashlib.sha256((root/'dataset_manifest.json').read_bytes()).hexdigest(),
        'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['jax','jaxlib','flax','numpy','scipy']},
        'devices':[str(d) for d in jax.devices()],'elapsed_seconds':time.monotonic()-start,'no_training':True})
    status('complete',evaluated_model_trajectories=320,conditions=8)


if __name__=='__main__':main()
