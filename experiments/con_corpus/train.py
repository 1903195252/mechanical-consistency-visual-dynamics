"""Full upstream CON video objective; separately labeled privileged-state control."""
import argparse
import json
import math
from pathlib import Path
import pickle
import signal
import time

from .common import OUT, data, jax, jnp, model, state_model, setup, sha, write, save
import numpy as np
import optax
from diffrax import Dopri5, ODETerm, SaveAt, diffeqsolve
from src.tasks.dynamics_autoencoder import task_factory
from src.training.optim import create_learning_rate_fn

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)


def state_objective(parameters, batch, weights):
    network = state_model()
    def solve(x, v, command):
        initial = jnp.r_[x[1], v[1]].astype(jnp.float64)
        fun = lambda t, y, u: network.apply({'params': parameters}, y, u)
        result = diffeqsolve(ODETerm(fun), Dopri5(), t0=1/30, t1=49/30, dt0=.01,
                            y0=initial, args=command, saveat=SaveAt(ts=jnp.arange(1, 50)/30), max_steps=170)
        return result.ys[:, :8].astype(jnp.float32)
    history = jax.vmap(solve)(batch['points'], batch['velocity'], batch['tau'])
    mse = jnp.mean((history - batch['points'][:, 1:]) ** 2)
    reconstructed = jax.vmap(lambda u: network.apply({'params': parameters}, u, method=network.autoencode_input))(batch['tau'])
    rec = jnp.mean((reconstructed - batch['tau']) ** 2)
    return weights['mse_z'] * mse + weights['mse_tau_rec'] * rec, {'mse_z': mse, 'mse_tau_rec': rec}


def run(arm, seed, smoke=False):
    protocol, manifest = setup(); cfg = protocol['training']
    assert arm in cfg['arms'] and seed in cfg['seeds']
    folder = OUT / ('preflight' if smoke else 'runs') / f'{arm}_seed{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    if not smoke and (folder / 'training_report.json').exists():
        return
    started = time.monotonic(); privileged = arm == 'state_con_iae'
    identity = {'protocol': sha(OUT/'protocol.json'), 'data': sha(OUT/'dataset_manifest.json'),
                'train_source': sha(Path(__file__)), 'common_source': sha(Path(__file__).with_name('common.py'))}
    def status(stage, **values):
        write(folder/'status.json', {'stage': stage, 'arm': arm, 'seed': seed,
                                    'elapsed_s': time.monotonic()-started, **values})
    status('loading')
    records = [r for r in manifest['records'] if r['split'] == 'train']
    # No validation, calibration or test sequences are opened during training.
    if smoke:
        records = records[:cfg['batch_size']]
    examples = [data(r, state=privileged) for r in records]
    array = {k: np.stack([x[k] for x in examples]) for k in examples[0] if not (privileged and k == 'rendering_ts')}
    del examples
    generator = np.random.RandomState(seed); rng = jax.random.PRNGKey(seed)
    network = model(); batch_size = cfg['batch_size']
    steps_per_epoch = math.ceil(manifest['counts']['train']/batch_size)
    schedule = create_learning_rate_fn(cfg['epochs'], steps_per_epoch, cfg['base_lr'], warmup_epochs=cfg['warmup_epochs'])
    optimizer = optax.adamw(schedule, weight_decay=cfg['weight_decay'])
    if privileged:
        component=state_model()
        parameters = component.init(rng, jnp.zeros(16), jnp.zeros(14), method=component.forward_all_layers)['params']
        objective = lambda p, b, key: state_objective(p, b, cfg['loss_weights'])
    else:
        parameters = network.init(rng, jnp.zeros((1,64,64,3)), method=network.forward_all_layers)['params']
        task, _ = task_factory('planar_closed_chain', network, jnp.arange(50)/30, .01,
                               loss_weights=cfg['loss_weights'], ae_type='beta_vae',
                               solver=Dopri5(), latent_velocity_source='image-space-finite-differences')
        def objective(p, b, key):
            loss, preds = task.loss_fn(b, p, rng=key, training=True)
            metrics = task.compute_metrics_fn(b, preds)
            return loss, {k: v for k, v in metrics.items() if k.startswith('mse_')}
    opt_state = optimizer.init(parameters)
    @jax.jit
    def step(p, opt, b, key):
        (loss, aux), grads = jax.value_and_grad(objective, has_aux=True)(p, b, key)
        updates, opt = optimizer.update(grads, opt, p)
        return optax.apply_updates(p, updates), opt, loss, aux
    checkpoint = folder/'resume.p'; history = []; epoch = 0; cursor = 0; order = None; updates = 0
    epoch_losses = []; samples = 0
    if checkpoint.exists() and not smoke:
        with checkpoint.open('rb') as handle:
            previous = pickle.load(handle)
        assert previous['identity'] == identity, 'Source or frozen inputs changed since checkpoint'
        parameters, opt_state, rng = previous['params'], previous['opt_state'], previous['rng']
        generator.set_state(previous['numpy_rng'])
        epoch, cursor, order, updates = [previous[k] for k in ['epoch','cursor','order','updates']]
        history, epoch_losses, samples = previous['history'], previous['epoch_losses'], previous['samples']
    parameter_count = sum(x.size for x in jax.tree.leaves(parameters))
    def checkpoint_now():
        save(checkpoint, {'params': parameters, 'opt_state': opt_state, 'rng': rng,
                          'numpy_rng': generator.get_state(), 'epoch': epoch, 'cursor': cursor,
                          'order': order, 'updates': updates, 'history': history,
                          'epoch_losses': epoch_losses, 'samples': samples, 'identity': identity})
    status('compiling', updates=updates, parameters=parameter_count)
    for current_epoch in range(epoch, cfg['epochs']):
        epoch = current_epoch
        if order is None:
            order = generator.permutation(len(records)); cursor = 0; epoch_losses = []
        while cursor < len(order):
            if STOP:
                checkpoint_now(); status('paused_at_budget', epoch=epoch+1, updates=updates)
                return
            indices = order[cursor:cursor+batch_size]
            batch = {k: jnp.asarray(v[indices]) for k,v in array.items()}
            rng, key = jax.random.split(rng)
            tick = time.monotonic()
            parameters, opt_state, loss, aux = step(parameters, opt_state, batch, key)
            loss = float(loss)
            if not np.isfinite(loss):
                status('failed', reason='Non-finite full objective', epoch=epoch+1, updates=updates)
                raise RuntimeError('Nonfinite objective; no sequences omitted and no surrogate model used')
            updates += 1; samples += len(indices); cursor += len(indices); epoch_losses.append(loss)
            status('training', epoch=epoch+1, epochs=cfg['epochs'], updates=updates,
                   target_updates=steps_per_epoch*cfg['epochs'], batch=math.ceil(cursor/batch_size),
                   loss=loss, last_step_s=time.monotonic()-tick, samples_seen=samples, parameters=parameter_count)
            if smoke:
                write(folder/'smoke.json', {'passed': True, 'parameters': parameter_count,
                      'loss': loss, 'metrics': {k:float(v) for k,v in aux.items()},
                      'step_s': time.monotonic()-tick, 'devices': [str(d) for d in jax.devices()],
                      'scientific_result': False, 'official_full_loss_used': not privileged})
                return
            if updates % 20 == 0:
                checkpoint_now()
        history.append({'epoch': epoch+1, 'updates': updates, 'mean_batch_loss': float(np.mean(epoch_losses)), 'samples_seen': samples})
        write(folder/'history.json', history)
        print(json.dumps(history[-1]), flush=True)
        epoch = current_epoch+1; order = None; cursor = 0; epoch_losses = []; checkpoint_now()
    save(folder/'params.p', parameters)
    write(folder/'training_report.json', {'arm':arm, 'seed':seed, 'epochs':cfg['epochs'], 'updates':updates,
          'expected_updates':steps_per_epoch*cfg['epochs'], 'samples_seen':samples,
          'parameters':parameter_count, 'checkpoint_sha256':sha(folder/'params.p'),
          'identity':identity, 'privileged_state':privileged, 'elapsed_s':time.monotonic()-started,
          'devices':[str(d) for d in jax.devices()], 'full_official_video_objective':not privileged})
    status('training_complete', epoch=cfg['epochs'], updates=updates)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',required=True);parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    run(args.arm,args.seed,args.smoke)
