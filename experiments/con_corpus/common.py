"""Shared exact upstream model and hash-bound data contract."""
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'official'))
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from flax import linen as nn
from src.models.autoencoders.vae import VAE
from src.models.neural_odes.con_iae import ConIaeOde
from src.models.dynamics_autoencoder import DynamicsAutoencoder

OUT = ROOT / 'output/con_corpus'
DATA = ROOT / 'output/keycld_corpus'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    with tmp.open('wb') as handle:
        pickle.dump(jax.device_get(obj), handle, protocol=4)
    tmp.replace(path)


def setup():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    manifest = json.loads((OUT / 'dataset_manifest.json').read_text())
    assert sha(OUT / 'dataset_manifest.json') == protocol['dataset_manifest_sha256']
    return protocol, manifest


def model():
    return DynamicsAutoencoder(
        autoencoder=VAE(img_shape=(64, 64, 3), latent_dim=8, norm_layer=nn.LayerNorm),
        dynamics=ConIaeOde(latent_dim=8, input_dim=14, num_layers=5, hidden_dim=30))


def state_model():
    return ConIaeOde(latent_dim=8, input_dim=14, num_layers=5, hidden_dim=30)


def data(record, state=False, prefix=False):
    field = 'inputs' if prefix else 'observations'
    path = DATA / record[field]
    assert sha(path) == record[field + '_sha256'], 'Input changed: ' + record['case_id']
    import numpy as np
    with np.load(path) as z:
        images = z['x'].astype(np.float32) / 127.5 - 1
        action = np.r_[z['action'], z['geometry']].astype(np.float32)
    scales = np.array([5, 12, 12, 1, 1] + [1] * 9, dtype=np.float32)
    result = {'rendering_ts': images, 'tau': action / scales}
    if state:
        path = DATA / record['state_labels']
        assert sha(path) == record['state_labels_sha256']
        with np.load(path) as z:
            count = 3 if prefix else 50
            result['points'] = np.pad(z['x'][:count].reshape(count, 6), ((0, 0), (0, 2))).astype(np.float32)
            result['velocity'] = np.pad(z['v'][:count].reshape(count, 6), ((0, 0), (0, 2))).astype(np.float32)
    return result


def load_params(folder):
    report = json.loads((folder / 'training_report.json').read_text())
    assert report['epochs'] == 50 and sha(folder / 'params.p') == report['checkpoint_sha256']
    with (folder / 'params.p').open('rb') as handle:
        return pickle.load(handle), report
