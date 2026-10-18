#!/usr/bin/env python3
"""Fetch a pinned upstream revision and prepare experiment import adapters."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = json.loads((ROOT / 'configs/upstream.json').read_text())


def git(source, *args):
    return subprocess.check_output(['git', '-C', str(source), *args])


def archive_files(source, commit):
    result = {}
    with tarfile.open(fileobj=io.BytesIO(git(source, 'archive', commit))) as archive:
        for item in archive.getmembers():
            name = Path(item.name)
            if item.isfile() and not name.is_absolute() and '..' not in name.parts:
                result[item.name] = archive.extractfile(item).read()
    return result


def write_tree(destination, files):
    for name, data in files.items():
        target = destination / name
        if target.exists() and target.read_bytes() != data:
            raise FileExistsError('Refusing to overwrite modified upstream file: ' + str(target))
    for name, data in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def setup(name, source=None):
    spec = MODELS[name]
    checkout = ROOT / spec['path']
    if source is None:
        if not checkout.exists():
            checkout.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['git', 'init', str(checkout)], check=True)
            subprocess.run(['git', '-C', str(checkout), 'remote', 'add', 'origin', spec['url']], check=True)
            subprocess.run(['git', '-C', str(checkout), 'fetch', '--depth', '1', 'origin', spec['commit']], check=True)
        elif name != 'con' and git(checkout, 'status', '--porcelain').strip():
            raise RuntimeError('Upstream checkout has local changes: ' + str(checkout))
        if name != 'con':
            subprocess.run(['git', '-C', str(checkout), 'checkout', '--detach', spec['commit']], check=True)
        source = checkout
    source = Path(source).resolve()
    commit = git(source, 'rev-parse', spec['commit'] + '^{commit}').decode().strip()
    if commit != spec['commit']:
        raise ValueError('Upstream revision differs')
    originals = archive_files(source, commit)
    if name == 'con':
        files = {k: v for k, v in originals.items()
                 if ((k.startswith('src/') and k.endswith('.py') and Path(k).stem.lower() != 'con')
                     or k in ('LICENSE', 'README.md'))}
        initializers = {
            'src/__init__.py': '', 'src/models/__init__.py': '',
            'src/models/autoencoders/__init__.py': 'from .vae import VAE\n',
            'src/models/neural_odes/__init__.py': 'from .con_iae import ConIaeOde\n',
            'src/models/discrete_forward_dynamics/__init__.py': 'from .con_iae_cfa import DiscreteConIaeCfaDynamics\n',
            'src/tasks/__init__.py': '', 'src/training/__init__.py': '',
        }
        files.update({k: v.encode() for k, v in initializers.items()})
        destination = ROOT / 'official'
    elif name == 'keycld':
        files = {k: v for k, v in originals.items()
                 if k.startswith('keycld/') or Path(k).name.upper().startswith('LICENSE')}
        util = files['keycld/util.py'].decode()
        util = util.replace('import torch\n', 'try:\n    import torch\nexcept ImportError:\n    torch = None\n')
        util = util.replace('class NumpyLoader(torch.utils.data.DataLoader):',
                            'class NumpyLoader(torch.utils.data.DataLoader if torch is not None else object):')
        marker = '                 timeout=0, worker_init_fn=None):\n'
        if marker not in util:
            raise ValueError('Unexpected upstream DataLoader signature')
        util = util.replace(marker, marker + '        if torch is None:\n            raise ImportError("PyTorch is required for the training DataLoader.")\n')
        files['keycld/util.py'] = util.encode()
        files['keycld/__init__.py'] = b''
        destination = ROOT / 'output/keycld_corpus/runtime'
    else:
        files = originals
        destination = checkout
    write_tree(destination, files)
    record = {'model': name, 'url': spec['url'], 'commit': commit,
              'path': str(destination.relative_to(ROOT)),
              'files_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    target = ROOT / 'output/upstream' / (name + '_manifest.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'model': name, 'commit': commit, 'path': record['path']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', choices=sorted(MODELS))
    parser.add_argument('--source', type=Path, help='Existing clone containing the pinned commit')
    args = parser.parse_args()
    setup(args.model, args.source)
