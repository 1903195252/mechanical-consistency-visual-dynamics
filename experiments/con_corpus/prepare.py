"""Bind CON to the frozen planar dataset and its independent calibration."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/con_corpus'


def main():
    source = ROOT / 'output/keycld_corpus'
    protocol = json.loads(Path(__file__).with_name('protocol.json').read_text())
    manifest = (source / 'dataset_manifest.json').read_bytes()
    digest = hashlib.sha256(manifest).hexdigest()
    if digest != protocol['dataset_manifest_sha256']:
        raise ValueError('Dataset differs from the registered planar experiment')
    if not (ROOT / 'official/src/models/neural_odes/con_iae.py').is_file():
        raise FileNotFoundError('Run python scripts/setup_upstream.py con first')
    OUT.mkdir(parents=True, exist_ok=True)
    files = {
        'dataset_manifest.json': manifest,
        'protocol.json': Path(__file__).with_name('protocol.json').read_bytes(),
        'calibration.json': (source / 'calibration.json').read_bytes(),
    }
    for name, content in files.items():
        target = OUT / name
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError('Existing experiment differs: ' + str(target))
        target.write_bytes(content)
    extension = OUT / 'dynamics_extension'
    extension.mkdir(exist_ok=True)
    for name, config in [('protocol.json', ROOT/'configs/con_integrators.json'),
                         ('solver_protocol.json', ROOT/'results/planar/extension/solver_protocol.json')]:
        target = extension / name
        if target.exists() and target.read_bytes() != config.read_bytes():
            raise FileExistsError('Existing extension differs: ' + str(target))
        target.write_bytes(config.read_bytes())
    print(json.dumps({'counts': json.loads(manifest)['counts'], 'dataset_sha256': digest}))


if __name__ == '__main__':
    main()
