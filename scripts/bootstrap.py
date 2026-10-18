"""Create isolated environments and fetch the registered model sources."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(args, dry=False):
    import shlex
    print('+ ' + ' '.join(shlex.quote(str(x)) for x in args), flush=True)
    if not dry:
        subprocess.run([str(x) for x in args], cwd=ROOT, check=True)


def artifact_inputs(directory):
    directory=Path(directory).resolve()
    if any((directory/name).is_file() for name in ('package.json','artifact_manifest.json')):return [directory]
    collection=directory/'artifact_collection.json'
    if collection.is_file():
        result=[]
        for item in json.loads(collection.read_text())['packages']:
            if not item.get('default',True):continue
            package=(directory/item['path']).resolve()
            if directory not in package.parents or not any((package/name).is_file() for name in ('package.json','artifact_manifest.json')):
                raise ValueError('Invalid artifact collection entry: '+item['path'])
            result.append(package)
        return result
    return sorted(p for p in directory.iterdir() if p.is_file() and p.name.endswith(('.tar','.tar.gz','.tgz')))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=['audit','planar','vjepa','lingbot','spatial','all'], default='audit')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--download-models', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-checks', action='store_true')
    p.add_argument('--artifact-dir', type=Path, help='Import a manifested artifact tree, or .tar/.tar.gz/.tgz bundles')
    a = p.parse_args()
    if sys.platform != 'linux':
        raise SystemExit('The registered training launchers target Linux.')
    cache = ROOT/'output/runtime/install'
    if not a.dry_run:
        cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('UV_CACHE_DIR', str(cache/'uv-cache'))
    os.environ.setdefault('UV_PYTHON_INSTALL_DIR', str(cache/'python'))
    os.environ.setdefault('PIP_CACHE_DIR', str(cache/'pip-cache'))
    uv = shutil.which('uv')
    if uv is None:
        uv = str(ROOT/'.bootstrap/bin/uv')
        if not Path(uv).exists():
            run([sys.executable,'-m','venv',ROOT/'.bootstrap'],a.dry_run)
            run([ROOT/'.bootstrap/bin/python','-m','pip','install','uv'],a.dry_run)
    selected = ['audit']
    if a.profile in ('planar','all'):
        selected.append('planar')
    if a.profile in ('vjepa','lingbot','spatial','all'):
        selected.append('spatial')
    for name in selected:
        version = '3.10' if name == 'audit' else '3.12'
        env = ROOT/('.venv-'+name)
        python = env/'bin/python'
        if not python.exists():
            run([uv,'venv','--python',version,env],a.dry_run)
        req = {'audit':'audit.txt','planar':'oscillators.txt','spatial':'lingbot.txt'}[name]

        if name == 'planar' and a.device == 'cuda':
            run([uv,'pip','install','--python',python,'jax[cuda12]==0.11.2'],a.dry_run)
        if name == 'spatial':
            index = 'https://download.pytorch.org/whl/'+('cu128' if a.device == 'cuda' else 'cpu')
            run([uv,'pip','install','--python',python,'--index-url',index,
                 'torch==2.9.0','torchvision==0.24.0','torchaudio==2.9.0'],a.dry_run)
        run([uv,'pip','install','--python',python,'-r',ROOT/'requirements'/req],a.dry_run)
        run([uv,'pip','check','--python',python],a.dry_run)
    run([sys.executable,'scripts/restore_metadata.py'],a.dry_run)
    models = []
    if 'planar' in selected:
        models += ['keycld','con']
    if 'spatial' in selected:
        models += ['vjepa']
    if a.profile in ('lingbot','spatial','all'):
        models += ['lingbot']
    if models and not shutil.which('git') and not a.dry_run:
        raise SystemExit('Install Git before fetching model sources.')
    for model in models:
        run([sys.executable,'scripts/setup_upstream.py',model],a.dry_run)
    if a.artifact_dir:
        bundles=artifact_inputs(a.artifact_dir)
        if not bundles:
            raise SystemExit('No artifact bundles found in '+str(a.artifact_dir))
        run([sys.executable,'scripts/artifacts.py','import',*bundles],a.dry_run)
    if a.download_models:
        for model in ('vjepa','lingbot'):
            if model in models:
                run([sys.executable,'scripts/download_models.py',model],a.dry_run)
    if not a.skip_checks:
        run(['bash','run.sh','verify'],a.dry_run)
    if not a.dry_run:
        target = ROOT/'output/runtime/environments.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(dict(environments=selected,device=a.device,upstream_models=models),indent=2)+'\n')
        print('Ready. See docs/reproduction.md for experiment commands.')


if __name__ == '__main__':
    main()
