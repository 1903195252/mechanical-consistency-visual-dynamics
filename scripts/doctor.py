"""Check whether the selected experiment's required local artifacts are present."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def required(profile):
    paths=set()
    if profile in ('planar','all'):
        for root,manifest in [('output/keycld_corpus','planar.json'),('output/planar_resolution/256','planar_256.json')]:
            for row in json.loads((ROOT/'data/manifests'/manifest).read_text())['records']:
                paths.update(str(Path(root)/row[field]) for field in ('observations','inputs','state_labels'))
                paths.update(row[field] for field in ('physical_parameters','physics_reference'))
                if 'images' in row:paths.add(str(Path(root)/row['images']))
    if profile in ('spatial','all'):
        rows=json.loads((ROOT/'data/manifests/spatial_lingbot.json').read_text())['videos']
        for row in rows:
            paths.update(row[field] for field in ('video','public_inputs','commands') if field in row)
            paths.update(str(Path(row['evaluator_directory'])/name) for name in ('model.xml','physics.npz','physical_parameters.json','controller_inputs.json'))
    return sorted(paths)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['planar','spatial','all'],default='all')
    p.add_argument('--weights',action='store_true',help='Also check the downloaded official weights')
    a=p.parse_args();paths=required(a.profile)
    missing=[name for name in paths if not (ROOT/name).is_file()]
    print('Dataset files: %d / %d present'%(len(paths)-len(missing),len(paths)))
    for name in missing[:20]:print('Missing: '+name)
    if len(missing)>20:print('... %d more missing files'%(len(missing)-20))
    if a.weights:
        from download_models import model_files,valid
        for model in ('vjepa','lingbot'):
            for item in model_files(model):
                if not valid(ROOT/item['path'],item):
                    missing.append(item['path']);print('Missing or invalid weight: '+item['path'])
    if missing:raise SystemExit(1)
    print('Required paths are present. Bundle import and model download perform checksum verification.')


if __name__=='__main__':main()
