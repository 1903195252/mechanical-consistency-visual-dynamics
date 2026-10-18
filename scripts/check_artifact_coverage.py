"""Check current experiment case coverage in an imported artifact directory."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def check(directory, profile='all'):
    root=Path(directory);errors=[];counts={}
    locations=None
    if (root/'artifact_collection.json').is_file():
        locations={}
        for item in json.loads((root/'artifact_collection.json').read_text())['packages']:
            package=(root/item['path']).resolve()
            if root.resolve() not in package.parents:raise ValueError('External package path')
            descriptor=package/('package.json' if (package/'package.json').is_file() else 'artifact_manifest.json')
            for entry in json.loads(descriptor.read_text())['files']:
                if entry['path'] in locations:raise ValueError('Duplicate runtime path: '+entry['path'])
                locations[entry['path']]=package/entry['path']
    def require(relative):
        path=(locations or {}).get(relative,root/relative)
        if not path.is_file():errors.append(str(relative))
        return path
    if profile in ('all','planar'):
        rows=json.loads((ROOT/'data/manifests/planar.json').read_text())['records']
        cases={r['case_id'] for r in rows if r['split']=='test'}
        runs=[]
        for seed in (42,43,44):
            runs.extend('output/keycld_corpus/runs/%s_seed%d'%(arm,seed) for arm in ('video_euler','state_euler','state_rk4'))
            runs.extend('output/con_corpus/runs/%s_seed%d'%(arm,seed) for arm in ('video_con_iae','state_con_iae'))
            runs.extend('output/con_corpus/dynamics_extension/runs/%s_con_iae_seed%d'%(solver,seed) for solver in ('state_euler','state_rk4'))
            runs.extend('output/planar_resolution/runs/%s_256_seed%d'%(model,seed) for model in ('keycld','con'))
        for run in runs:
            require(run+'/params.p');require(run+'/training_report.json')
            if locations is None:
                present={p.stem for p in (root/run/'predictions/test').glob('*.npz')}
            else:
                prefix=run+'/predictions/test/'
                present={Path(p).stem for p in locations if p.startswith(prefix) and p.endswith('.npz') and '/' not in p[len(prefix):] and locations[p].is_file()}
            if present!=cases:errors.append(run+': test case identities differ')
        counts['planar_final_checkpoints']=len(runs)
        counts['planar_test_predictions']=len(runs)*len(cases)
    if profile in ('all','spatial','vjepa','lingbot'):
        rows=json.loads((ROOT/'data/manifests/spatial_lingbot.json').read_text())['videos']
        rows=[r for r in rows if r['split']=='test' and r['view']=='front']
        if profile in ('all','spatial','vjepa'):
            for seed in (42,43,44):
                base='output/vjepa_expansion/seed_%d'%seed
                for family in ('H-A','H-B'):
                    for component in ('predictor','readout'):require(base+'/'+family+'_'+component+'.pt')
                for r in rows:
                    require(base+'/forecast_features/'+r['video_id']+'.npy')
                    for branch in ('native','calibrated','reference_internal'):require(base+'/'+branch+'/cases/'+r['video_id']+'.json')
            for r in rows:require('output/vjepa_expansion/reference/calibrated/cases/'+r['video_id']+'.json')
            counts.update(vjepa_predictors=6,vjepa_feedback_heads=6,vjepa_forecasts=len(rows)*3)
        if profile in ('all','spatial','lingbot'):
            for seed in (42,43,44):
                base='output/lingbot_expansion/runs/seed%d'%seed
                require(base+'/checkpoints/checkpoint_step_50/transformer/diffusion_pytorch_model.safetensors')
                require(base+'/checkpoints/checkpoint_step_50/transformer/config.json')
                require(base+'/checkpoint_0050.json')
                for step in (0,50):
                    run=base+'/step%04d'%step;require(run+'/reading_complete.json')
                    for r in rows:
                        key=r['video_id'];require(run+'/predictions/'+key+'_rgb.npz');require(run+'/predictions/'+key+'.json')
                        require(run+'/cases/'+key+'.json')
                        readout=run+'/readout/predictions/'+r['category']+'/'+key
                        marker=require(readout+'.json')
                        if marker.exists() and json.loads(marker.read_text())['finite']:require(readout+'.npz')
            base='output/lingbot_expansion/reference';require(base+'/reading_complete.json')
            for r in rows:
                require(base+'/rgb/'+r['video_id']+'.npz');require(base+'/cases/'+r['video_id']+'.json')
                require(base+'/readout/predictions/'+r['category']+'/'+r['video_id']+'.npz')
            counts.update(lingbot_adapted_checkpoints=3,lingbot_videos=len(rows)*6,reference_videos=len(rows))
    return dict(complete=not errors,counts=counts,missing_or_conflicting=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path,nargs='?',default=ROOT)
    p.add_argument('--profile',choices=['all','planar','spatial','vjepa','lingbot'],default='all');p.add_argument('--json',type=Path)
    a=p.parse_args();report=check(a.directory,a.profile)
    if a.json:a.json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));raise SystemExit(0 if report['complete'] else 1)


if __name__=='__main__':main()
