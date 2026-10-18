"""Resume full videos and audit each physical assembly before counting it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .catalog import ROOT,OUTPUT,connect,summary,write_json
from .verify import verify_assembly
from .naming import canonical_id

BLENDER=os.environ.get('BLENDER_BIN', 'blender')


def within_reviewed_domain(assembly,review):
    c=assembly['config'];domain=review['parameter_domain']
    if canonical_id(assembly['category'])!=canonical_id(review.get('category','S-A')):raise ValueError('Template review covers a different category')
    if review.get('reviewed_subtypes') and c.get('subtype') not in review['reviewed_subtypes']:
        raise ValueError('Unreviewed mechanical subtype')
    for key,expected in review.get('required_configuration',{}).items():
        if c.get(key)!=expected:raise ValueError('Template configuration revision differs: '+key)
    for key,bounds in domain.items():
        if key=='length_to_ground_ratios':
            for length,(lo,hi) in zip(c['lengths_m'],bounds):
                if not lo<=length/c['ground_length_m']<=hi:raise ValueError('Length outside reviewed family')
        elif isinstance(bounds[0],list):
            if len(c[key])!=len(bounds):raise ValueError('Wrong number of parameter values')
            for value,(lo,hi) in zip(c[key],bounds):
                if not lo<=value<=hi:raise ValueError('Length outside reviewed family')
        elif not bounds[0]<=c[key]<=bounds[1]:raise ValueError('Outside reviewed parameter domain: '+key)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--category',default='S-A')
    parser.add_argument('--start',type=int,default=2)
    parser.add_argument('--count',type=int,default=119)
    parser.add_argument('--template-review',type=Path,
                        default=OUTPUT/'assemblies/S-A/S-A-001-P01/visual_review.json')
    args=parser.parse_args();review_path=args.template_review.resolve()
    review=json.loads(review_path.read_text())
    if not review.get('passed') or review.get('mode')!='visual_inspection':
        raise ValueError('Missing actual template inspection')
    db=connect()
    rows=[dict(r) for r in db.execute('SELECT * FROM assemblies WHERE category=? AND ordinal>=? AND ordinal<? ORDER BY ordinal',
                                    (args.category,args.start,args.start+args.count))]
    db.close()
    if len(rows)!=args.count:raise ValueError('Requested range differs from catalog')
    job=OUTPUT/'jobs'/('%s_%03d_%03d'%(args.category,args.start,args.start+args.count-1))
    job.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy()
    env.update(TMPDIR=str(ROOT/'output/fourbar/.tmp'),XDG_CACHE_HOME=str(ROOT/'output/fourbar/.cache'),
               CUDA_CACHE_PATH=str(ROOT/'output/fourbar/.cache/cuda'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    state={'pid':os.getpid(),'category':args.category,'requested':len(rows),'completed_in_job':0,
           'status':'running','started_unix_s':time.time(),'current':None}
    try:
        for entry in rows:
            state['current']=entry['assembly_id'];write_json(job/'status.json',state)
            folder=OUTPUT/'assemblies'/entry['category']/entry['assembly_id']
            if entry['status']=='verified':
                state['completed_in_job']+=1;continue
            assembly=json.loads((folder/'assembly.json').read_text())
            within_reviewed_domain(assembly,review)
            with (job/(entry['assembly_id']+'.log')).open('a') as log:
                for script,extra in [('blender_render.py',['--resume']),('audit_blender_solids.py',[]),('audit_blender_clearance.py',[])]:
                    subprocess.run([str(BLENDER),'--background','--factory-startup','--python-exit-code','1','-t','4','--python',
                                    str(ROOT/'experiments/mechanism_corpus'/script),'--','--assembly-dir',str(folder),*extra],
                                   cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            if (folder/'visual_review.json').resolve()!=review_path:
                write_json(folder/'visual_review.json',{
                    'passed':True,'mode':'reviewed_template_with_instance_checks',
                    'assembly_physical_digest':assembly['physical_digest'],
                    'template_review':str(review_path.relative_to(OUTPUT)),
                    'template_review_sha256':hashlib.sha256(review_path.read_bytes()).hexdigest(),
                    'parameter_domain_check':True,
                    'scope':'Template visual inspection inherited within the stated geometry domain; this instance requires independent solid, pose, framing, physics and video checks. No individual manual visual review is claimed.'})
            result=verify_assembly(entry)
            if not result['passed']:raise RuntimeError(json.dumps(result))
            state['completed_in_job']+=1;write_json(job/'status.json',state)
            print(json.dumps({'verified':entry['assembly_id'],'completed_in_job':state['completed_in_job'],
                              'requested':len(rows),'elapsed_s':time.time()-state['started_unix_s']}),flush=True)
        state['status']='complete'
    except Exception as error:
        state['status']='failed';state['error']=str(error)
        raise
    finally:
        state['updated_unix_s']=time.time();write_json(job/'status.json',state)


if __name__=='__main__':main()
