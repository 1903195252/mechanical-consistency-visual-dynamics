"""Bounded, restartable local production of 90 new held-out spatial assemblies."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

from .common import ROOT,OUT,CORPUS,OLD_CORPUS,CONDITIONS,NEW_ASSEMBLIES,sha,write

BLENDER=Path('/opt/blender-5.1.1-linux-x64/blender-launcher')


def connect():
    db=sqlite3.connect(str(CORPUS/'catalog.sqlite'),timeout=60)
    db.row_factory=sqlite3.Row
    return db


def update(identity,status,**fields):
    db=connect();fields.update(status=status)
    db.execute('UPDATE assemblies SET '+','.join(k+'=?' for k in fields)+' WHERE assembly_id=?',
               tuple(fields.values())+(identity,));db.commit();db.close()


def build(entry):
    from experiments.mechanism_corpus import stewart,delta,physics_stewart,physics_delta
    from experiments.mechanism_corpus.render_batch import within_reviewed_domain
    mechanism,physics=(stewart,physics_stewart) if entry['category']=='H-A' else (delta,physics_delta)
    folder=CORPUS/'assemblies'/entry['category']/entry['assembly_id']
    folder.mkdir(parents=True,exist_ok=True)
    assembly=mechanism.build(entry)
    review=OLD_CORPUS/'assemblies'/entry['category']/(entry['category']+'-001-P01')/'visual_review.json'
    within_reviewed_domain(assembly,json.loads(review.read_text()))
    prior=sqlite3.connect('file:'+str(OLD_CORPUS/'catalog.sqlite')+'?mode=ro',uri=True)
    assert not prior.execute('SELECT 1 FROM assemblies WHERE geometry_digest=?',(assembly['geometry_digest'],)).fetchone(), 'Duplicate original geometry'
    prior.close()
    db=connect()
    assert not db.execute('SELECT 1 FROM assemblies WHERE geometry_digest=? AND assembly_id!=?',(assembly['geometry_digest'],entry['assembly_id'])).fetchone(), 'Duplicate expanded geometry'
    db.close()
    write(folder/'assembly.json',assembly)
    write(folder/'assembly_validation.json',assembly['assembly_audit'])
    (folder/'model.xml').write_text(mechanism.mjcf(assembly['config'],assembly['bodies'])[0])
    update(entry['assembly_id'],'built',geometry_digest=assembly['geometry_digest'],manifest_path=str((folder/'assembly.json').relative_to(CORPUS)),last_error=None)
    for condition in CONDITIONS:
        directory=folder/'conditions'/condition
        report=directory/'physics_validation.json'
        old=json.loads(report.read_text()) if report.exists() else {}
        if old.get('passed') and old.get('assembly_physical_digest')==assembly['physical_digest'] and old.get('generator_source_digest')==physics.source_digest():continue
        physics.simulate(assembly,condition,directory)
    write(folder/'visual_review.json',dict(passed=True,mode='reviewed_template_with_instance_checks',
        assembly_physical_digest=assembly['physical_digest'],template_review=os.path.relpath(review,CORPUS),
        template_review_sha256=sha(review),parameter_domain_check=True,
        scope='Existing inspected template; independent full-trajectory solids, clearance, physics, framing and front-view acceptance still required.'))
    update(entry['assembly_id'],'physics_validated',last_error=None)
    return entry['assembly_id']


def render(entry):
    from experiments.mechanism_corpus.verify import verify_assembly
    folder=CORPUS/'assemblies'/entry['category']/entry['assembly_id']
    for script,extra in [('blender_render.py',['--views','front','--resume']),
                         ('audit_blender_solids.py',[]),('audit_blender_clearance.py',[])]:
        subprocess.run([str(BLENDER),'--background','--factory-startup','--python-exit-code','1','-t','4',
            '--python',str(ROOT/'experiments/mechanism_corpus'/script),'--','--assembly-dir',str(folder),*extra],
            cwd=ROOT,check=True)
    result=verify_assembly(entry,output=CORPUS,required_views=['front'])
    if not result['passed']:raise ValueError(result['reasons'])
    return result


def acceptance_status():
    registered={e['assembly_id'] for e in json.loads((OUT/'assembly_plan.json').read_text())['entries']}
    db=connect();rows=[dict(r) for r in db.execute('SELECT assembly_id,category,status,last_error FROM assemblies ORDER BY ordinal,category') if r['assembly_id'] in registered];db.close()
    return dict(total_new_assemblies=NEW_ASSEMBLIES,accepted_new_assemblies=sum(r['status']=='verified' for r in rows),
                physics_ready=sum(r['status'] in ('physics_validated','rendered','verified') for r in rows),
                failures=[r for r in rows if r['last_error']],assemblies=rows)


def guarded_build(entry):
    try:return build(entry)
    except Exception as error:
        update(entry['assembly_id'],'failed',last_error=str(error));raise


def run(args):
    (OUT/'logs').mkdir(exist_ok=True)
    for name in ('tmp','cache'):(OUT/name).mkdir(exist_ok=True)
    os.environ.update(TMPDIR=str(OUT/'tmp'),XDG_CACHE_HOME=str(OUT/'cache'),
        CUDA_CACHE_PATH=str(OUT/'cache/cuda'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    lock=(OUT/'production.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    entries=json.loads((OUT/'assembly_plan.json').read_text())['entries']
    entries.sort(key=lambda r:(r['ordinal'],r['category']))
    if args.limit:entries=entries[:args.limit]
    started=time.time();render_child=None;render_identity=None;attempted=set()
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        db=connect();states={r['assembly_id']:r['status'] for r in db.execute('SELECT * FROM assemblies')};db.close()
        pending={pool.submit(guarded_build,e):e for e in entries if states[e['assembly_id']] not in ('physics_validated','rendered','verified')}
        while True:
            for future in list(pending):
                if future.done():
                    entry=pending.pop(future)
                    try:future.result()
                    except Exception as error:print(json.dumps(dict(failed=entry['assembly_id'],error=str(error))),flush=True)
            if render_child is not None and render_child.poll() is not None:
                if render_child.returncode:
                    update(render_identity,'failed',last_error='Rendering/acceptance failed; see logs/'+render_identity+'.log')
                render_child=None
            status=acceptance_status();status.update(pid=os.getpid(),started_unix=started,updated_unix=time.time(),
                stage='preparing',active_render=render_identity if render_child else None,
                physics_workers=args.workers)
            write(OUT/'preparation_status.json',status)
            if args.stage=='all' and render_child is None:
                eligible={r['assembly_id'] for r in status['assemblies'] if r['status'] in ('physics_validated','rendered')}
                candidate=next((e for e in entries if e['assembly_id'] in eligible and e['assembly_id'] not in attempted),None)
                if candidate:
                    render_identity=candidate['assembly_id'];attempted.add(render_identity)
                    log=(OUT/'logs'/(render_identity+'.log')).open('ab',buffering=0)
                    render_child=subprocess.Popen([sys.executable,'-m','experiments.lingbot_expansion.production',
                        '--stage','render-one','--assembly',render_identity],cwd=ROOT,stdin=subprocess.DEVNULL,
                        stdout=log,stderr=subprocess.STDOUT)
                    log.close()
            if not pending and render_child is None:break
            time.sleep(3)
    status=acceptance_status();status.update(pid=os.getpid(),updated_unix=time.time(),started_unix=started,
        stage='complete' if status['accepted_new_assemblies']==NEW_ASSEMBLIES else 'physics_complete' if args.stage=='physics' and not status['failures'] else 'incomplete')
    write(OUT/'preparation_status.json',status)
    if status['stage']=='complete':
        from .prepare import publish
        publish()
    print(json.dumps({k:v for k,v in status.items() if k!='assemblies'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['all','physics','render-one'],default='all')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--assembly')
    a=p.parse_args()
    if a.stage=='render-one':
        db=connect();entry=dict(db.execute('SELECT * FROM assemblies WHERE assembly_id=?',(a.assembly,)).fetchone());db.close();render(entry)
    else:run(a)
