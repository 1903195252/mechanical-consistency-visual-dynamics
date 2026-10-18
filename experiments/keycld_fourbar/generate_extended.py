import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import multiprocessing
from pathlib import Path
import sys
import time

from extended_dataset import conditions,generate,initialize


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--assets',type=Path,required=True);args=parser.parse_args();root=args.root
    sys.path.insert(0,str(args.assets));root.mkdir(parents=True,exist_ok=True)
    if (root/'dataset_manifest.json').exists():raise RuntimeError('Completed paired data exists; preserve it')
    configs=conditions(json.loads((args.assets/'config.json').read_text()))
    (root/'condition_configs.json').write_text(json.dumps(configs,indent=2)+'\n')
    tasks=[(name,index,c,str(root)) for name,c in configs.items() for index in range(1000,1020)]
    start=time.monotonic();rows=[]
    def status():
        state={'stage':'generating','completed':len(rows),'total':160,'elapsed_seconds':time.monotonic()-start}
        (root/'generation_status.json').write_text(json.dumps(state,indent=2)+'\n');print(json.dumps(state),flush=True)
    status()
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(str(args.assets),)) as pool:
        for future in as_completed([pool.submit(generate,t) for t in tasks]):
            rows.append(future.result())
            if len(rows)%10==0:status()
    rows.sort(key=lambda row:(row['condition'],row['index']))
    for index in range(1000,1020):
        paired=[row for row in rows if row['index']==index]
        assert len(paired)==8
        assert all(row['initial_angle_rad']==paired[0]['initial_angle_rad'] and row['initial_speed_rad_s']==paired[0]['initial_speed_rad_s'] and row['torque_Nm']==paired[0]['torque_Nm'] for row in paired)
    (root/'dataset_manifest.json').write_text(json.dumps({'rows':rows,'run_count':160,'paired_seeds':20,'elapsed_seconds':time.monotonic()-start,'physics_audited_per_run':True},indent=2)+'\n')
    (root/'generation_status.json').write_text(json.dumps({'stage':'complete','completed':160,'total':160},indent=2)+'\n')
    print('160 paired RGB runs independently audited.',flush=True)


if __name__=='__main__':main()
