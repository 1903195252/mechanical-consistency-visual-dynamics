"""Verify a frozen Delta image instrument against paired SI controls."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from scipy.signal import savgol_filter
from .common import ROOT,write,sha
from .delta_instrument import DIRECTORY,CALIBRATION,rows,reference,FEATURES,decode
from .delta_optical import OPTICAL,track_rgb,platform_plane,images
from .delta_fusion import positions


def case(row):
    from experiments.model_expansion.evaluate_spatial import Audit
    selection=json.loads((OPTICAL/'fusion_selection.json').read_text())
    kind,window=selection['selected'].split('/');window=int(window)
    assert kind=='relative_depth'
    role=row['calibration_role'];destination=OPTICAL/role/(row['video_id']+'.json')
    if destination.exists():return json.loads(destination.read_text())
    if role=='calibration':
        with np.load(OPTICAL/'fusion_development'/(row['video_id']+'.npz')) as d:p=d[kind]
    else:
        track_path=OPTICAL/role/(row['video_id']+'_tracks.npz');track_path.parent.mkdir(parents=True,exist_ok=True)
        if track_path.exists():
            with np.load(track_path) as d:tracks={k:d[k] for k in d.files}
        else:
            tracks=track_rgb(row,images(row));np.savez_compressed(track_path,**tracks)
        plane,_=platform_plane(row,tracks)
        # Nominal calibration state is static CAD; no future labels are opened.
        from experiments.closed_chain_eval.geometry_spatial_reader import nominal
        with np.load(DIRECTORY/'relative_head.npz') as d:matrix,head_window=d['matrix'],int(d['window'])
        relative=decode(np.load(FEATURES/(row['video_id']+'.npy')),nominal(row),matrix,head_window)
        camera=json.loads((ROOT/row['public_inputs']).read_text())['observed_camera_frames'][0]
        direction=np.asarray(camera['R_world_to_camera'])[2]
        p=plane+np.outer((relative[:,:3]-plane)@direction,direction)
    p=savgol_filter(p,window,3,axis=0) if window>1 else p
    audit=Audit(row)
    # Full measurements are opened after the image reconstruction is complete.
    model,truth,v,a=reference(row);t=np.arange(120)/30
    q=np.broadcast_to(model.qpos0,(len(p),model.nq)).copy();q[:,:3]=p
    projected=np.stack([audit.project(pose) for pose in q])
    updated=audit.measure(t,projected,truth)
    with np.load(CALIBRATION/'readouts/geometry_spatial_256'/(row['video_id']+'.npz')) as d:old=d['qpos'].astype(float)
    baseline=audit.measure(t,np.stack([audit.project(pose) for pose in old]),truth)
    exact=audit.measure(t,truth,truth,(v,a));numerical=audit.measure(t,truth,truth)
    result=dict(case_id=row['video_id'],assembly_id=row['assembly_id'],role=role,
        baseline=baseline,updated=updated,exact=exact,numerical=numerical,
        platform_position_rmse_mm=float(np.sqrt(np.mean(np.sum((p[3:]-truth[3:,:3])**2,axis=1))))*1000,
        selection_sha256=sha(OPTICAL/'fusion_selection.json'),source_sha256=sha(__file__),
        interpretation='CAD reconstruction intervention; closure is imposed, not credited to the video model',
        test_labels_used_for_selection=False,engineering_limits_changed=False)
    write(destination,result);print(json.dumps(dict(id=row['video_id'],before=baseline['raw_equivalent_effort_rmse'],
        after=updated['raw_equivalent_effort_rmse'],energy_before=baseline['raw_work_energy_max_J'],
        energy_after=updated['raw_work_energy_max_J'],coverage=updated['mechanical_coverage'])),flush=True)
    return result


def run(role):
    if role=='verification':
        from .delta_optical import raymap
        # Three conditions share one assembly map; construct it before workers.
        seen=set()
        for row in rows(role):
            if row['assembly_id'] not in seen:raymap(row);seen.add(row['assembly_id'])
    with ProcessPoolExecutor(max_workers=3) as pool:results=list(pool.map(case,rows(role)))
    summary={branch:{metric:float(np.mean([r[branch][metric] for r in results])) for metric in
        ['raw_equivalent_effort_rmse','raw_work_energy_max_J','platform_position_rmse_mm']} for branch in ['baseline','updated','exact','numerical']}
    write(OPTICAL/(role+'_summary.json'),dict(cases=results,means=summary,role=role,
        selection_sha256=sha(OPTICAL/'fusion_selection.json')))
    print(json.dumps(dict(role=role,means=summary)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('role',choices=['calibration','verification']);args=p.parse_args();run(args.role)
