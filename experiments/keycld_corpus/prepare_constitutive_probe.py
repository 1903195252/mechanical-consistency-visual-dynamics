"""Prepare independent true-state response probes without modifying the main protocol."""
import json
import time
import numpy as np
from experiments.mechanism_corpus.catalog import ROOT,write_json
from .prepare import OUT,point_states,sha
from .physics_score import MODULES


def main():
    if (OUT/'constitutive_probe_inputs.npz').exists():
        raise RuntimeError('Preserve registered probe states')
    records=json.loads((OUT/'dataset_manifest.json').read_text())['records']
    arrays={k:[] for k in ('x','unit_velocity','geometry','external','true_effective_inertia','true_friction_odd_acceleration')};items=[]
    for record in records:
        if record['split']!='test':continue
        params=json.loads((ROOT/record['physical_parameters']).read_text());c=params['config'];module=MODULES[c['category']]
        with np.load(OUT/record['inputs']) as z:g=z['geometry'];external=z['action'][3:5]
        with np.load(ROOT/record['physics_reference']) as z:
            qs=z['q'][np.array([280,400,520,640,760,880])]
        for index,q in enumerate(qs):
            J=module.closure_jacobian(q,c);N=np.r_[1.,np.linalg.solve(J[:,1:],-J[:,0])]
            x,v,_=point_states(q[None],N[None],np.zeros((1,3)),c)
            M,_,_,_,_=module.mass_bias_energy(q,np.zeros(3),c,params['bodies']);inertia=float(N@M@N)
            odd=float(N@(module.friction(N,c)-module.friction(-N,c)))/(2*inertia)
            arrays['x'].append(((x[0]-[.2,.35])/.65).ravel());arrays['unit_velocity'].append((v[0]/.65).ravel())
            arrays['geometry'].append(g);arrays['external'].append(external)
            arrays['true_effective_inertia'].append(inertia);arrays['true_friction_odd_acceleration'].append(odd)
            items.append({'case_id':record['case_id'],'assembly_id':record['assembly_id'],'condition':record['condition'],'pose_index':index})
    np.savez_compressed(OUT/'constitutive_probe_inputs.npz',**{k:np.asarray(v,dtype=np.float32 if k in ('x','unit_velocity','geometry','external') else float) for k,v in arrays.items()})
    write_json(OUT/'constitutive_probe_manifest.json',{'registered_unix_s':time.time(),'post_hoc_diagnostic':True,
               'fixed_checkpoints_and_main_score_unchanged':True,'test_assemblies':36,'records':items,
               'inertia_definition':'Ieff = 1/(d alpha_crank / d physical_motor_torque)',
               'friction_definition':'Matched states and +1/-1 rad/s independent speed, same 0Nm motor input and known external force; odd acceleration times true Ieff gives friction torque.'})


if __name__=='__main__':main()
