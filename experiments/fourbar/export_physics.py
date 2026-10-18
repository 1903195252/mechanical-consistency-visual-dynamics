"""Export explicit equation terms as privileged evaluation ground truth."""

import json
from pathlib import Path
import numpy as np

from mechanics import mass_bias_energy,friction,coupler_point,closure_jacobian


def enrich(folder):
    config=json.loads((folder/'physical_metadata.json').read_text())
    with np.load(folder/'ground_truth.npz') as saved:
        data={name:saved[name] for name in saved.files}
    terms={name:[] for name in ['mass_matrix_kg_m2','coriolis_Nm','gravity_Nm','generalized_friction_Nm',
                               'joint_relative_rates_rad_s','joint_friction_torque_Nm','external_generalized_force_Nm',
                               'generalized_constraint_force_Nm']}
    for i,(q,v) in enumerate(zip(data['q'],data['qdot'])):
        mass,c,g,_,_=mass_bias_energy(q,v,config)
        f,rates,joint_torques=friction(v,config)
        _,jacobian=coupler_point(q,config,config['force_application_along_coupler_m'])
        values=[mass,c,g,f,rates,joint_torques,jacobian.T@np.asarray(config['external_force_xz_N']),
                closure_jacobian(q,config).T@data['constraint_reaction'][i]]
        for name,value in zip(terms,values):terms[name].append(value)
    for name,value in terms.items():
        data[name]=np.asarray(value)
        assert np.isfinite(data[name]).all(),name
    np.savez_compressed(folder/'ground_truth.npz',**data)
    print(folder.name,'exported',len(terms),'equation fields;',len(data['time']),'timestamps')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]/'output'/'fourbar'
    for case in ['nominal','heavy_payload','high_friction']:enrich(root/case)
