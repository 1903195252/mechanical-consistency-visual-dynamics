"""Strict-Grashof double-crank inversion: both grounded links rotate fully."""
import hashlib
import json
import numpy as np

from . import fourbar
from .fourbar import (inventory,solid_properties,mjcf,mass_bias_energy,friction,external,
                      inverse_kinematics,closure,closure_jacobian,closure_curvature,TREE_TO_ABSOLUTE)


def configuration(entry):
    c=fourbar.configuration(entry);rng=np.random.default_rng(entry['seed']+109)
    d=float(rng.uniform(.12,.17))
    lengths=[d*float(rng.uniform(1.70,1.85)),d*float(rng.uniform(1.45,1.55)),d*float(rng.uniform(1.65,1.80))]
    c.update(category='S-J',subtype='strict_grashof_double_crank',ground_length_m=d,lengths_m=lengths,
             pivot_height_m=max(lengths)+.17,payload_along_coupler_m=lengths[1]*.6,
             force_application_along_coupler_m=lengths[1]*.8)
    c['actuator'].update(speed_gain_Nm_s_rad=12.,torque_limit_Nm=12.)
    return c


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts)
    audit=fourbar.geometry_audit(c,parts,bodies)
    lengths=sorted([c['ground_length_m'],*c['lengths_m']])
    margin=lengths[1]+lengths[2]-lengths[0]-lengths[3]
    if margin<=0 or lengths[0]!=c['ground_length_m']:raise ValueError('Not a double-crank inversion')
    audit.update(grashof_margin_m=margin,shortest_link='ground',output_link_full_rotation=True)
    shape=[{key:p[key] for key in ['body','shape','axis','position_m','dimensions_m','rotation_matrix'] if key in p} for p in parts]
    physical={'config':{key:value for key,value in c.items() if key not in ['assembly_id','seed','hardware_recipe']},'bodies':bodies}
    return {'schema_version':'1.0.0','assembly_id':entry['assembly_id'],'category':'S-J','config':c,
            'parts':parts,'bodies':bodies,'geometry_digest':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
            'physical_digest':hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest(),'assembly_audit':audit,
            'presentation':{'description':'Closed-chain double-crank mechanism with the shortest link grounded. Both the input and output cranks can rotate continuously.',
                            'link_marks':[['crank','IN',0],['coupler','BC',1],['rocker','OUT',2]],'focus_body':'coupler',
                            'focus_local_m':[c['lengths_m'][1]/2,0,0]},
            'provenance':{'type':'procedural_reconstruction','sources':['fourbar.py','double_crank.py'],
                          'change':'Different Grashof inversion with short ground and two fully rotating grounded links; separately raised bearing frame.'},
            'physics_scope':'Rigid 4R double-crank mechanism; finite torque actuation and explicit four-joint friction. No parallelogram subtype is claimed.'}
