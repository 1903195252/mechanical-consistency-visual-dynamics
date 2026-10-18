"""Bounded, named physical contact rules; never blanket collision exemptions."""
import numpy as np


def evaluate(rule,parts,transforms,normal_force):
    if rule['geometry']!='eccentric_cylinder_on_box_bottom':raise ValueError('Unsupported declared contact geometry')
    cylinder=parts[rule['cylinder_part']];box=parts[rule['box_part']]
    if cylinder['shape']!='eccentric_disk' or box['shape']!='box':raise ValueError('Declared contact primitive differs from its part')
    Rc,pc=transforms[cylinder['body']];Rb,pb=transforms[box['body']]
    d=cylinder['dimensions_m'];size=np.asarray(box['dimensions_m']['size'])
    center=pc+Rc@(np.asarray(cylinder['position_m'])+[d['eccentricity'],0,0])
    plane=pb+Rb@(np.asarray(box['position_m'])+[0,0,-size[2]/2]);normal=Rb[:,2];axis=Rc[:,1]
    gap=float((plane-center)@normal-d['radius'])
    contact_center=center+normal*d['radius'];local=Rb.T@(contact_center-plane)
    footprint_half=d['depth']/2*abs(Rb.T@axis)
    checks={'axis_parallel_to_face':abs(float(normal@axis))<1e-7,
            'within_flat_face':bool(np.all(abs(local[:2])+footprint_half[:2]<=size[:2]/2)),
            'bounded_normal_penetration':gap>=-rule['maximum_penetration_m'],
            'closed_contact':gap<=rule.get('maximum_gap_m',1e-8),
            'compressive_reference_force':float(normal_force)>rule.get('minimum_normal_force_N',1e-6)}
    return {'passed':all(checks.values()),'checks':checks,'normal_gap_m':gap,'normal_force_N':float(normal_force),
            'parts':[rule['cylinder_part'],rule['box_part']],'maximum_penetration_m':rule['maximum_penetration_m']}
