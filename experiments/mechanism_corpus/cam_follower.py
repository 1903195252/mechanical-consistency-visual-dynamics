"""Eccentric disk cam and force-loaded flat follower with resolved contact."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from . import gear_pair
from .solids import part,aggregate,cylinder,shift_tensor,MATERIALS

BODY_NAMES=['cam','follower']


def eccentric_part(name,body,radius,depth,eccentricity,bore,material,position=(0.,0.,0.)):
    density=MATERIALS[material]['density_kg_m3']
    full,_,Ifull=cylinder(radius,depth,density);hole,_,Ihole=cylinder(bore,depth,density)
    mass=full-hole;center=np.array([full*eccentricity/mass,0,0])
    tensor=Ifull+shift_tensor(full,np.array([eccentricity,0,0])-center)-Ihole-shift_tensor(hole,-center)
    if eccentricity+bore>=radius or np.linalg.eigvalsh(tensor).min()<=0:raise ValueError('Invalid eccentric bored disk')
    return {'name':name,'body':body,'shape':'eccentric_disk','axis':'Y','position_m':list(position),
            'dimensions_m':{'radius':radius,'depth':depth,'eccentricity':eccentricity,'bore_radius':bore},
            'material':material,'density_kg_m3':density,'mass_kg':mass,
            'com_body_m':(np.asarray(position)+center).tolist(),'inertia_com_body_kg_m2':tensor.tolist(),
            'note':'Exact cylinder-minus-offset-bore solid moments; cam profile is an eccentric circle.'}


def configuration(entry):
    rng=np.random.default_rng(entry['seed']);radius=float(rng.uniform(.052,.072));ecc=float(rng.uniform(.008,.016))
    height=radius+.11;shoe=height+radius+ecc
    return {'assembly_id':entry['assembly_id'],'category':'S-E','seed':entry['seed'],'hardware_recipe':entry['hardware_recipe'],
            'subtype':'eccentric_disk_flat_follower','cam_radius_m':radius,'eccentricity_m':ecc,
            'cam_depth_m':float(rng.uniform(.012,.016)),'shaft_radius_m':.006,'pivot_height_m':height,
            'follower_reference_height_m':shoe,'initial_cam_angle_rad':float(np.pi/2),
            'barrel_bottom_height_m':shoe+.060,'barrel_top_height_m':shoe+.210,
            'preload_force_N':float(rng.uniform(8.,16.)),
            'gravity_m_s2':9.81,'motor_armature_kg_m2':.0002,
            'bearing_viscosity_Nm_s_rad':.010,'bearing_coulomb_Nm':.003,
            'guide_viscosity_N_s_m':1.4,'guide_coulomb_N':.15,
            'contact_friction_coefficient':float(rng.uniform(.07,.12)),
            'sliding_regularization_speed_m_s':.005,
            'contact_tangent_model':'explicit_regularized_coulomb_world_forces',
            'normal_contact_representation':'equivalent_sphere',
            'actuator':{'type':'speed_controlled_cam_and_force_loaded_follower','speed_gain_Nm_s_rad':3.,'torque_limit_Nm':5.,'ramp_s':.5},
            'simulation':{'timestep_s':1/4800,'record_hz':240,'fps':30,'duration_s':4.,'width':640,'height':640,
                          'integrator':'implicitfast','contact_solref_s':.001,'contact_impedance':.97,'contact_margin_m':0.,
                          'collision_tolerance_m':1e-12}}


def inventory(c):
    # Reuse the audited shaft/bearing hardware; the cam and follower are new solids.
    stub_gear={'teeth':40,'normal_module_m':.003,'normal_pressure_angle_rad':float(np.pi/9),
               'helix_angle_rad':0.,'depth_m':c['cam_depth_m'],'bore_radius_m':c['shaft_radius_m']+.00002}
    stub={'ground_length_m':.12,'pivot_height_m':c['pivot_height_m'],'shaft_radius_m':c['shaft_radius_m'],
          'gear_face_width_m':c['cam_depth_m'],'gears':[stub_gear,stub_gear]}
    parts=[]
    for p in gear_pair.inventory(stub):
        if p['body']=='input_gear' and not any(token in p['name'] for token in ['involute_blank','face_plate','balanced_index','hub_screw']):
            p['body']='cam';parts.append(p)
        elif p['body']=='ground' and (p['name'].startswith('input_') or p['name'].startswith('motor_')):
            parts.append(p)
    def add(name,body,shape,pos,dims,material,**kw):parts.append(part(name,body,shape,pos,dims,material,**kw))
    R,e,t=c['cam_radius_m'],c['eccentricity_m'],c['cam_depth_m']
    parts.append(eccentric_part('eccentric_cam_blank','cam',R,t,e,c['shaft_radius_m']+.00002,'steel'))
    parts.append(eccentric_part('cam_face_plate','cam',R-.003,.0008,e,.026,'orange',[0,-t/2-.0004,0]))
    add('cam_angle_index','cam','cylinder',[e+.50*R,-t/2-.0018,0],{'radius':.0025,'depth':.002},'graphite')
    add('follower_contact_shoe','follower','box',[0,0,.004],{'size':[.100,.036,.008]},'steel')
    add('follower_crosshead','follower','box',[0,0,.018],{'size':[.024,.024,.020]},'green')
    add('follower_rod','follower','cylinder',[0,0,(.028+.133)/2],{'radius':.005,'depth':.105},'steel',axis='Z')
    add('follower_piston','follower','cylinder',[0,0,.139],{'radius':.01196,'depth':.012},'steel',axis='Z')
    for x in [-.036,.036]:
        add('shoe_screw_'+str(x),'follower','socket_head',[x,0,.0095],{'radius':.0025,'depth':.003},'steel',axis='Z')
    bottom,top=c['barrel_bottom_height_m'],c['barrel_top_height_m']
    add('force_actuator_barrel','ground','ring',[0,0,(bottom+top)/2],{'radius':.020,'inner_radius':.012,'depth':top-bottom},'blue',axis='Z')
    add('rod_guide_bushing','ground','ring',[0,0,bottom-.005],{'radius':.022,'inner_radius':.00503,'depth':.010},'bronze',axis='Z')
    add('actuator_top_cap','ground','cylinder',[0,0,top+.004],{'radius':.023,'depth':.008},'steel',axis='Z')
    for z in [bottom+.012,top-.012]:
        add('actuator_band_'+str(z),'ground','ring',[0,0,z],{'radius':.021,'inner_radius':.020,'depth':.004},'steel',axis='Z')
    beam_bottom=top+.008
    add('actuator_mount_beam','ground','box',[0,0,beam_bottom+.009],{'size':[.26,.09,.018]},'aluminum')
    for x in [-.110,.110]:
        add('frame_foot_'+str(x),'ground','box',[x,.030,.050],{'size':[.070,.085,.020]},'aluminum')
        add('frame_column_'+str(x),'ground','box',[x,.030,(.060+beam_bottom)/2],{'size':[.030,.030,beam_bottom-.060]},'frame')
        for dx in [-.022,.022]:
            for y in [.002,.058]:
                add('frame_base_screw_'+str((x,dx,y)),'ground','socket_head',[x+dx,y,.062],{'radius':.003,'depth':.004},'steel',axis='Z')
        add('frame_top_screw_'+str(x),'ground','socket_head',[x,.030,beam_bottom+.020],{'radius':.0035,'depth':.004},'steel',axis='Z')
    add('machined_base','ground','box',[0,.060,.025],{'size':[.34,.26,.030]},'aluminum')
    for x in [-.12,.12]:
        add('base_foot_'+str(x),'ground','box',[x,.060,.005],{'size':[.050,.21,.010]},'graphite')
    return parts


def solid_properties(parts):return {name:aggregate(parts,name) for name in BODY_NAMES}


def mass_bias_energy(q,v,c,bodies):
    cam=bodies['cam'];follower=bodies['follower'];x,_,z=cam['com_local_m'];theta=c['initial_cam_angle_rad']+q[0]
    I=cam['inertia_com_local_kg_m2'][1][1]+cam['mass_kg']*(x*x+z*z)+c['motor_armature_kg_m2']
    M=np.diag([I,follower['mass_kg']]);g=c['gravity_m_s2']
    G=np.array([cam['mass_kg']*g*(x*np.cos(theta)-z*np.sin(theta)),follower['mass_kg']*g])
    V=cam['mass_kg']*g*(c['pivot_height_m']+x*np.sin(theta)+z*np.cos(theta))
    V+=follower['mass_kg']*g*(c['follower_reference_height_m']+q[1]+follower['com_local_m'][2])
    return M,G,float(.5*v@M@v),float(V)


def friction(v,c):
    return np.array([-c['bearing_viscosity_Nm_s_rad']*v[0]-c['bearing_coulomb_Nm']*np.tanh(v[0]/.04),
                     -c['guide_viscosity_N_s_m']*v[1]-c['guide_coulomb_N']*np.tanh(v[1]/.004)])


def gap_and_jacobian(q,c):
    theta=c['initial_cam_angle_rad']+q[0]
    gap=c['follower_reference_height_m']+q[1]-c['pivot_height_m']-c['cam_radius_m']-c['eccentricity_m']*np.sin(theta)
    return float(gap),np.array([-c['eccentricity_m']*np.cos(theta),1.])


def mjcf(c,bodies):
    vec=lambda a:' '.join('%.16g'%v for v in np.ravel(a));root=ET.Element('mujoco',model=c['assembly_id'])
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false');s=c['simulation']
    option=ET.SubElement(root,'option',timestep=str(s['timestep_s']),gravity=vec([0,0,-c['gravity_m_s2']]),integrator=s['integrator'],
                        ccd_tolerance=str(s['collision_tolerance_m']),
                        solver='Newton',cone='elliptic',jacobian='dense',iterations='100',tolerance='1e-12')
    ET.SubElement(option,'flag',energy='enable');world=ET.SubElement(root,'worldbody');theta=c['initial_cam_angle_rad']
    cam=ET.SubElement(world,'body',name='cam',pos=vec([0,0,c['pivot_height_m']]),quat=vec([np.cos(theta/2),0,-np.sin(theta/2),0]))
    ET.SubElement(cam,'joint',name='cam_shaft',type='hinge',axis='0 -1 0',armature=str(c['motor_armature_kg_m2']))
    follower=ET.SubElement(world,'body',name='follower',pos=vec([0,0,c['follower_reference_height_m']]))
    ET.SubElement(follower,'joint',name='follower_guide',type='slide',axis='0 0 1')
    for name,node in [('cam',cam),('follower',follower)]:
        b=bodies[name];principal,axes=np.linalg.eigh(b['inertia_com_local_kg_m2'])
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        orientation=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
        ET.SubElement(node,'inertial',mass=str(b['mass_kg']),pos=vec(b['com_local_m']),diaginertia=vec(principal),quat=vec(orientation))
    if c.get('normal_contact_representation','cylinder')=='equivalent_sphere':
        # For this planar cylinder/flat-face pair, the sphere has exactly the
        # same support height, normal Jacobian and material-point velocity.
        # Mass/inertia and all visual/clearance geometry remain the real disk.
        ET.SubElement(cam,'geom',name='cam_surface',type='sphere',pos=vec([c['eccentricity_m'],0,0]),size=str(c['cam_radius_m']))
    else:
        ET.SubElement(cam,'geom',name='cam_surface',type='cylinder',pos=vec([c['eccentricity_m'],0,0]),
                      quat=vec([np.cos(np.pi/4),np.sin(np.pi/4),0,0]),size=vec([c['cam_radius_m'],c['cam_depth_m']/2]))
    ET.SubElement(follower,'geom',name='follower_face',type='box',pos='0 0 .004',size='.050 .018 .004')
    contact=ET.SubElement(root,'contact');mu=c['contact_friction_coefficient']
    ET.SubElement(contact,'pair',geom1='cam_surface',geom2='follower_face',condim='1',friction='0 0 0 0 0',
                  solref=vec([s['contact_solref_s'],1]),solimp=vec([s['contact_impedance'],s['contact_impedance'],.001,.5,2]),
                  margin=str(s['contact_margin_m']))
    actuators=ET.SubElement(root,'actuator');limit=c['actuator']['torque_limit_Nm']
    ET.SubElement(actuators,'motor',name='cam_motor',joint='cam_shaft',ctrllimited='true',ctrlrange=vec([-limit,limit]))
    ET.SubElement(actuators,'motor',name='follower_preload',joint='follower_guide',ctrllimited='true',ctrlrange='-100 0')
    return ET.tostring(root,encoding='unicode'),np.zeros(2)


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts)
    if c['eccentricity_m']+.026>=c['cam_radius_m']-.003:raise ValueError('Face plate/bore overlap')
    shape=[{key:p[key] for key in ['body','shape','axis','position_m','dimensions_m']} for p in parts]
    physical={'config':{k:v for k,v in c.items() if k not in ['assembly_id','seed','hardware_recipe']},'bodies':bodies}
    return {'schema_version':'1.0.0','assembly_id':entry['assembly_id'],'category':'S-E','config':c,'parts':parts,'bodies':bodies,
            'geometry_digest':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
            'physical_digest':hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest(),
            'assembly_audit':{'status':'nominal_solids_defined_pending_contact_validation','tree_dof':2,'nominal_active_normal_rank':1,
                              'independent_dof_with_contact':1,'free_dof_if_detached':2,
                              'rod_guide_radial_clearance_m':.00003,'piston_barrel_radial_clearance_m':.00004,
                              'body_masses_kg':{k:v['mass_kg'] for k,v in bodies.items()}},
            'presentation':{'description':'Torque-driven eccentric disk cam with a force-loaded flat follower, a bored linear actuator housing, and resolved frictional cam contact.',
                            'link_marks':[],'focus_body':'cam','focus_bodies':BODY_NAMES},
            'permitted_contacts':[{'geometry':'eccentric_cylinder_on_box_bottom','cylinder_part':'eccentric_cam_blank',
                                   'box_part':'follower_contact_shoe','maximum_penetration_m':2e-6,'maximum_gap_m':1e-8,
                                   'normal_force_field':'summed_normal_contact_force_N','minimum_normal_force_N':1e-6}],
            'provenance':{'type':'procedural_reconstruction','sources':['cam_follower.py','gear_pair.py'],
                          'change':'New offset-bore eccentric cam, vertical follower and load actuator. Follower motion is produced by MuJoCo contact, without a prescribed lift trajectory.'},
            'physics_scope':'Rigid eccentric cylindrical cam with equivalent analytic sphere/flat-face normal contact for the constrained planar motion. The real disk geometry, mass and inertia are retained. Explicit regularized Coulomb sliding forces are solved consistently with normal contact; the upper actuator applies a calibrated downward force.'}
