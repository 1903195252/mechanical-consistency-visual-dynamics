"""Conjugate external involute gears with bearing stacks and an ideal mesh law."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .solids import part,aggregate
from . import gear_geometry

BODY_NAMES=['input_gear','output_gear']


def configuration(entry):
    rng=np.random.default_rng(entry['seed']);helical=entry['ordinal']%2==0
    beta=float(np.deg2rad(rng.choice([15.,20.,25.]))) if helical else 0.
    module=float(rng.choice([.003,.0035,.004]));teeth=[int(rng.choice([20,24,28,32])),int(rng.choice([36,40,48,56,64]))]
    width=float(rng.uniform(.013,.020));shaft=.006
    gears=[{'teeth':z,'normal_module_m':module,'normal_pressure_angle_rad':float(np.deg2rad(20)),
            'helix_angle_rad':beta*(1 if i==0 else -1),'depth_m':width,'bore_radius_m':shaft+.00002}
           for i,z in enumerate(teeth)]
    dims=[gear_geometry.dimensions(g) for g in gears];distance=sum(g['pitch_radius_m'] for g in dims)
    return {'assembly_id':entry['assembly_id'],'category':'S-C','seed':entry['seed'],'hardware_recipe':entry['hardware_recipe'],
            'subtype':'helical' if helical else 'spur','gears':gears,'teeth':teeth,'normal_module_m':module,
            'gear_face_width_m':width,'shaft_radius_m':shaft,'ground_length_m':distance,
            'pivot_height_m':max(g['tip_radius_m'] for g in dims)+.12,
            'initial_angles_rad':[0.,float(np.pi-np.pi/teeth[1])],
            'signed_speed_ratio':-teeth[0]/teeth[1],'gravity_m_s2':9.81,'motor_armature_kg_m2':.00015,
            'bearing_viscosity_Nm_s_rad':[.008,.010],'bearing_coulomb_Nm':[.003,.004],
            'coulomb_smoothing_rad_s':.04,'nominal_output_load_torque_Nm':float(rng.uniform(.035,.065)),
            'actuator':{'type':'finite_torque_velocity_servo','speed_gain_Nm_s_rad':2.,'torque_limit_Nm':5.,'ramp_s':.5},
            'simulation':{'timestep_s':1/4800,'record_hz':240,'fps':30,'duration_s':4.,'width':640,'height':640,
                          'integrator':'RK4','equality_solref_s':.001,'equality_impedance':.999}}


def inventory(c):
    parts=[];distance=c['ground_length_m'];height=c['pivot_height_m'];shaft=c['shaft_radius_m']
    def add(name,body,shape,pos,dims,material,**kw):parts.append(part(name,body,shape,pos,dims,material,**kw))
    for i,(body,parameters,color) in enumerate(zip(BODY_NAMES,c['gears'],['blue','orange'])):
        w=parameters['depth_m'];dims=gear_geometry.dimensions(parameters);root=dims['root_radius_m'];prefix='G%d_'%(i+1)
        hub=min(.023,.43*root);index_radius=min(.0035,.055*root);washer_radius=min(.010,.82*hub)
        parts.append(gear_geometry.part(prefix+'involute_blank',body,parameters))
        add(prefix+'front_hub',body,'ring',[0,-w/2-.003,0],{'radius':hub,'inner_radius':shaft+.00002,'depth':.006},'steel')
        add(prefix+'rear_hub',body,'ring',[0,w/2+.004,0],{'radius':min(.020,hub),'inner_radius':shaft+.00002,'depth':.008},'steel')
        add(prefix+'face_plate',body,'ring',[0,-w/2-.0004,0],{'radius':root-.003,'inner_radius':hub+.002,'depth':.0008},color)
        for sign,material in [(1,'graphite'),(-1,'steel')]:
            add(prefix+'balanced_index_'+str(sign),body,'cylinder',[sign*.65*root,-w/2-.0018,0],
                {'radius':index_radius,'depth':.002},material)
        if hub>=.018:
            for angle in np.arange(4)*np.pi/2:
                add(prefix+'hub_screw_'+str(angle),body,'socket_head',[.75*hub*np.cos(angle),-w/2-.0075,.75*hub*np.sin(angle)],
                    {'radius':.1*hub,'depth':.003},'steel')
        front=-w/2-.008
        add(prefix+'front_washer',body,'ring',[0,-w/2-.007,0],{'radius':washer_radius,'inner_radius':shaft+.00004,'depth':.002},'steel')
        add(prefix+'shaft_cap',body,'socket_head',[0,front-.002,0],{'radius':.009,'depth':.004},'steel')
        add(prefix+'shaft',body,'cylinder',[0,(front+.098)/2,0],{'radius':shaft,'depth':.098-front},'steel')
        add(prefix+'bearing_spacer',body,'ring',[0,(w/2+.008+.038)/2,0],
            {'radius':.009,'inner_radius':shaft+.00004,'depth':.038-w/2-.008},'steel')
        add(prefix+'rear_washer',body,'ring',[0,.092,0],{'radius':.010,'inner_radius':shaft+.00004,'depth':.002},'steel')
        add(prefix+'retaining_nut',body,'hex_nut',[0,.096,0],{'radius':.009,'inner_radius':shaft+.00004,'depth':.006},'steel')
    add('input_rotor_hub','input_gear','cylinder',[0,.113,0],{'radius':.009,'depth':.030},'steel')
    add('machined_base','ground','box',[distance/2,.075,.025],{'size':[distance+.28,.28,.030]},'aluminum')
    for x in [-.095,distance+.095]:
        add('base_foot_'+str(x),'ground','box',[x,.075,.005],{'size':[.05,.23,.010]},'graphite')
    for label,x in [('input',0.),('output',distance)]:
        add(label+'_support_foot','ground','box',[x,.072,.050],{'size':[.105,.112,.020]},'aluminum')
        add(label+'_column','ground','box',[x,.073,(.060+height-.035)/2],{'size':[.050,.046,height-.095]},'frame')
        add(label+'_bearing_block','ground','bearing_block',[x,.064,height],{'size':[.074,.052,.070],'bore_radius':.020},'aluminum')
        add(label+'_bearing_outer','ground','ring',[x,.064,height],{'radius':.020,'inner_radius':.015,'depth':.052},'steel')
        add(label+'_bearing_inner','ground','ring',[x,.064,height],{'radius':.010,'inner_radius':shaft+.00003,'depth':.052},'steel')
        add(label+'_bearing_seal','ground','ring',[x,.0365,height],{'radius':.019,'inner_radius':.0091,'depth':.003},'graphite')
        for dx in [-.026,.026]:
            for dz in [-.024,.024]:
                add(label+'_bearing_bolt_'+str((dx,dz)),'ground','socket_head',[x+dx,.0365,height+dz],{'radius':.003,'depth':.003},'steel')
        for dx in [-.039,.039]:
            for y in [.033,.111]:
                add(label+'_foot_bolt_'+str((dx,y)),'ground','socket_head',[x+dx,y,.0625],{'radius':.0035,'depth':.005},'steel',axis='Z')
    add('motor_flange','ground','ring',[0,.095,height],{'radius':.044,'inner_radius':.014,'depth':.010},'aluminum')
    add('motor_housing','ground','ring',[0,.135,height],{'radius':.039,'inner_radius':.013,'depth':.070},'graphite')
    for y in [.112,.124,.136,.148,.160]:
        add('motor_rib_'+str(y),'ground','ring',[0,y,height],{'radius':.041,'inner_radius':.039,'depth':.002},'graphite')
    return parts


def solid_properties(parts):return {name:aggregate(parts,name) for name in BODY_NAMES}


def inertias(c,bodies):
    values=[]
    for name in BODY_NAMES:
        b=bodies[name];rx,_,rz=b['com_local_m']
        values.append(b['inertia_com_local_kg_m2'][1][1]+b['mass_kg']*(rx*rx+rz*rz))
    values[0]+=c['motor_armature_kg_m2'];return np.asarray(values)


def mass_bias_energy(q,v,c,bodies):
    M=np.diag(inertias(c,bodies));G=np.zeros(2);V=0.
    for i,name in enumerate(BODY_NAMES):
        b=bodies[name];x,_,z=b['com_local_m'];theta=c['initial_angles_rad'][i]+q[i]
        G[i]=b['mass_kg']*c['gravity_m_s2']*(x*np.cos(theta)-z*np.sin(theta))
        V+=b['mass_kg']*c['gravity_m_s2']*(c['pivot_height_m']+x*np.sin(theta)+z*np.cos(theta))
    return M,G,float(.5*v@M@v),float(V)


def friction(v,c):
    return -np.asarray(c['bearing_viscosity_Nm_s_rad'])*v-np.asarray(c['bearing_coulomb_Nm'])*np.tanh(v/c['coulomb_smoothing_rad_s'])


def mjcf(c,bodies):
    vec=lambda a:' '.join('%.16g'%v for v in np.ravel(a));root=ET.Element('mujoco',model=c['assembly_id'])
    ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false');s=c['simulation']
    opt=ET.SubElement(root,'option',timestep=str(s['timestep_s']),gravity='0 0 -9.81',integrator='RK4',
                      solver='Newton',jacobian='dense',iterations='100',tolerance='1e-12');ET.SubElement(opt,'flag',energy='enable')
    world=ET.SubElement(root,'worldbody')
    for i,name in enumerate(BODY_NAMES):
        angle=c['initial_angles_rad'][i];node=ET.SubElement(world,'body',name=name,pos=vec([i*c['ground_length_m'],0,c['pivot_height_m']]),
                                                          quat=vec([np.cos(angle/2),0,-np.sin(angle/2),0]))
        ET.SubElement(node,'joint',name=name+'_shaft',type='hinge',axis='0 -1 0',armature=str(c['motor_armature_kg_m2'] if i==0 else 0))
        b=bodies[name];principal,axes=np.linalg.eigh(b['inertia_com_local_kg_m2'])
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        orientation=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
        ET.SubElement(node,'inertial',mass=str(b['mass_kg']),pos=vec(b['com_local_m']),diaginertia=vec(principal),quat=vec(orientation))
        ET.SubElement(node,'geom',type='cylinder',size=vec([gear_geometry.dimensions(c['gears'][i])['pitch_radius_m'],c['gear_face_width_m']/2]),
                      quat=vec([np.cos(np.pi/4),np.sin(np.pi/4),0,0]),contype='0',conaffinity='0')
    eq=ET.SubElement(root,'equality')
    ET.SubElement(eq,'joint',name='ideal_involute_mesh',joint1='output_gear_shaft',joint2='input_gear_shaft',
                  polycoef=vec([0,c['signed_speed_ratio'],0,0,0]),solref=vec([s['equality_solref_s'],1]),
                  solimp=vec([s['equality_impedance'],s['equality_impedance'],.001,.5,2]))
    actuator=ET.SubElement(root,'actuator');limit=c['actuator']['torque_limit_Nm']
    ET.SubElement(actuator,'motor',name='input_motor',joint='input_gear_shaft',ctrllimited='true',ctrlrange=vec([-limit,limit]))
    return ET.tostring(root,encoding='unicode'),np.zeros(2)


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts);dims=[gear_geometry.dimensions(p) for p in c['gears']]
    alpha=dims[0]['transverse_pressure_angle_rad'];base_pitch=np.pi*dims[0]['transverse_module_m']*np.cos(alpha)
    contact_ratio=(sum(np.sqrt(d['tip_radius_m']**2-d['base_radius_m']**2) for d in dims)-c['ground_length_m']*np.sin(alpha))/base_pitch
    paths=[np.sqrt(d['tip_radius_m']**2-d['base_radius_m']**2)-d['pitch_radius_m']*np.sin(alpha) for d in dims]
    minimum_active_roll=[dims[i]['pitch_radius_m']*np.sin(alpha)-paths[1-i] for i in [0,1]]
    if contact_ratio<=1.1 or min(inertias(c,bodies))<=0:raise ValueError('Invalid gear pair')
    if min(minimum_active_roll)<=0:raise ValueError('Active mating flank would enter the non-involute root region')
    shape=[{key:p[key] for key in ['body','shape','axis','position_m','dimensions_m']} for p in parts]
    physical={'config':{k:v for k,v in c.items() if k not in ['assembly_id','seed','hardware_recipe']},'bodies':bodies}
    return {'schema_version':'1.0.0','assembly_id':entry['assembly_id'],'category':'S-C','config':c,'parts':parts,'bodies':bodies,
            'geometry_digest':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
            'physical_digest':hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest(),
            'assembly_audit':{'status':'gear_geometry_and_inertia_validated','tree_dof':2,'closure_rank':1,'independent_dof':1,
                              'transverse_contact_ratio':float(contact_ratio),'shaft_bearing_radial_clearance_m':.00003,
                              'minimum_active_involute_roll_m':[float(v) for v in minimum_active_roll],
                              'opposite_helix_angles':c['gears'][0]['helix_angle_rad']==-c['gears'][1]['helix_angle_rad'],
                              'body_masses_kg':{k:v['mass_kg'] for k,v in bodies.items()}},
            'presentation':{'description':('Opposite-hand helical' if c['subtype']=='helical' else 'Spur')+
                            ' external involute gear pair with a driven input shaft, loaded output shaft, bearings and balanced visual index plugs.',
                            'link_marks':[],'focus_body':'input_gear','focus_bodies':BODY_NAMES},
            'provenance':{'type':'procedural_reconstruction_from_open_reference',
                          'reference':'https://github.com/gabemorris12/mechanism','revision':'b68554a5415812b3c63fb5d8a4db4db7e9f8c98a',
                          'license':'MIT; retained in licenses/mechanism_MIT.txt',
                          'engineering_reference':'https://khkgears.net/gear-knowledge/gear-technical-reference/calculation-gear-dimensions/',
                          'change':'Metric bored solid gear profiles, optional conjugate helical sweep, explicit complete rotating hardware inertia and physical mounting.'},
            'physics_scope':'Rigid shafts with an ideal zero-backlash conjugate mesh constraint and bearing friction. Generalized mesh torques are resolved; elastic tooth stress and distributed contact pressure are not modeled.'}
