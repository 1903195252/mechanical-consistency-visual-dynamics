"""Rotary three-DOF Delta with explicit paired U--S forearm rods."""
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .spatial_solids import part,rx,ry,rz
from .solids import aggregate
from .joint_hardware import universal
from . import gear_pair


def configuration(entry):
    rng=np.random.default_rng(entry['seed']);radial_drop=float(rng.uniform(.080,.110));rp=float(rng.uniform(.065,.090))
    upper=float(rng.uniform(.17,.21));lower=float(rng.uniform(.38,.46));alpha=float(rng.uniform(.07,.17))
    height=float(rng.uniform(.78,.82));spacing=float(rng.uniform(.065,.075));offset=.062
    gamma=np.pi-alpha;elbow=ry(gamma)@np.array([upper,0,offset]);R=rp-elbow[0]+radial_drop;radial=R+elbow[0]
    platform_height=height+elbow[2]-np.sqrt(lower*lower-(radial-rp)**2)
    lower_world=np.array([rp-radial,0,platform_height-height-elbow[2]])
    lower_local=ry(gamma).T@lower_world;beta=float(np.arctan2(lower_local[0],lower_local[2]))
    return {'assembly_id':entry['assembly_id'],'category':'H-B','seed':entry['seed'],'hardware_recipe':entry['hardware_recipe'],
            'base_radius_m':R,'platform_anchor_radius_m':rp,'upper_beam_length_m':upper,'forearm_length_m':lower,
            'upper_joint_offset_m':offset,'rod_pair_spacing_m':spacing,'upper_reference_angle_rad':gamma,
            'nominal_forearm_radial_offset_m':radial_drop,
            'forearm_reference_pitch_rad':beta,'base_height_m':height,'pivot_height_m':height,'platform_height_m':float(platform_height),
            'leg_azimuths_rad':[0.,float(2*np.pi/3),float(4*np.pi/3)],'gravity_m_s2':9.81,
            'shaft_radius_m':.006,'payload_mass_kg':float(rng.uniform(.25,.50)),'motor_armature_kg_m2':.0002,
            'drive_viscosity_Nm_s_rad':.018,'drive_coulomb_Nm':.003,'passive_viscosity_Nm_s_rad':.003,'passive_coulomb_Nm':.0005,
            'actuator':{'type':'three_torque_limited_joint_position_servos','position_gain_Nm_rad':160.,'velocity_gain_Nm_s_rad':8.0,
                        'torque_limit_Nm':15.,'ramp_s':.7},
            'simulation':{'timestep_s':1/4800,'record_hz':240,'fps':30,'duration_s':4.,'width':640,'height':640,
                          'integrator':'RK4','equality_solref_s':.001,'equality_impedance':.999}}


def body_names():
    return ['platform']+[name for i in range(3) for name in ['upper_%d'%i,*[kind+'_%d_%d'%(i,j) for j in range(2) for kind in ['cross','rod']]]]


def anchor_pairs(c):
    result=[]
    for i,phi in enumerate(c['leg_azimuths_rad']):
        R=rz(phi)
        for j,sign in enumerate([-1,1]):
            upper=np.array([c['upper_beam_length_m'],sign*c['rod_pair_spacing_m']/2,c['upper_joint_offset_m']])
            platform=R@np.array([c['platform_anchor_radius_m'],sign*c['rod_pair_spacing_m']/2,0])
            result.append((i,j,upper,platform))
    return result


def inventory(c):
    parts=[];YtoZ=rx(np.pi/2);head_up=rx(-np.pi/2);Rbase=c['base_radius_m'];H=c['base_height_m'];L=c['upper_beam_length_m']
    def add(name,body,shape,p,d,material,R=None,density=None):parts.append(part(name,body,shape,p,d,material,R,density))
    add('upper_frame_ring','ground','ring',[0,0,H+.056],{'radius':Rbase+.09,'inner_radius':Rbase-.07,'depth':.022},'aluminum',YtoZ)
    for k,angle in enumerate(np.pi/2+np.arange(3)*2*np.pi/3):
        R=rz(angle);location=(Rbase+.06)*np.array([np.cos(angle),np.sin(angle),0])
        add('frame_foot_%d'%k,'ground','box',location+[0,0,.0075],{'size':[.11,.09,.015]},'aluminum',R)
        add('frame_post_%d'%k,'ground','box',location+[0,0,(.015+H+.045)/2],{'size':[.030,.030,H+.030]},'frame',R)
        for dx in [-.038,.038]:
            for dy in [-.028,.028]:
                add('frame_bolt_'+str((k,dx,dy)),'ground','socket_head',location+R@[dx,dy,.017],{'radius':.0035,'depth':.004},'steel',head_up)
    stub={'ground_length_m':.12,'pivot_height_m':.20,'shaft_radius_m':.006,'gear_face_width_m':.028,
          'gears':[{'teeth':40,'normal_module_m':.003,'normal_pressure_angle_rad':float(np.pi/9),'helix_angle_rad':0.,'depth_m':.028,'bore_radius_m':.00602}]*2}
    hardware=gear_pair.inventory(stub)
    for i,phi in enumerate(c['leg_azimuths_rad']):
        R=rz(phi);origin=np.array([Rbase*np.cos(phi),Rbase*np.sin(phi),H]);upper='upper_%d'%i
        for source in hardware:
            p=copy.deepcopy(source)
            if p['body']=='input_gear' and not any(token in p['name'] for token in ['involute_blank','face_plate','balanced_index','hub_screw']):
                if p['name']=='input_rotor_hub':
                    p=part('input_rotor_hub',upper,'cylinder',[0,.1135,0],{'radius':.009,'depth':.029},'steel')
                p['name']='drive_%d_'%i+p['name'];p['body']=upper;parts.append(p)
            elif p['body']=='ground' and (p['name'].startswith('motor_') or p['name'].startswith('input_bearing_')):
                rotation=R
                # Source fixture parts are all Y-axis parts; retain socket orientation.
                add('motor_%d_'%i+p['name'],'ground',p['shape'],origin+R@(np.asarray(p['position_m'])-[0,0,stub['pivot_height_m']]),p['dimensions_m'],p['material'],rotation,p['density_kg_m3'])
        add('motor_mount_%d'%i,'ground','box',origin+R@[0,.064,.040],{'size':[.074,.052,.010]},'aluminum',R)
        color=['blue','orange','green'][i]
        add('upper_hub_%d'%i,upper,'bearing_block',[0,0,0],{'size':[.060,.028,.060],'bore_radius':.00602},color)
        add('upper_beam_%d'%i,upper,'box',[(.030+L-.040)/2,0,0],{'size':[L-.070,.028,.020]},color)
        add('upper_crossbeam_%d'%i,upper,'box',[L,0,0],{'size':[.080,c['rod_pair_spacing_m']+.060,.030]},color)
    forearm=c['forearm_length_m'];ball_radius=.010;neck_radius=.0055;ball_lower=-np.sqrt(ball_radius**2-neck_radius**2)
    for i,j,B,A in anchor_pairs(c):
        cross='cross_%d_%d'%(i,j);rod='rod_%d_%d'%(i,j);prefix='leg_%d_%d'%(i,j)
        parts.extend(universal(prefix,'upper_%d'%i,cross,rod,B))
        add(prefix+'_tube',rod,'ring',[0,0,(.064+forearm-.044)/2],{'radius':.009,'inner_radius':.006,'depth':forearm-.108},'frame',YtoZ,density=1600.)
        add(prefix+'_tip_cap',rod,'cylinder',[0,0,forearm-.040],{'radius':.009,'depth':.008},'steel',YtoZ)
        add(prefix+'_tip_neck',rod,'cylinder',[0,0,forearm+(-.036+ball_lower)/2],{'radius':neck_radius,'depth':ball_lower+.036},'steel',YtoZ)
        add(prefix+'_ball',rod,'axial_profile',[0,0,forearm],{'profile_type':'sphere_segment','radius':ball_radius,'lower':float(ball_lower),'upper':ball_radius},'steel',YtoZ)
        add(prefix+'_socket','platform','axial_profile',A,{'profile_type':'ball_socket','outer_radius':.016,'cavity_radius':.01004,'lower':-.003,'upper':.026},'bronze',rx(-np.pi/2))
        add(prefix+'_socket_mount','platform','cylinder',A+[0,0,-.0305],{'radius':.016,'depth':.009},'aluminum',YtoZ)
    platform_radius=np.hypot(c['platform_anchor_radius_m'],c['rod_pair_spacing_m']/2)+.025
    add('platform_plate','platform','ring',[0,0,-.045],{'radius':platform_radius,'inner_radius':.016,'depth':.020},'aluminum',YtoZ)
    add('tool_mount','platform','cylinder',[0,0,-.060],{'radius':.032,'depth':.010},'blue',YtoZ)
    add('tool_payload','platform','box',[.008,-.006,-.085],{'size':[.055,.055,.040]},'graphite',density=c['payload_mass_kg']/(.055*.055*.040))
    for x in [-.020,.020]:
        for y in [-.020,.020]:
            add('tool_mount_bolt_'+str((x,y)),'platform','socket_head',[x,y,-.033],{'radius':.003,'depth':.004},'steel',head_up)
    return parts


def solid_properties(parts):return {name:aggregate(parts,name) for name in body_names()}


def inverse_kinematics(position,c):
    result=[];L=c['upper_beam_length_m'];off=c['upper_joint_offset_m'];forearm=c['forearm_length_m'];gamma0=c['upper_reference_angle_rad']
    for phi in c['leg_azimuths_rad']:
        local=rz(phi).T@np.asarray(position)-[c['base_radius_m'],0,c['base_height_m']]
        local[0]+=c['platform_anchor_radius_m'];A=local[0]*L+local[2]*off;B=local[0]*off-local[2]*L
        target=(local@local+L*L+off*off-forearm*forearm)/2;radius=np.hypot(A,B)
        if abs(target)>=radius:raise ValueError('Delta pose is outside its reachable branch')
        center=np.arctan2(B,A);spread=np.arccos(target/radius)
        choices=[(angle-gamma0+np.pi)%(2*np.pi)-np.pi for angle in [center-spread,center+spread]]
        result.append(min(choices,key=abs))
    return np.asarray(result)


def mjcf(c,bodies):
    vec=lambda a:' '.join('%.16g'%v for v in np.ravel(a));quat=lambda R:vec(Rotation.from_matrix(R).as_quat()[[3,0,1,2]])
    root=ET.Element('mujoco',model=c['assembly_id']);ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false')
    s=c['simulation'];opt=ET.SubElement(root,'option',timestep=str(s['timestep_s']),gravity=vec([0,0,-c['gravity_m_s2']]),integrator=s['integrator'],
                                      solver='Newton',jacobian='dense',iterations='100',tolerance='1e-12');ET.SubElement(opt,'flag',energy='enable')
    world=ET.SubElement(root,'worldbody');platform=ET.SubElement(world,'body',name='platform',pos=vec([0,0,c['platform_height_m']]))
    ET.SubElement(platform,'freejoint',name='platform_free');nodes={'platform':platform}
    for i,phi in enumerate(c['leg_azimuths_rad']):
        upper=ET.SubElement(world,'body',name='upper_%d'%i,pos=vec([c['base_radius_m']*np.cos(phi),c['base_radius_m']*np.sin(phi),c['base_height_m']]),
                            quat=quat(rz(phi)@ry(c['upper_reference_angle_rad'])))
        ET.SubElement(upper,'joint',name='drive_%d'%i,type='hinge',axis='0 1 0',armature=str(c['motor_armature_kg_m2']));nodes['upper_%d'%i]=upper
        for ii,j,B,A in anchor_pairs(c):
            if ii!=i:continue
            cross=ET.SubElement(upper,'body',name='cross_%d_%d'%(i,j),pos=vec(B));ET.SubElement(cross,'joint',name='universal_x_%d_%d'%(i,j),type='hinge',axis='1 0 0')
            rod=ET.SubElement(cross,'body',name='rod_%d_%d'%(i,j),quat=quat(ry(c['forearm_reference_pitch_rad'])));ET.SubElement(rod,'joint',name='universal_y_%d_%d'%(i,j),type='hinge',axis='0 1 0')
            nodes['cross_%d_%d'%(i,j)]=cross;nodes['rod_%d_%d'%(i,j)]=rod
    for name,node in nodes.items():
        b=bodies[name];principal,axes=np.linalg.eigh(b['inertia_com_local_kg_m2'])
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        ET.SubElement(node,'inertial',pos=vec(b['com_local_m']),mass=str(b['mass_kg']),diaginertia=vec(principal),quat=quat(axes))
        visual={'contype':'0','conaffinity':'0','rgba':'.4 .5 .6 1'}
        if name=='platform':
            radius=np.hypot(c['platform_anchor_radius_m'],c['rod_pair_spacing_m']/2)+.025
            ET.SubElement(node,'geom',type='cylinder',pos='0 0 -.045',size=vec([radius,.010]),**visual)
        elif name.startswith('upper_'):
            ET.SubElement(node,'geom',type='box',size='.030 .014 .030',**visual)
            ET.SubElement(node,'geom',type='box',pos=vec([(c['upper_beam_length_m']-.010)/2,0,0]),
                          size=vec([(c['upper_beam_length_m']-.070)/2,.014,.010]),**visual)
        elif name.startswith('rod_'):
            ET.SubElement(node,'geom',type='cylinder',pos=vec([0,0,(.064+c['forearm_length_m']-.044)/2]),
                          size=vec([.009,(c['forearm_length_m']-.108)/2]),**visual)
            ET.SubElement(node,'geom',type='sphere',pos=vec([0,0,c['forearm_length_m']]),size='.010',**visual)
        else:ET.SubElement(node,'geom',type='box',size='.009 .009 .009',**visual)
    eq=ET.SubElement(root,'equality')
    for i,j,B,A in anchor_pairs(c):
        ET.SubElement(eq,'connect',name='ball_%d_%d'%(i,j),body1='rod_%d_%d'%(i,j),body2='platform',anchor=vec([0,0,c['forearm_length_m']]),
                      solref=vec([s['equality_solref_s'],1]),solimp=vec([s['equality_impedance'],s['equality_impedance'],.001,.5,2]))
    actuators=ET.SubElement(root,'actuator');limit=c['actuator']['torque_limit_Nm']
    for i in range(3):ET.SubElement(actuators,'motor',name='shoulder_motor_%d'%i,joint='drive_%d'%i,ctrllimited='true',ctrlrange=vec([-limit,limit]))
    return ET.tostring(root,encoding='unicode'),None


def closure(states,tree,c):
    names={b['name']:b['id'] for b in tree.bodies};residual=[];rows=[]
    for i,j,B,A in anchor_pairs(c):
        tip,Jtip=tree.point(states[names['rod_%d_%d'%(i,j)]],[0,0,c['forearm_length_m']]);anchor,Janchor=tree.point(states[names['platform']],A)
        residual.extend(tip-anchor);rows.extend(Jtip-Janchor)
    return np.asarray(residual),np.asarray(rows)


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts)
    if c['platform_height_m']-.13<=0:raise ValueError('Platform would approach the floor')
    if np.linalg.norm(inverse_kinematics([0,0,c['platform_height_m']],c))>1e-10:raise ValueError('Delta reference branch does not close')
    import mujoco
    from .spatial_tree import SpatialTree
    xml,_=mjcf(c,bodies);model=mujoco.MjModel.from_xml_string(xml)
    tree=SpatialTree.from_mujoco(model).use_body_calibration(bodies);states=tree.forward(model.qpos0,np.zeros(model.nv))
    initial_residual,J=closure(states,tree,c);rank=int(np.linalg.matrix_rank(J))
    _,_,Vh=np.linalg.svd(J,full_matrices=True);null=Vh[rank:].T;actuation=np.zeros((model.nv,3))
    for i in range(3):
        joint=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'drive_%d'%i);actuation[model.jnt_dofadr[joint],i]=1
    actuation_rank=int(np.linalg.matrix_rank(null.T@actuation))
    if model.nv-rank!=3 or actuation_rank!=3 or abs(initial_residual).max()>1e-10:raise ValueError('Delta closure or actuation is singular')
    shape=[{key:p[key] for key in ['body','shape','axis','rotation_matrix','position_m','dimensions_m'] if key in p} for p in parts]
    physical={'config':{k:v for k,v in c.items() if k not in ['assembly_id','seed','hardware_recipe']},'bodies':bodies}
    angle=11*np.pi/6;radial=np.array([np.cos(angle),np.sin(angle),0]);id_position=(c['base_radius_m']+.06+.055+.00005)*radial+[0,0,.004]
    return {'schema_version':'1.0.0','assembly_id':entry['assembly_id'],'category':'H-B','config':c,'parts':parts,'bodies':bodies,
            'geometry_digest':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
            'physical_digest':hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest(),
            'assembly_audit':{'status':'initial_geometry_inertia_and_mobility_validated','tree_dof':model.nv,'closure_rank':rank,'independent_dof':model.nv-rank,
                              'actuation_rank':actuation_rank,'maximum_initial_closure_m':float(abs(initial_residual).max()),
                              'rod_ends':'Six explicitly modeled universal upper and spherical lower joints; no free axial spin is suppressed numerically.',
                              'body_masses_kg':{k:v['mass_kg'] for k,v in bodies.items()}},
            'presentation':{'description':'Spatial rotary Delta robot with three driven upper arms, six paired universal-spherical forearm rods and a translating end platform.',
                            'link_marks':[],'focus_body':'platform','focus_bodies':['platform']+[f'rod_{i}_{j}' for i in range(3) for j in range(2)],
                            'id_mark_position_m':id_position.tolist(),'id_mark_rotation_matrix':(rz(angle+np.pi/2)@rx(np.pi/2)).tolist()},
            'provenance':{'type':'procedural_reconstruction','sources':['delta.py','joint_hardware.py','spatial_solids.py','gear_pair.py'],
                          'change':'New rotary Delta topology, six true U-S rods with hollow composite tubes, open rigid frame and explicit spherical sockets.'},
            'physics_scope':'Three spatial translation DOFs enforced by six parallel forearm constraints. All moving joint hardware contributes to inertia; pose playback uses raw MuJoCo transforms.'}
