"""Six-UPS spatial Stewart platform with explicit universal-joint hardware."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

from .solids import aggregate
from .spatial_solids import part,rx,ry,rz


def skew(a):
    x,y,z=a
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def quaternion_matrix(q):
    w,x,y,z=np.asarray(q)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def configuration(entry):
    rng=np.random.default_rng(entry['seed']);R=float(rng.uniform(.26,.33));r=R*float(rng.uniform(.52,.61))
    height=float(rng.uniform(.43,.53));beta=float(rng.uniform(.15,.22));alpha=float(rng.uniform(.18,.25))
    b_angles=np.array([-beta,beta,2*np.pi/3-beta,2*np.pi/3+beta,4*np.pi/3-beta,4*np.pi/3+beta])
    a_angles=np.array([-np.pi/3+alpha,np.pi/3-alpha,np.pi/3+alpha,np.pi-alpha,np.pi+alpha,5*np.pi/3-alpha])
    B=np.column_stack([R*np.cos(b_angles),R*np.sin(b_angles),np.full(6,.083)])
    A=np.column_stack([r*np.cos(a_angles),r*np.sin(a_angles),np.zeros(6)])
    vectors=A+np.array([0,0,height])-B
    return {'assembly_id':entry['assembly_id'],'category':'H-A','seed':entry['seed'],
            'hardware_recipe':entry['hardware_recipe'],'base_radius_m':R,'platform_anchor_radius_m':r,
            'platform_height_m':height,'base_anchors_m':B.tolist(),'platform_anchors_m':A.tolist(),
            'initial_leg_lengths_m':np.linalg.norm(vectors,axis=1).tolist(),
            'leg_yaw_rad':np.arctan2(vectors[:,1],vectors[:,0]).tolist(),
            'leg_pitch_rad':np.arctan2(np.linalg.norm(vectors[:,:2],axis=1),vectors[:,2]).tolist(),
            'gravity_m_s2':9.81,'payload_mass_kg':float(rng.uniform(.30,.65)),
            'ball_radius_m':.010,'rod_radius_m':.0055,'joint_clearance_m':.000025,
            'linear_armature_kg':.025,'hinge_viscosity_Nm_s_rad':.012,'hinge_coulomb_Nm':.002,
            'slide_viscosity_N_s_m':12.,'slide_coulomb_N':.30,
            'external_force_world_N':[.25,0.,-.5],
            'actuator':{'type':'six_force_limited_linear_position_servos','position_gain_N_m':2500.,
                        'velocity_gain_N_s_m':85.,'force_limit_N':150.,'ramp_s':.7},
            'simulation':{'timestep_s':1/4800,'record_hz':240,'fps':30,'duration_s':4.,
                          'width':640,'height':640,'integrator':'RK4','equality_solref_s':.001,'equality_impedance':.999}}


def inventory(c):
    parts=[];YtoZ=rx(np.pi/2);head_up=rx(-np.pi/2)
    def add(name,body,shape,pos,dims,mat,R=None,density=None):
        parts.append(part(name,body,shape,pos,dims,mat,R,density))
    baseR=c['base_radius_m'];topR=c['platform_anchor_radius_m']+.040
    add('base_plate','ground','ring',[0,0,.026],{'radius':baseR+.075,'inner_radius':.060,'depth':.020},'aluminum',YtoZ)
    add('base_inner_trim','ground','ring',[0,0,.037],{'radius':baseR-.058,'inner_radius':.067,'depth':.002},'frame',YtoZ)
    for i,angle in enumerate(np.linspace(0,2*np.pi,3,endpoint=False)+np.pi/3):
        x,y=(baseR+.023)*np.array([np.cos(angle),np.sin(angle)])
        add('base_foot_%d'%i,'ground','cylinder',[x,y,.008],{'radius':.028,'depth':.016},'graphite',YtoZ)
        add('base_foot_bolt_%d'%i,'ground','socket_head',[x,y,.039],{'radius':.006,'depth':.006},'steel',head_up)
    add('platform_plate','platform','ring',[0,0,.045],{'radius':topR,'inner_radius':.026,'depth':.020},'blue',YtoZ)
    add('platform_top_rim','platform','ring',[0,0,.056],{'radius':topR-.004,'inner_radius':topR-.012,'depth':.002},'steel',YtoZ)
    # Payload is offset to excite rotational/translational coupling and touches the top plate.
    add('payload_mount','platform','box',[.01,-.012,.058],{'size':[.094,.080,.006]},'aluminum')
    add('calibrated_payload','platform','box',[.01,-.012,.078],{'size':[.070,.055,.034]},'graphite',
        density=c['payload_mass_kg']/(.070*.055*.034))
    for dx in [-.039,.039]:
        for dy in [-.032,.032]:
            add('payload_bolt_'+str((dx,dy)),'platform','socket_head',[.01+dx,-.012+dy,.063],
                {'radius':.0035,'depth':.004},'steel',head_up)
    for i,(anchor,upper,L,yaw) in enumerate(zip(c['base_anchors_m'],c['platform_anchors_m'],
                                               c['initial_leg_lengths_m'],c['leg_yaw_rad'])):
        B=np.asarray(anchor);A=np.asarray(upper);Rz=rz(yaw);cross='cross_%d'%i;barrel='barrel_%d'%i;rod='rod_%d'%i
        def ground(name,shape,local,dims,mat,rotation=None):
            add('%02d_'%i+name,'ground',shape,B+Rz@local,dims,mat,Rz@(np.eye(3) if rotation is None else rotation))
        ground('mount_foot','box',[0,0,-.0335],{'size':[.065,.048,.027]},'frame')
        for sign in [-1,1]:
            ground('clevis_ear_'+str(sign),'bearing_block',[sign*.017,0,0],
                   {'size':[.016,.010,.040],'bore_radius':.004525},'aluminum',rz(np.pi/2))
            add('%02d_cross_x_pin_%d'%(i,sign),cross,'cylinder',[sign*.0155,0,0],
                {'radius':.0045,'depth':.013},'steel',rz(np.pi/2))
            add('%02d_cross_x_cap_%d'%(i,sign),cross,'socket_head',[sign*.0245,0,0],
                {'radius':.0065,'depth':.005},'steel',rz(sign*np.pi/2))
            add('%02d_cross_y_pin_%d'%(i,sign),cross,'cylinder',[0,sign*.0155,0],
                {'radius':.0045,'depth':.013},'steel')
            add('%02d_cross_y_cap_%d'%(i,sign),cross,'socket_head',[0,sign*.0245,0],
                {'radius':.0065,'depth':.005},'steel',rz(np.pi if sign==1 else 0))
            add('%02d_fork_ear_%d'%(i,sign),barrel,'bearing_block',[0,sign*.017,0],
                {'size':[.014,.010,.020],'bore_radius':.004525},'aluminum')
            add('%02d_fork_stem_%d'%(i,sign),barrel,'box',[0,sign*.017,.027],
                {'size':[.014,.010,.034]},'aluminum')
            ground('mount_bolt_'+str(sign),'socket_head',[sign*.027,0,-.018],
                   {'radius':.0035,'depth':.004},'steel',head_up)
        add('%02d_universal_cross'%i,cross,'box',[0,0,0],{'size':[.018,.018,.018]},'steel')
        add('%02d_fork_bridge'%i,barrel,'box',[0,0,.049],{'size':[.025,.044,.010]},'aluminum')
        add('%02d_actuator_end_cap'%i,barrel,'cylinder',[0,0,.059],{'radius':.019,'depth':.010},'steel',YtoZ)
        end=.66*L
        add('%02d_actuator_barrel'%i,barrel,'ring',[0,0,(.064+end)/2],
            {'radius':.019,'inner_radius':.010,'depth':end-.064},'blue',YtoZ)
        add('%02d_rod_guide'%i,barrel,'ring',[0,0,end+.006],
            {'radius':.021,'inner_radius':c['rod_radius_m']+.00003,'depth':.012},'bronze',YtoZ)
        for z in [.078,end-.014]:
            add('%02d_barrel_band_'%i+str(z),barrel,'ring',[0,0,z],
                {'radius':.020,'inner_radius':.019,'depth':.004},'steel',YtoZ)
        bottom=-.55*L;ball_lower=-np.sqrt(c['ball_radius_m']**2-c['rod_radius_m']**2)
        add('%02d_piston_head'%i,rod,'cylinder',[0,0,bottom+.004],{'radius':.00995,'depth':.008},'steel',YtoZ)
        add('%02d_piston_rod'%i,rod,'cylinder',[0,0,(bottom+.008+ball_lower)/2],
            {'radius':c['rod_radius_m'],'depth':ball_lower-bottom-.008},'steel',YtoZ)
        add('%02d_spherical_tip'%i,rod,'axial_profile',[0,0,0],
            {'profile_type':'sphere_segment','radius':c['ball_radius_m'],'lower':float(ball_lower),'upper':c['ball_radius_m']},'steel',YtoZ)
        add('%02d_spherical_socket'%i,'platform','axial_profile',A,
            {'profile_type':'ball_socket','outer_radius':.016,'cavity_radius':c['ball_radius_m']+.00004,'lower':-.003,'upper':.026},'bronze',YtoZ)
        add('%02d_socket_mount'%i,'platform','cylinder',A+[0,0,.0305],{'radius':.016,'depth':.009},'aluminum',YtoZ)
        radial=A/np.linalg.norm(A)
        for sign in [-1,1]:
            add('%02d_socket_bolt_%d'%(i,sign),'platform','socket_head',A+radial*(sign*.022)+[0,0,.057],
                {'radius':.0035,'depth':.004},'steel',head_up)
    return parts


def solid_properties(parts):
    return {body:aggregate(parts,body) for body in ['platform']+[name+'_%d'%i for i in range(6) for name in ['cross','barrel','rod']]}


def initial_qpos(c):return np.r_[0.,0.,c['platform_height_m'],1.,0.,0.,0.,np.zeros(18)]


def kinematics(q,v,c,bodies):
    """Analytic rigid-body Jacobians and convective accelerations, no MuJoCo calls."""
    result={};R=quaternion_matrix(q[3:7]);nv=24
    for name in ['platform']:
        local=np.asarray(bodies[name]['com_local_m']);Jv=np.zeros((3,nv));Jw=np.zeros((3,nv))
        Jv[:,:3]=np.eye(3);Jv[:,3:6]=-R@skew(local);Jw[:,3:6]=R
        result[name]={'position':q[:3],'rotation':R,'com':q[:3]+R@local,'Jv':Jv,'Jw':Jw,
                      'acceleration_bias':R@np.cross(v[3:6],np.cross(v[3:6],local)),
                      'angular_acceleration_bias':np.zeros(3),'omega':R@v[3:6]}
    for i,(anchor,L,yaw,pitch) in enumerate(zip(c['base_anchors_m'],c['initial_leg_lengths_m'],c['leg_yaw_rad'],c['leg_pitch_rad'])):
        B=np.asarray(anchor);qx,qy,s=q[7+3*i:10+3*i];vx,vy,vs=v[6+3*i:9+3*i]
        R0=rz(yaw);Rc=R0@rx(qx);Rl=Rc@ry(pitch+qy)
        ax=R0[:,0];ay=Rc[:,1];az=Rl[:,2];omega=ax*vx+ay*vy;alpha=np.cross(ax*vx,ay*vy)
        for label in ['cross','barrel','rod']:
            name=label+'_%d'%i;Jv=np.zeros((3,nv));Jw=np.zeros((3,nv));j=6+3*i
            rot=Rc if label=='cross' else Rl
            offset=np.zeros(3) if label!='rod' else np.array([0,0,L+s])
            radial=rot@(offset+np.asarray(bodies[name]['com_local_m']))
            w=ax*vx if label=='cross' else omega;aa=np.zeros(3) if label=='cross' else alpha
            Jv[:,j]=np.cross(ax,radial);Jw[:,j]=ax
            if label!='cross':Jv[:,j+1]=np.cross(ay,radial);Jw[:,j+1]=ay
            bias=np.cross(aa,radial)+np.cross(w,np.cross(w,radial))
            if label=='rod':Jv[:,j+2]=az;bias+=2*np.cross(w,az*vs)
            result[name]={'position':B+rot@offset,'rotation':rot,'com':B+radial,'Jv':Jv,'Jw':Jw,
                          'acceleration_bias':bias,'angular_acceleration_bias':aa,'omega':w}
    return result


def mass_bias_energy(q,v,c,bodies,kin=None):
    kin=kinematics(q,v,c,bodies) if kin is None else kin
    M=np.zeros((24,24));bias=np.zeros(24);V=0.;gravity=np.array([0,0,-c['gravity_m_s2']])
    for name,b in bodies.items():
        k=kin[name];mass=b['mass_kg'];R=k['rotation'];Iw=R@np.asarray(b['inertia_com_local_kg_m2'])@R.T
        Jv,Jw=k['Jv'],k['Jw'];M+=mass*Jv.T@Jv+Jw.T@Iw@Jw
        bias+=Jv.T@(mass*(k['acceleration_bias']-gravity))+Jw.T@(Iw@k['angular_acceleration_bias']+np.cross(k['omega'],Iw@k['omega']))
        V-=mass*gravity@k['com']
    for i in range(6):M[8+3*i,8+3*i]+=c['linear_armature_kg']
    return M,bias,float(.5*v@M@v),float(V)


def closure_jacobian(q,c):
    R=quaternion_matrix(q[3:7]);J=np.zeros((18,24));residual=[]
    for i,(B,A,L,yaw,pitch) in enumerate(zip(c['base_anchors_m'],c['platform_anchors_m'],c['initial_leg_lengths_m'],c['leg_yaw_rad'],c['leg_pitch_rad'])):
        qx,qy,s=q[7+3*i:10+3*i];R0=rz(yaw);Rc=R0@rx(qx);Rl=Rc@ry(pitch+qy)
        radial=Rl@np.array([0,0,L+s]);rows=slice(3*i,3*i+3);j=6+3*i
        J[rows,:3]=-np.eye(3);J[rows,3:6]=R@skew(A)
        J[rows,j]=np.cross(R0[:,0],radial);J[rows,j+1]=np.cross(Rc[:,1],radial);J[rows,j+2]=Rl[:,2]
        residual.extend(np.asarray(B)+radial-q[:3]-R@A)
    return np.asarray(residual),J


def friction(v,c):
    out=np.zeros(24)
    for i in range(6):
        j=6+3*i
        out[j:j+2]=-c['hinge_viscosity_Nm_s_rad']*v[j:j+2]-c['hinge_coulomb_Nm']*np.tanh(v[j:j+2]/.03)
        out[j+2]=-c['slide_viscosity_N_s_m']*v[j+2]-c['slide_coulomb_N']*np.tanh(v[j+2]/.005)
    return out


def external(c):return np.r_[c['external_force_world_N'],np.zeros(21)]


def mjcf(c,bodies):
    vec=lambda a:' '.join('%.16g'%v for v in np.ravel(a))
    quat_y=lambda a:vec([np.cos(a/2),0,np.sin(a/2),0])
    quat_z=lambda a:vec([np.cos(a/2),0,0,np.sin(a/2)])
    root=ET.Element('mujoco',model=c['assembly_id']);ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false')
    s=c['simulation'];option=ET.SubElement(root,'option',timestep=str(s['timestep_s']),gravity=vec([0,0,-c['gravity_m_s2']]),
                                          integrator=s['integrator'],solver='Newton',jacobian='dense',iterations='100',tolerance='1e-12')
    ET.SubElement(option,'flag',energy='enable');default=ET.SubElement(root,'default')
    ET.SubElement(default,'geom',contype='0',conaffinity='0')
    world=ET.SubElement(root,'worldbody');platform=ET.SubElement(world,'body',name='platform',pos=vec([0,0,c['platform_height_m']]))
    ET.SubElement(platform,'freejoint',name='platform_free');nodes={'platform':platform}
    for i,(B,L,yaw,pitch) in enumerate(zip(c['base_anchors_m'],c['initial_leg_lengths_m'],c['leg_yaw_rad'],c['leg_pitch_rad'])):
        cross=ET.SubElement(world,'body',name='cross_%d'%i,pos=vec(B),quat=quat_z(yaw))
        ET.SubElement(cross,'joint',name='universal_x_%d'%i,type='hinge',axis='1 0 0')
        lower=ET.SubElement(cross,'body',name='barrel_%d'%i,quat=quat_y(pitch))
        ET.SubElement(lower,'joint',name='universal_y_%d'%i,type='hinge',axis='0 1 0')
        upper=ET.SubElement(lower,'body',name='rod_%d'%i,pos=vec([0,0,L]))
        ET.SubElement(upper,'joint',name='prismatic_%d'%i,type='slide',axis='0 0 1',armature=str(c['linear_armature_kg']))
        nodes.update({'cross_%d'%i:cross,'barrel_%d'%i:lower,'rod_%d'%i:upper})
    for name,node in nodes.items():
        b=bodies[name];I=np.asarray(b['inertia_com_local_kg_m2'])
        principal,axes=np.linalg.eigh(I)
        if np.linalg.det(axes)<0:axes[:,0]*=-1
        orientation=Rotation.from_matrix(axes).as_quat()[[3,0,1,2]]
        ET.SubElement(node,'inertial',pos=vec(b['com_local_m']),mass=str(b['mass_kg']),
                      diaginertia=vec(principal),quat=vec(orientation))
        if name=='platform':ET.SubElement(node,'geom',type='cylinder',pos='0 0 .045',size=vec([c['platform_anchor_radius_m']+.04,.010]))
        elif name.startswith('cross'):ET.SubElement(node,'geom',type='box',size='.009 .009 .009')
        elif name.startswith('barrel'):
            L=c['initial_leg_lengths_m'][int(name.split('_')[-1])]
            ET.SubElement(node,'geom',type='cylinder',pos=vec([0,0,(.064+.66*L)/2]),size=vec([.019,(.66*L-.064)/2]))
        else:ET.SubElement(node,'geom',type='sphere',size=str(c['ball_radius_m']))
    eq=ET.SubElement(root,'equality');act=ET.SubElement(root,'actuator')
    for i in range(6):
        ET.SubElement(eq,'connect',name='ball_closure_%d'%i,body1='rod_%d'%i,body2='platform',anchor='0 0 0',
                      solref=vec([s['equality_solref_s'],1]),solimp=vec([s['equality_impedance'],s['equality_impedance'],.001,.5,2]))
        limit=c['actuator']['force_limit_N']
        ET.SubElement(act,'motor',name='linear_drive_%d'%i,joint='prismatic_%d'%i,ctrllimited='true',ctrlrange=vec([-limit,limit]))
    return ET.tostring(root,encoding='unicode'),initial_qpos(c)


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts);q=initial_qpos(c)
    residual,J=closure_jacobian(q,c);rank=int(np.linalg.matrix_rank(J))
    M,bias,_,_=mass_bias_energy(q,np.zeros(24),c,bodies)
    if rank!=18 or np.linalg.norm(residual)>1e-12 or np.linalg.eigvalsh(M).min()<=0:raise ValueError('Invalid UPS model')
    # Check the six spatial actuation directions after eliminating closed-chain coordinates.
    _,_,Vh=np.linalg.svd(J,full_matrices=True);null=Vh[rank:].T
    actuation=np.zeros((24,6));actuation[np.arange(6)*3+8,np.arange(6)]=1
    if np.linalg.matrix_rank(null.T@actuation)!=6:raise ValueError('Stewart actuation singularity')
    shape=[{key:p[key] for key in ['body','shape','rotation_matrix','position_m','dimensions_m']} for p in parts]
    physical={'config':{k:v for k,v in c.items() if k not in ['assembly_id','seed','hardware_recipe']},'bodies':bodies}
    return {'schema_version':'1.0.0','assembly_id':entry['assembly_id'],'category':'H-A','config':c,'parts':parts,'bodies':bodies,
            'geometry_digest':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
            'physical_digest':hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest(),
            'assembly_audit':{'status':'initial_geometry_and_inertia_validated','independent_dof':24-rank,'tree_dof':24,'closure_rank':rank,
                              'actuation_rank':6,'initial_closure_m':float(np.linalg.norm(residual)),
                              'moving_hardware_mass_included':True,'universal_pin_radial_clearance_m':.000025,
                              'spherical_socket_radial_clearance_m':.00004,'rod_guide_radial_clearance_m':.00003,
                              'body_masses_kg':{k:v['mass_kg'] for k,v in bodies.items()}},
            'presentation':{'description':'Six-degree-of-freedom spatial Stewart platform with six UPS legs, two-axis universal joints, force-driven telescopic actuators and spherical platform joints.',
                            'link_marks':[],'focus_body':'platform','focus_local_m':[0,0,.02],
                            'closeup_scale_m':2*(c['platform_anchor_radius_m']+.07)*1.25,
                            'id_mark_position_m':[0,-c['base_radius_m']-.075-.00005,.022]},
            'provenance':{'type':'procedural_reconstruction','sources':['stewart.py','spatial_solids.py'],
                          'related_open_reference':'https://github.com/BerkeleyAutomation/sprk',
                          'change':'Original UPS reconstruction; downloaded SPRK is a rotary-actuated reference and its CAD is not claimed as this linear-actuator model.'},
            'physics_scope':'Rigid 6-UPS platform, explicit massive universal crosses and hollow telescopic parts; finite motor forces, linear/rotational friction and compliant numerical loop closure.'}
