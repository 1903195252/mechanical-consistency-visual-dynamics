"""R-R-R-P closed chain with bored crosshead and two physical guide rails."""
import hashlib
import json
import math
import xml.etree.ElementTree as ET
import numpy as np

from . import fourbar
from .solids import aggregate,part

BODY_NAMES=['crank','coupler','slider']
TREE_TO_ABSOLUTE=np.array([[1.,0.,0.],[1.,1.,0.],[0.,0.,1.]])
JOINT_RATE_MAP=np.array([[1.,0.,0.],[-1.,1.,0.],[0.,-1.,0.],[0.,0.,1.]])
direction=fourbar.direction
tangent=fourbar.tangent


def configuration(entry):
    c=fourbar.configuration(entry);rng=np.random.default_rng(entry['seed']+91)
    r=float(rng.uniform(.065,.105));L=r*float(rng.uniform(2.9,3.6))
    c.update(category='S-B',lengths_m=[r,L,L*.75],pivot_height_m=max(.26,r+.15),
             guide_offset_m=float(rng.uniform(-.018,.025)),
             viscous_friction_Nm_s_rad=[.02,.003,.003,1.2],
             coulomb_friction_Nm=[.003,.002,.002,.20],
             coordinate_units=['rad','rad','m'],joint_rate_units=['rad/s','rad/s','rad/s','m/s'],
             friction_coefficient_units=['Nm s/rad','Nm s/rad','Nm s/rad','N s/m'],
             coulomb_force_units=['Nm','Nm','Nm','N'],
             payload_along_coupler_m=L*.61,force_application_along_coupler_m=L*.8,
             friction_smoothing_rates=[.05,.05,.05,.005])
    c['viscous_friction_coefficients']=c.pop('viscous_friction_Nm_s_rad')
    c['coulomb_friction_coefficients']=c.pop('coulomb_friction_Nm')
    positions=np.array([inverse_kinematics(a,c)[2] for a in np.linspace(0,2*np.pi,721)])
    c['slider_travel_m']=[float(positions.min()),float(positions.max())]
    c['rail_support_x_m']=[c['slider_travel_m'][0]-.070,c['slider_travel_m'][1]+.070]
    c['ground_length_m']=c['rail_support_x_m'][1]+.030
    return c


def inverse_kinematics(angle,c):
    r,L=c['lengths_m'][:2];s=(c['guide_offset_m']-r*np.sin(angle))/L
    if abs(s)>=1:raise ValueError('Unreachable slider-crank')
    phi=np.arcsin(s);x=r*np.cos(angle)+L*np.cos(phi)
    return np.array([angle,phi,x])


def closure(q,c):
    r,L=c['lengths_m'][:2]
    return r*direction(q[0])+L*direction(q[1])-np.array([q[2],c['guide_offset_m']])


def closure_jacobian(q,c):
    r,L=c['lengths_m'][:2]
    return np.column_stack([r*tangent(q[0]),L*tangent(q[1]),[-1.,0.]])


def closure_curvature(q,v,c):
    r,L=c['lengths_m'][:2]
    return -r*direction(q[0])*v[0]**2-L*direction(q[1])*v[1]**2


def inventory(c):
    original=fourbar.inventory(c)
    parts=[p for p in original if p['body'] in ['crank','coupler'] or
           (p['body']=='ground' and not p['name'].startswith('D_') and p['name'] not in ['ground_tie','base_inlay'])]
    def add(name,body,shape,pos,dims,material,**kw):
        parts.append(part(name,body,shape,pos,dims,material,**kw))
    t=c['bar_depth_m'];bore=c['joint_bore_radius_m'];r=c['shaft_radius_m'];layer=c['visual_and_com_y_m'][2]
    add('crosshead_pin_block','slider','bearing_block',[0,layer,0],
        {'size':[.050,t,.040],'bore_radius':bore},'green')
    add('crosshead_pin_sleeve','slider','ring',[0,layer,0],
        {'radius':bore,'inner_radius':r+.000025,'depth':t},'bronze')
    add('crosshead_pin_flange','slider','ring',[0,layer-t/2-.0005,0],
        {'radius':bore*1.45,'inner_radius':r+.000025,'depth':.001},'bronze')
    saddle_front=layer+t/2;saddle_back=.124
    add('crosshead_saddle','slider','box',[0,(saddle_front+saddle_back)/2,-.0145],
        {'size':[.055,saddle_back-saddle_front,.009]},'green')
    for y in [.055,.105]:
        add('guide_carriage_'+str(y),'slider','rectangular_sleeve',[0,y,-.030],
            {'size':[.055,.026,.022],'opening_yz':[.0141,.0101]},'bronze')
        for x in [-.018,.018]:
            add('carriage_screw_'+str((x,y)),'slider','socket_head',[x,y,-.0085],
                {'radius':.0025,'depth':.003},'steel',axis='Z')
    h=c['pivot_height_m']+c['guide_offset_m'];left,right=c['rail_support_x_m']
    for y in [.055,.105]:
        add('precision_guide_'+str(y),'ground','box',[(left+right)/2,y,h-.030],
            {'size':[right-left-.040,.014,.010]},'steel')
    for x in [left,right]:
        add('guide_support_foot_'+str(x),'ground','box',[x,.08,.049],
            {'size':[.075,.140,.020]},'aluminum')
        top=h-.017
        add('guide_support_column_'+str(x),'ground','box',[x,.08,(top+.059)/2],
            {'size':[.040,.110,top-.059]},'frame')
        for dx in [-.028,.028]:
            for y in [.028,.132]:
                add('guide_foot_screw_'+str((x,dx,y)),'ground','socket_head',[x+dx,y,.0615],
                    {'radius':.003,'depth':.005},'steel',axis='Z')
    add('base_inlay','ground','box',[c['ground_length_m']/2,-.03,.040],
        {'size':[c['ground_length_m']-.10,.045,.002]},'frame')
    return parts


def solid_properties(parts):return {name:aggregate(parts,name) for name in BODY_NAMES}


def mass_bias_energy(q,v,c,bodies):
    M=np.zeros((3,3));C=np.zeros(3);G=np.zeros(3);V=0.
    for i,name in enumerate(BODY_NAMES):
        b=bodies[name];mass=b['mass_kg'];rx,_,rz=b['com_local_m'];J=np.zeros((2,3))
        if name=='slider':
            J[:,2]=[1,0];centripetal=np.zeros(2);height=c['pivot_height_m']+c['guide_offset_m']+rz
        else:
            local=rx*direction(q[i])+rz*tangent(q[i]);J[:,i]=rx*tangent(q[i])-rz*direction(q[i])
            centripetal=-local*v[i]**2;height=c['pivot_height_m']+local[1]
            if name=='coupler':
                J[:,0]+=c['lengths_m'][0]*tangent(q[0])
                centripetal-=c['lengths_m'][0]*direction(q[0])*v[0]**2
                height+=c['lengths_m'][0]*np.sin(q[0])
            M[i,i]+=b['inertia_com_local_kg_m2'][1][1]
        M+=mass*J.T@J;C+=mass*J.T@centripetal;G+=mass*c['gravity_m_s2']*J[1]
        V+=mass*c['gravity_m_s2']*height
    M[0,0]+=c['motor_armature_kg_m2']
    return M,C,G,float(.5*v@M@v),float(V)


def friction(v,c):
    rates=JOINT_RATE_MAP@v
    return -JOINT_RATE_MAP.T@(np.asarray(c['viscous_friction_coefficients'])*rates+
                             np.asarray(c['coulomb_friction_coefficients'])*np.tanh(rates/c['friction_smoothing_rates']))


external=fourbar.external


def geometry_audit(c,parts,bodies):
    qs=np.array([inverse_kinematics(a,c) for a in np.linspace(-np.pi,np.pi,361)])
    residual=max(np.linalg.norm(closure(q,c)) for q in qs)
    ranks={int(np.linalg.matrix_rank(closure_jacobian(q,c))) for q in qs}
    left,right=c['rail_support_x_m']
    end_clearance=min(qs[:,2].min()-.055/2-(left+.020),right-.020-(qs[:,2].max()+.055/2))
    if residual>1e-10 or ranks!={2} or end_clearance<=0:raise ValueError('Invalid crank-slider assembly')
    if len({p['name'] for p in parts})!=len(parts):raise ValueError('Repeated physical part')
    return {'status':'geometry_and_inertia_validated','independent_dof':1,'tree_dof':3,'closure_rank':2,
            'full_rotation_reachable':True,'sampled_full_rotation_configurations':len(qs),
            'maximum_analytic_closure_m':float(residual),'carriage_end_clearance_m':float(end_clearance),
            'guide_radial_clearance_yz_m':[.00005,.00005],'moving_hardware_mass_included':True,
            'body_masses_kg':{name:b['mass_kg'] for name,b in bodies.items()}}


def mjcf(c,bodies):
    q0=inverse_kinematics(c['initial_crank_angle_rad'],c)
    vec=lambda a:' '.join('%.16g'%v for v in np.ravel(a))
    quat=lambda angle:vec([np.cos(angle/2),0,-np.sin(angle/2),0])
    root=ET.Element('mujoco',model=c['assembly_id']);ET.SubElement(root,'compiler',angle='radian',inertiafromgeom='false')
    s=c['simulation']
    opt=ET.SubElement(root,'option',timestep=str(s['timestep_s']),gravity=vec([0,0,-c['gravity_m_s2']]),
                      integrator=s['integrator'],solver='Newton',iterations='100',tolerance='1e-12',jacobian='dense')
    ET.SubElement(opt,'flag',energy='enable');default=ET.SubElement(root,'default')
    ET.SubElement(default,'joint',damping='0',frictionloss='0',limited='false')
    ET.SubElement(default,'geom',contype='0',conaffinity='0')
    assets=ET.SubElement(root,'asset')
    for i,L in enumerate(c['lengths_m'][:2]):
        vertices,faces=fourbar.link_mesh(L,c['bar_height_m'],c['bar_depth_m'],c['joint_bore_radius_m'])
        ET.SubElement(assets,'mesh',name='plate_%d'%i,vertex=vec(vertices),face=' '.join(str(v) for face in faces for v in face))
    world=ET.SubElement(root,'worldbody')
    a=ET.SubElement(world,'body',name='crank',pos=vec([0,0,c['pivot_height_m']]),quat=quat(q0[0]))
    ET.SubElement(a,'joint',name='A',type='hinge',axis='0 -1 0',armature=str(c['motor_armature_kg_m2']))
    b=ET.SubElement(a,'body',name='coupler',pos=vec([c['lengths_m'][0],0,0]),quat=quat(q0[1]-q0[0]))
    ET.SubElement(b,'joint',name='B',type='hinge',axis='0 -1 0')
    p=ET.SubElement(world,'body',name='slider',pos=vec([q0[2],0,c['pivot_height_m']+c['guide_offset_m']]))
    ET.SubElement(p,'joint',name='guide',type='slide',axis='1 0 0')
    for i,(name,node) in enumerate(zip(BODY_NAMES,[a,b,p])):
        prop=bodies[name];I=np.asarray(prop['inertia_com_local_kg_m2'])
        ET.SubElement(node,'inertial',pos=vec(prop['com_local_m']),mass=str(prop['mass_kg']),
                      fullinertia=vec([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        if i<2:ET.SubElement(node,'geom',type='mesh',mesh='plate_%d'%i,pos=vec([0,c['visual_and_com_y_m'][i],0]))
        else:ET.SubElement(node,'geom',type='box',size=vec([.025,c['bar_depth_m']/2,.020]),pos=vec([0,c['visual_and_com_y_m'][2],0]))
    equality=ET.SubElement(root,'equality')
    ET.SubElement(equality,'connect',name='crosshead_pin',body1='coupler',body2='slider',anchor=vec([c['lengths_m'][1],0,0]),
                  solref=vec([s['equality_solref_s'],1]),solimp=vec([s['equality_impedance'],s['equality_impedance'],.001,.5,2]))
    actuator=ET.SubElement(root,'actuator');limit=c['actuator']['torque_limit_Nm']
    ET.SubElement(actuator,'motor',name='crank_motor',joint='A',gear='1',ctrllimited='true',ctrlrange=vec([-limit,limit]))
    return ET.tostring(root,encoding='unicode'),q0


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts);audit=geometry_audit(c,parts,bodies)
    shape=[{key:p[key] for key in ['body','shape','axis','position_m','dimensions_m','rotation_matrix'] if key in p} for p in parts]
    physical={'config':{key:value for key,value in c.items() if key not in ['assembly_id','seed','hardware_recipe']},'bodies':bodies}
    return {'schema_version':'1.0.0','assembly_id':entry['assembly_id'],'category':'S-B','config':c,
            'parts':parts,'bodies':bodies,'geometry_digest':hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest(),
            'physical_digest':hashlib.sha256(json.dumps(physical,sort_keys=True).encode()).hexdigest(),'assembly_audit':audit,
            'presentation':{'description':'Offset closed-chain slider-crank with a torque-driven crank, a bored connecting rod, crosshead and two precision linear guides.',
                            'link_marks':[['crank','AB',0],['coupler','BC',1]],'focus_body':'coupler',
                            'focus_local_m':[c['lengths_m'][1]/2,0,0]},
            'provenance':{'type':'procedural_reconstruction','sources':['fourbar.py','slider_crank.py'],
                          'change':'New R-R-R-P topology, two through-bored sliding carriages, parallel rails and fully accounted hardware.'},
            'physics_scope':'Ideal revolute/prismatic constraints; explicitly dissipative rotational and linear guide friction. Nominal solids include all moving hardware.'}
