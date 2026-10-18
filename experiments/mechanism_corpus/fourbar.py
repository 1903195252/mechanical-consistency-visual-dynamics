"""Parameterized 4R assembly with explicit moving hardware mass and SI moments."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

from .catalog import ROOT
from .solids import aggregate, part

SOURCE=ROOT/"experiments/fourbar"
if str(SOURCE) not in sys.path:sys.path.insert(0,str(SOURCE))
from mechanics import (direction,tangent,inverse_kinematics,closure,closure_jacobian,closure_curvature,
                       TREE_TO_ABSOLUTE,JOINT_RATE_MAP)
from geometry import link_mesh

BODY_NAMES=["crank","coupler","rocker"]


def configuration(entry):
    rng=np.random.default_rng(entry["seed"])
    d=float(rng.uniform(.34,.50))
    width=float(rng.uniform(.026,.034));thickness=float(rng.uniform(.009,.013))
    bore=float(rng.uniform(.0045,.0055))
    lengths=[d*float(rng.uniform(.25,.31)),d*float(rng.uniform(.88,.96)),d*float(rng.uniform(.67,.75))]
    c={"assembly_id":entry["assembly_id"],"category":"S-A","seed":entry["seed"],"hardware_recipe":entry["hardware_recipe"],
       "ground_length_m":d,"lengths_m":lengths,"bar_height_m":width,"bar_depth_m":thickness,
       "joint_bore_radius_m":bore,"shaft_radius_m":.60*bore,"assembly_branch":1,
       "pivot_height_m":max(.26,lengths[0]+.16),"visual_and_com_y_m":[-2.*thickness,0.,2.*thickness],
       "input_bearing_side":"exterior_front","assembly_revision":"opposed_ground_bearings_02",
       "initial_crank_angle_rad":float(rng.uniform(.35,.95)),"density_kg_m3":2700.,
       "gravity_m_s2":9.81,"motor_armature_kg_m2":.0002,
       "viscous_friction_Nm_s_rad":[.02,.004,.004,.008],
       "coulomb_friction_Nm":[.003,.002,.002,.003],"coulomb_smoothing_rad_s":.05,
       "payload_along_coupler_m":lengths[1]*float(rng.uniform(.5,.66)),
       "payload_size_m":[.044,.012,.023],"payload_mass_kg":float(rng.uniform(.06,.11)),
       "payload_mount_plate_thickness_m":.0015,
       "external_force_xz_N":[0.,-.15],"force_application_along_coupler_m":lengths[1]*.8,
       "actuator":{"type":"finite_torque_velocity_servo","speed_gain_Nm_s_rad":2.0,"torque_limit_Nm":5.0,"ramp_s":.5},
       "simulation":{"timestep_s":1/4800,"record_hz":240,"fps":30,"duration_s":4.,"width":640,"height":640,
                     "integrator":"RK4","equality_solref_s":.001,"equality_impedance":.999}}
    c["payload_y_m"]=-thickness/2-c["payload_mount_plate_thickness_m"]-c["payload_size_m"][1]/2
    return c


def inventory(c):
    p=[];t=c["bar_depth_m"];r=c["shaft_radius_m"];bore=c["joint_bore_radius_m"]
    layers=c["visual_and_com_y_m"];d=c["ground_length_m"];h=c["pivot_height_m"]
    def add(name,body,shape,pos,dims,mat,**kw):
        p.append(part(name,body,shape,pos,dims,mat,**kw))
    for body,L,layer,mat in zip(BODY_NAMES,c["lengths_m"],layers,["blue","orange","green"]):
        add(body+"_plate",body,"link",[0.,layer,0.],{"length":L,"width":c["bar_height_m"],"thickness":t,"bore_radius":bore},mat)
        for label,x in [("proximal",0.),("distal",L)]:
            add(body+"_"+label+"_sleeve",body,"ring",[x,layer,0.],{"radius":bore,"inner_radius":r+.000025,"depth":t},"bronze")
            add(body+"_"+label+"_flange",body,"ring",[x,layer-t/2-.0005,0.],{"radius":bore*1.45,"inner_radius":r+.000025,"depth":.001},"bronze")
    # Each complete axle stack has one rigid-body owner; shared joints are not double-counted.
    joints=[("A","crank",0.,layers[0]-t/2-.001,.091),
            ("B","crank",c["lengths_m"][0],layers[0]-t/2-.001,layers[1]+t/2+.001),
            ("C","coupler",c["lengths_m"][1],layers[1]-t/2-.001,layers[2]+t/2+.001),
            ("D","rocker",0.,layers[2]-t/2-.001,.091)]
    for label,owner,x,front,back in joints:
        cap_depth=.004
        cap_center=front-.002-cap_depth/2
        # The shaft stops at the cap rear face; it cannot fill the blind socket.
        shaft_start=front-.002;shaft_end=back+.007
        add(label+"_shoulder_shaft",owner,"cylinder",[x,(shaft_start+shaft_end)/2,0.],
            {"radius":r,"depth":shaft_end-shaft_start},"steel")
        for side,y in [("front",front-.001),("rear",back+.001)]:
            add(label+"_"+side+"_washer",owner,"ring",[x,y,0.],{"radius":.008,"inner_radius":r+.00004,"depth":.002},"steel")
        add(label+"_socket_cap",owner,"socket_head",[x,cap_center,0.],{"radius":.0055,"depth":cap_depth},"steel")
        add(label+"_retaining_nut",owner,"hex_nut",[x,back+.0045,0.],{"radius":.0055,"inner_radius":r+.00004,"depth":.005},"steel")
    for label,owner,x,lo,hi in [("B","crank",c["lengths_m"][0],layers[0]+t/2,layers[1]-t/2-.001),
                               ("C","coupler",c["lengths_m"][1],layers[1]+t/2,layers[2]-t/2-.001)]:
        add(label+"_interplate_spacer",owner,"ring",[x,(lo+hi)/2,0.],{"radius":bore*1.2,"inner_radius":r+.00004,"depth":hi-lo},"steel")
    for label,owner,layer in [("A","crank",layers[0]),("D","rocker",layers[2])]:
        lo=layer+t/2;hi=.038
        add(label+"_bearing_standoff",owner,"ring",[0.,(lo+hi)/2,0.],{"radius":.008,"inner_radius":r+.00004,"depth":hi-lo},"steel")
    payload_density=c["payload_mass_kg"]/np.prod(c["payload_size_m"])
    add("removable_payload","coupler","box",[c["payload_along_coupler_m"],c["payload_y_m"],0.],
        {"size":c["payload_size_m"]},"graphite",density=float(payload_density))
    plate_t=c["payload_mount_plate_thickness_m"]
    add("payload_mount_plate","coupler","box",[c["payload_along_coupler_m"],-t/2-plate_t/2,0.],
        {"size":[c["payload_size_m"][0]+.010,plate_t,c["payload_size_m"][2]+.002]},"aluminum")
    for dx in [-.016,.016]:
        add("payload_screw_"+str(dx),"coupler","socket_head",
            [c["payload_along_coupler_m"]+dx,c["payload_y_m"]-c["payload_size_m"][1]/2-.0015,0.],
            {"radius":.0022,"depth":.003},"steel")
    add("machined_base","ground","box",[d/2,0.,.025],{"size":[d+.20,.46,.028]},"aluminum")
    add("base_inlay","ground","box",[d/2,.065,.040],{"size":[max(.12,d-.10),.14,.002]},"frame")
    for x in [-.065,d+.065]:
        add("base_foot_"+str(x),"ground","box",[x,0.,.0055],{"size":[.045,.40,.011]},"graphite")
    for label,x in [("A",0.),("D",d)]:
        add(label+"_support_foot","ground","box",[x,.070,.049],{"size":[.100,.11,.02]},"aluminum")
        add(label+"_support_column","ground","box",[x,.076,(h+.024)/2],{"size":[.048,.045,h-.094]},"frame")
        add(label+"_bearing_block","ground","bearing_block",[x,.064,h],{"size":[.070,.052,.070],"bore_radius":.020},"aluminum")
        add(label+"_outer_bearing_race","ground","ring",[x,.064,h],{"radius":.020,"inner_radius":.014,"depth":.052},"steel")
        add(label+"_inner_bearing_race","ground","ring",[x,.064,h],{"radius":.009,"inner_radius":r+.00004,"depth":.052},"steel")
        add(label+"_bearing_seal","ground","ring",[x,.0365,h],{"radius":.019,"inner_radius":.0085,"depth":.003},"graphite")
        for dx in [-.024,.024]:
            for dz in [-.024,.024]:
                add(label+"_bearing_screw_"+str(dx)+"_"+str(dz),"ground","socket_head",[x+dx,.0365,h+dz],
                    {"radius":.003,"depth":.003},"steel")
        for dx in [-.037,.037]:
            for y in [.033,.107]:
                add(label+"_foot_screw_"+str(dx)+"_"+str(y),"ground","socket_head",[x+dx,y,.0615],
                    {"radius":.0035,"depth":.005},"steel",axis="Z")
    add("motor_housing","ground","ring",[0.,.134,h],{"radius":.037,"inner_radius":.009,"depth":.068},"graphite")
    add("motor_flange","ground","ring",[0.,.095,h],{"radius":.041,"inner_radius":.010,"depth":.010},"aluminum")
    add("A_rotor_hub","crank","cylinder",[0.,.113,0.],{"radius":.007,"depth":.030},"steel")
    for y in [.112,.122,.132,.142,.152,.162]:
        add("motor_rib_"+str(y),"ground","ring",[0.,y,h],{"radius":.039,"inner_radius":.037,"depth":.002},"graphite")
    # Put the input bearing and motor outside the crank layer. A shaft routed
    # behind that layer would intersect the coupler at the extended alignment.
    # The output bearing remains behind its own link; neither ground shaft
    # traverses the other moving-link layers.
    reflection=np.diag([1.,-1.,1.])
    for item in p:
        if item["name"].startswith("A_") or item["name"].startswith("motor_"):
            item["position_m"][1]=2*layers[0]-item["position_m"][1]
            item["com_body_m"][1]=2*layers[0]-item["com_body_m"][1]
            item["inertia_com_body_kg_m2"]=(reflection@np.asarray(item["inertia_com_body_kg_m2"])@reflection).tolist()
            if item["shape"]=="socket_head" and item["axis"]=="Y":
                item["rotation_matrix"]=[[-1.,0.,0.],[0.,-1.,0.],[0.,0.,1.]]
    return p


def solid_properties(parts):
    return {name:aggregate(parts,name) for name in BODY_NAMES}


def mass_bias_energy(q,v,c,bodies):
    M=np.zeros((3,3));C=np.zeros(3);G=np.zeros(3);V=0.
    l1=c["lengths_m"][0]
    for i,name in enumerate(BODY_NAMES):
        body=bodies[name];m=body["mass_kg"];rx,_,rz=body["com_local_m"]
        local=rx*direction(q[i])+rz*tangent(q[i])
        J=np.zeros((2,3));J[:,i]=rx*tangent(q[i])-rz*direction(q[i])
        centripetal=-local*v[i]**2
        height=c["pivot_height_m"]+local[1]
        if i==1:
            J[:,0]+=l1*tangent(q[0])
            centripetal-=l1*direction(q[0])*v[0]**2
            height+=l1*np.sin(q[0])
        M+=m*J.T@J
        M[i,i]+=body["inertia_com_local_kg_m2"][1][1]
        C+=m*J.T@centripetal
        G+=m*c["gravity_m_s2"]*J[1]
        V+=m*c["gravity_m_s2"]*height
    M[0,0]+=c["motor_armature_kg_m2"]
    return M,C,G,float(.5*v@M@v),float(V)


def friction(v,c):
    rates=JOINT_RATE_MAP@v
    force=-JOINT_RATE_MAP.T@(np.asarray(c["viscous_friction_Nm_s_rad"])*rates+
                            np.asarray(c["coulomb_friction_Nm"])*np.tanh(rates/c["coulomb_smoothing_rad_s"]))
    return force


def external(q,c):
    J=np.zeros((2,3));J[:,0]=c["lengths_m"][0]*tangent(q[0]);J[:,1]=c["force_application_along_coupler_m"]*tangent(q[1])
    return J.T@np.asarray(c["external_force_xz_N"])


def geometry_audit(c,parts,bodies):
    ranks=[];clearances=[]
    for angle in np.linspace(-np.pi,np.pi,181):
        q=inverse_kinematics(angle,c)
        if np.linalg.norm(closure(q,c))>1e-10:raise ValueError("Invalid loop")
        ranks.append(int(np.linalg.matrix_rank(closure_jacobian(q,c))))
    gaps=np.diff(c["visual_and_com_y_m"])-c["bar_depth_m"]
    assert np.all(gaps>0) and set(ranks)=={2}
    assert len({p["name"] for p in parts})==len(parts)
    return {"independent_dof":1,"tree_dof":3,"closure_rank":2,
            "sampled_full_rotation_configurations":181,"full_rotation_reachable":True,
            "interplate_axial_gaps_m":gaps.tolist(),"pin_to_sleeve_radial_clearance_m":.000025,
            "moving_hardware_mass_included":True,"all_body_inertias_positive":True,
            "body_masses_kg":{k:v["mass_kg"] for k,v in bodies.items()},
            "status":"geometry_and_inertia_validated"}


def mjcf(c,bodies):
    q0=inverse_kinematics(c["initial_crank_angle_rad"],c)
    vec=lambda a:" ".join("%.16g"%v for v in np.ravel(a))
    quat=lambda angle:vec([np.cos(angle/2),0,-np.sin(angle/2),0])
    root=ET.Element("mujoco",model=c["assembly_id"])
    ET.SubElement(root,"compiler",angle="radian",inertiafromgeom="false")
    s=c["simulation"]
    opt=ET.SubElement(root,"option",timestep=str(s["timestep_s"]),gravity=vec([0,0,-c["gravity_m_s2"]]),
                      integrator=s["integrator"],solver="Newton",iterations="100",tolerance="1e-12",jacobian="dense")
    ET.SubElement(opt,"flag",energy="enable")
    default=ET.SubElement(root,"default")
    ET.SubElement(default,"joint",type="hinge",axis="0 -1 0",damping="0",frictionloss="0",limited="false")
    ET.SubElement(default,"geom",contype="0",conaffinity="0")
    assets=ET.SubElement(root,"asset")
    for i,L in enumerate(c["lengths_m"]):
        vertices,faces=link_mesh(L,c["bar_height_m"],c["bar_depth_m"],c["joint_bore_radius_m"])
        ET.SubElement(assets,"mesh",name="plate_%d"%i,vertex=vec(vertices),face=" ".join(str(v) for face in faces for v in face))
    world=ET.SubElement(root,"worldbody")
    a=ET.SubElement(world,"body",name="crank",pos=vec([0,0,c["pivot_height_m"]]),quat=quat(q0[0]))
    ET.SubElement(a,"joint",name="A",armature=str(c["motor_armature_kg_m2"]))
    b=ET.SubElement(a,"body",name="coupler",pos=vec([c["lengths_m"][0],0,0]),quat=quat(q0[1]-q0[0]))
    ET.SubElement(b,"joint",name="B")
    d=ET.SubElement(world,"body",name="rocker",pos=vec([c["ground_length_m"],0,c["pivot_height_m"]]),quat=quat(q0[2]))
    ET.SubElement(d,"joint",name="D")
    for i,(name,node) in enumerate(zip(BODY_NAMES,[a,b,d])):
        prop=bodies[name];I=np.asarray(prop["inertia_com_local_kg_m2"])
        ET.SubElement(node,"inertial",pos=vec(prop["com_local_m"]),mass=str(prop["mass_kg"]),
                      fullinertia=vec([I[0,0],I[1,1],I[2,2],I[0,1],I[0,2],I[1,2]]))
        ET.SubElement(node,"geom",type="mesh",mesh="plate_%d"%i,pos=vec([0,c["visual_and_com_y_m"][i],0]),
                      rgba=vec([[.04,.3,.6,1],[.9,.3,.04,1],[.04,.4,.2,1]][i]))
    equality=ET.SubElement(root,"equality")
    ET.SubElement(equality,"connect",name="C_loop",body1="coupler",body2="rocker",
                  anchor=vec([c["lengths_m"][1],0,0]),solref=vec([s["equality_solref_s"],1]),
                  solimp=vec([s["equality_impedance"],s["equality_impedance"],.001,.5,2]))
    actuator=ET.SubElement(root,"actuator");limit=c["actuator"]["torque_limit_Nm"]
    ET.SubElement(actuator,"motor",name="crank_motor",joint="A",gear="1",ctrllimited="true",ctrlrange=vec([-limit,limit]))
    return ET.tostring(root,encoding="unicode"),q0


def build(entry):
    c=configuration(entry);parts=inventory(c);bodies=solid_properties(parts)
    audit=geometry_audit(c,parts,bodies)
    shape=[{key:p[key] for key in ["body","shape","axis","position_m","dimensions_m","rotation_matrix"] if key in p} for p in parts]
    digest=hashlib.sha256(json.dumps(shape,sort_keys=True).encode()).hexdigest()
    physical=hashlib.sha256(json.dumps({"config":{key:value for key,value in c.items()
                                                if key not in ["assembly_id","seed","hardware_recipe"]},
                                     "bodies":bodies},sort_keys=True).encode()).hexdigest()
    return {"schema_version":"1.0.0","assembly_id":entry["assembly_id"],"category":"S-A","config":c,
            "parts":parts,"bodies":bodies,"geometry_digest":digest,"physical_digest":physical,"assembly_audit":audit,
            "provenance":{"type":"procedural_reconstruction","sources":["experiments/fourbar/geometry.py","experiments/fourbar/visual_parts.py"],
                          "change":"Explicit per-part mass properties and aggregate moving-hardware inertia; new parametric dimensions"},
            "physics_scope":"Rigid solids, ideal bolted attachments and revolute joints; all moving metal solids included in body inertia"}
