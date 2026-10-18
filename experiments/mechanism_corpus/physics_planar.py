"""Torque-driven three-coordinate planar closed chains with independent dynamics."""
import copy
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid

from .catalog import write_json
from . import fourbar


def source_digest(mechanism):
    files=[Path(__file__),Path(mechanism.__file__),Path(fourbar.__file__),Path(__file__).with_name("solids.py")]
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def command_speed(t, target, ramp):
    return target*.5*(1-np.cos(np.pi*min(max(t/ramp,0.),1.)))


def simulate(assembly, condition, directory, mechanism):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    c=copy.deepcopy(assembly["config"])
    target={"low_speed":1.,"rated":3.,"limit_or_load":5.}[condition]
    if condition=="limit_or_load":c["payload_mass_kg"]*=1.75
    parts=mechanism.inventory(c);bodies=mechanism.solid_properties(parts)
    xml,q0=mechanism.mjcf(c,bodies)
    (directory/"model.xml").write_text(xml)
    write_json(directory/"physical_parameters.json",{"config":c,"parts":parts,"bodies":bodies,
               "operating_condition":condition,"command_speed_rad_s":target})
    model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
    S=mechanism.TREE_TO_ABSOLUTE
    actuator=c["actuator"]
    def control(m,d):
        q=q0+S@d.qpos;v=S@d.qvel
        requested=command_speed(d.time,target,actuator["ramp_s"])
        d.ctrl[0]=np.clip(actuator["speed_gain_Nm_s_rad"]*(requested-v[0]),
                          -actuator["torque_limit_Nm"],actuator["torque_limit_Nm"])
        d.qfrc_applied[:]=S.T@(mechanism.friction(v,c)+mechanism.external(q,c))
    mujoco.set_mjcb_control(control)
    s=c["simulation"];steps=round(s["duration_s"]/s["timestep_s"])
    record_stride=round(1/(s["record_hz"]*s["timestep_s"]))
    video_stride=round(1/(s["fps"]*s["timestep_s"]))
    records=[];video=[];mass_error=bias_error=0.;dense=np.zeros((3,3));ranks=set()
    started=time.monotonic()
    try:
        for step in range(steps+1):
            if step%record_stride==0:
                mujoco.mj_forward(model,data)
                q=q0+S@data.qpos;v=S@data.qvel;a=S@data.qacc
                M,C,G,T,V=mechanism.mass_bias_energy(q,v,c,bodies)
                J=mechanism.closure_jacobian(q,c)
                gamma=np.r_[1.,np.linalg.solve(J[:,1:],-J[:,0])]
                Qf=mechanism.friction(v,c);Qext=mechanism.external(q,c)
                tau=float(data.ctrl[0])
                residual=M@a+C+G-np.array([tau,0.,0.])-Qf-Qext
                tangent_residual=float(gamma@residual)
                # Reaction power is measured from the simulator's explicit
                # constraint force. It is not fitted to reduce a model residual.
                reaction_power=float(data.qfrc_constraint@data.qvel)
                mujoco.mj_fullM(model,dense,data.qM)
                mass_error=max(mass_error,float(abs(dense-S.T@M@S).max()))
                bias_error=max(bias_error,float(abs(data.qfrc_bias-S.T@(C+G)).max()))
                ranks.add(int(np.linalg.matrix_rank(J)))
                records.append((data.time,q.copy(),v.copy(),a.copy(),data.qpos.copy(),data.qvel.copy(),
                                data.qacc.copy(),tau,command_speed(data.time,target,actuator["ramp_s"]),
                                T,V,mechanism.closure(q,c),J@v,J@a+mechanism.closure_curvature(q,v,c),
                                tangent_residual,tau*v[0],float(Qf@v),float(Qext@v),reaction_power,
                                data.qfrc_constraint.copy()))
            if step%video_stride==0 and step<steps:
                mujoco.mj_forward(model,data)
                video.append((data.time,data.xpos.copy(),data.xquat.copy()))
            if step<steps:mujoco.mj_step(model,data)
    finally:
        mujoco.set_mjcb_control(None)
    names=["time","q","qdot","qddot","qpos","qvel","qacc","actual_motor_torque_Nm","speed_command_rad_s",
           "kinetic_J","potential_J","closure_m","closure_velocity_m_s","closure_acceleration_m_s2",
           "motor_equivalent_residual_Nm","motor_power_W","friction_power_W","external_power_W","constraint_power_W",
           "constraint_generalized_force"]
    arrays={name:np.asarray([row[i] for row in records]) for i,name in enumerate(names)}
    np.savez_compressed(directory/"physics.npz",**arrays)
    np.savez_compressed(directory/"commands.npz",time=arrays["time"],speed_command_rad_s=arrays["speed_command_rad_s"])
    body_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or "ground" for i in range(model.nbody)]
    np.savez_compressed(directory/"render_poses.npz",time=np.asarray([row[0] for row in video]),
                        body_position_m=np.asarray([row[1] for row in video]),
                        body_quaternion_wxyz=np.asarray([row[2] for row in video]),body_names=np.asarray(body_names))
    E=arrays["kinetic_J"]+arrays["potential_J"]
    power=sum(arrays[key] for key in ["motor_power_W","friction_power_W","external_power_W","constraint_power_W"])
    balance=E-E[0]-cumulative_trapezoid(power,arrays["time"],initial=0)
    ignore=arrays["time"]>=actuator["ramp_s"]+0.25
    metrics={
        "independent_dof":model.nv-min(ranks),"constraint_ranks":sorted(ranks),
        "maximum_loop_closure_m":float(np.linalg.norm(arrays["closure_m"],axis=1).max()),
        "maximum_motor_equivalent_residual_Nm":float(abs(arrays["motor_equivalent_residual_Nm"]).max()),
        "independent_mass_matrix_max_abs":mass_error,"independent_bias_force_max_abs":bias_error,
        "maximum_work_energy_error_J":float(abs(balance).max()),
        "constraint_work_J":float(np.trapz(arrays["constraint_power_W"],arrays["time"])),
        "maximum_friction_power_W":float(arrays["friction_power_W"].max()),
        "achieved_mean_crank_speed_rad_s":float(arrays["qdot"][ignore,0].mean()),
        "crank_revolutions":float((arrays["q"][-1,0]-arrays["q"][0,0])/(2*np.pi)),
        "video_frames":len(video),"record_samples":len(records),"finite":all(np.isfinite(v).all() for v in arrays.values()),
    }
    thresholds={"maximum_loop_closure_m":1e-4,"maximum_motor_equivalent_residual_Nm":1e-7,
                "independent_mass_matrix_max_abs":1e-10,"independent_bias_force_max_abs":1e-8,
                "maximum_work_energy_error_J":.01}
    checks={key:metrics[key]<limit for key,limit in thresholds.items()}
    checks.update(correct_degrees_of_freedom=metrics["independent_dof"]==1,
                  finite=metrics["finite"],complete_video_support=len(video)==120,
                  attained_operating_speed=.65*target<metrics["achieved_mean_crank_speed_rad_s"]<1.35*target,
                  dissipative_friction=metrics["maximum_friction_power_W"]<=1e-9)
    report={"condition":condition,"passed":all(checks.values()),"checks":checks,"thresholds":thresholds,
            "metrics":metrics,"seconds":time.monotonic()-started,"mujoco_version":mujoco.__version__,
            "assembly_physical_digest":assembly["physical_digest"],"generator_source_digest":source_digest(mechanism),
            "scope":"Raw MuJoCo states, finite compliant loop closure, explicitly recorded reaction work; no pose projection.",
            "coordinate_units":c.get("coordinate_units",["rad","rad","rad"]),
            "command_interface":"Velocity setpoint is public; realized feedback torque remains evaluator-side by default."}
    write_json(directory/"physics_validation.json",report)
    if not report["passed"]:
        raise RuntimeError("Physics validation failed: "+str(directory)+" "+str({k:v for k,v in checks.items() if not v}))
    return report
