"""Independent spatial SI audit with an explicitly separate kinematic intervention."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import mujoco
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from experiments.mechanism_corpus import delta, stewart
from experiments.mechanism_corpus.spatial_tree import SpatialTree
from .prepare_spatial import ROOT, DESTINATION, write_json


class Audit:
    def __init__(self, row):
        self.row = row
        directory = ROOT / row["evaluator_directory"]
        self.parameters = json.loads((directory / "physical_parameters.json").read_text())
        self.c = self.parameters["config"]
        self.model = mujoco.MjModel.from_xml_path(str(directory / "model.xml"))
        self.control = json.loads((directory / "controller_inputs.json").read_text())
        self.tree = SpatialTree.from_mujoco(self.model).use_body_calibration(self.parameters["bodies"])
        self.names = {body["name"]: body["id"] for body in self.tree.bodies}
        self.kind = row["category"]
        if self.kind == "H-A":
            self.qi, self.vi = np.arange(9, 25, 3), np.arange(8, 24, 3)
            self.limit, self.unit = self.c["actuator"]["force_limit_N"], "N"
        else:
            drives = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "drive_%d" % i)
                      for i in range(3)]
            self.qi, self.vi = self.model.jnt_qposadr[drives], self.model.jnt_dofadr[drives]
            self.passive = np.array([self.model.jnt_dofadr[j] for j in range(self.model.njnt)
                                     if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and j not in drives])
            self.limit, self.unit = self.c["actuator"]["torque_limit_Nm"], "N m"
        self.B = np.zeros((self.model.nv, len(self.vi)))
        self.B[self.vi, np.arange(len(self.vi))] = 1
        with np.load(directory / "commands.npz") as data:
            if self.kind == "H-A":
                p, v = "commanded_prismatic_displacement_m", "commanded_prismatic_velocity_m_s"
            else:
                p, v = "shoulder_angle_command_rad", "shoulder_velocity_command_rad_s"
            self.command_times = data["time"]
            self.command_p, self.command_v = data[p], data[v]

    def closure(self, q):
        if self.kind == "H-A":
            return stewart.closure_jacobian(q, self.c)
        states = self.tree.forward(q, np.zeros(self.model.nv))
        return delta.closure(states, self.tree, self.c)

    def mechanics(self, q, v):
        c = self.c
        if self.kind == "H-A":
            M, bias, T, V = stewart.mass_bias_energy(q, v, c, self.parameters["bodies"])
            return M, bias, T + V, stewart.friction(v, c), stewart.external(c)
        M, bias, T, V = self.tree.dynamics(q, v)
        friction = np.zeros(self.model.nv)
        friction[self.vi] = -c["drive_viscosity_Nm_s_rad"] * v[self.vi] - c["drive_coulomb_Nm"] * np.tanh(v[self.vi] / .03)
        friction[self.passive] = -c["passive_viscosity_Nm_s_rad"] * v[self.passive] - c["passive_coulomb_Nm"] * np.tanh(v[self.passive] / .03)
        return M, bias, T + V, friction, np.zeros(self.model.nv)

    def effort(self, t, q, v):
        p = np.array([np.interp(t, self.command_times, column) for column in self.command_p.T])
        velocity = np.array([np.interp(t, self.command_times, column) for column in self.command_v.T])
        act = self.control["controller_calibration"]
        if self.kind == "H-A":
            feed = np.asarray(self.control["feedforward_force_N"])
            kp, kd = act["position_gain_N_m"], act["velocity_gain_N_s_m"]
        else:
            feed = np.asarray(self.control["feedforward_torque_Nm"])
            kp, kd = act["position_gain_Nm_rad"], act["velocity_gain_Nm_s_rad"]
        return np.clip(feed + kp * (p - q[self.qi]) + kd * (velocity - v[self.vi]), -self.limit, self.limit)

    def points(self, q):
        states = self.tree.forward(q, np.zeros(self.model.nv))
        result = [q[:3]]
        if self.kind == "H-A":
            for i in range(6):
                result.append(states[self.names["rod_%d" % i]]["position"])
        else:
            for i, j, elbow, _ in delta.anchor_pairs(self.c):
                result.append(self.tree.point(states[self.names["upper_%d" % i]], elbow)[0])
                result.append(self.tree.point(states[self.names["rod_%d_%d" % (i, j)]],
                                              [0, 0, self.c["forearm_length_m"]])[0])
        return np.asarray(result)

    def project(self, q):
        """Hold the readout platform position, reconstruct the supplied mechanism.

        This is a post-hoc kinematic intervention, never the native readout.
        It uses supplied geometry, not true future states or learned parameters.
        """
        c = self.c
        result = q.copy()
        if self.kind == "H-A":
            R = stewart.quaternion_matrix(q[3:7])
            for i, (base, anchor, length, yaw, pitch) in enumerate(zip(
                c["base_anchors_m"], c["platform_anchors_m"], c["initial_leg_lengths_m"],
                c["leg_yaw_rad"], c["leg_pitch_rad"])):
                vector = q[:3] + R @ anchor - base
                size = np.linalg.norm(vector)
                n = stewart.rz(yaw).T @ (vector / size)
                qx = np.arctan2(-n[1], n[2])
                beta = np.arctan2(n[0], np.hypot(n[1], n[2]))
                result[7 + 3 * i:10 + 3 * i] = [qx, beta - pitch, size - length]
        else:
            result[3:7] = [1, 0, 0, 0]
            result[self.qi] = delta.inverse_kinematics(result[:3], c)
            states = self.tree.forward(result, np.zeros(self.model.nv))
            for i, j, elbow, anchor in delta.anchor_pairs(c):
                upper = states[self.names["upper_%d" % i]]
                origin = self.tree.point(upper, elbow)[0]
                n = upper["rotation"].T @ ((result[:3] + anchor - origin) / c["forearm_length_m"])
                qx = np.arctan2(-n[1], n[2])
                beta = np.arctan2(n[0], np.hypot(n[1], n[2]))
                for name, value in [("universal_x_%d_%d" % (i, j), qx),
                                    ("universal_y_%d_%d" % (i, j), beta - c["forearm_reference_pitch_rad"])]:
                    joint = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                    result[self.model.jnt_qposadr[joint]] = value
        residual, _ = self.closure(result)
        if np.max(np.linalg.norm(residual.reshape(-1, 3), axis=1)) > 1e-8:
            raise ValueError("Kinematic intervention did not reach closure")
        return result

    def derivatives(self, q, dt):
        velocity = np.empty((len(q), self.model.nv))
        for i in range(len(q)):
            left, right = max(i - 1, 0), min(i + 1, len(q) - 1)
            forward, backward = np.zeros(self.model.nv), np.zeros(self.model.nv)
            if right != i:
                mujoco.mj_differentiatePos(self.model, forward, dt, q[i], q[right])
            if left != i:
                mujoco.mj_differentiatePos(self.model, backward, -dt, q[i], q[left])
            velocity[i] = ((forward + backward) / 2 if left != i and right != i
                           else forward if left == i else backward)
        acceleration = savgol_filter(velocity, 7, 3, deriv=1, delta=dt, axis=0)
        velocity = savgol_filter(velocity, 7, 3, axis=0)
        return velocity, acceleration

    def measure(self, time, q, truth, exact_derivatives=None):
        if not np.isfinite(q).all() or np.any(np.linalg.norm(q[:, 3:7], axis=1) < 1e-8):
            return {"finite": False, "physical_pass": False}
        velocity, acceleration = exact_derivatives if exact_derivatives is not None else self.derivatives(q, time[1] - time[0])
        closure, vclosure, aclosure, residuals, energies, powers, condition = [], [], [], [], [], [], []
        marker_errors, orientation = [], []
        for k, (t, pose, v, a) in enumerate(zip(time, q, velocity, acceleration)):
            gap, J = self.closure(pose)
            _, singular, vh = np.linalg.svd(J, full_matrices=True)
            rank = np.linalg.matrix_rank(J)
            N = vh[rank:].T
            mapping = N.T @ self.B
            M, bias, energy, friction, external = self.mechanics(pose, v)
            u = self.effort(t, pose, v)
            # Off-manifold readouts can change constraint rank. Preserve their
            # geometric/prediction errors; an actuator residual is then undefined.
            equivalent = np.full(len(self.vi), np.nan)
            projection_condition = np.nan
            if mapping.shape == (len(self.vi), len(self.vi)):
                projection_condition = np.linalg.cond(mapping)
                if np.isfinite(projection_condition) and projection_condition < 1e12:
                    equivalent = np.linalg.solve(mapping, N.T @ (M @ a + bias - friction - external))
            epsilon = 1e-5
            plus, minus = pose.copy(), pose.copy()
            mujoco.mj_integratePos(self.model, plus, v, epsilon)
            mujoco.mj_integratePos(self.model, minus, v, -epsilon)
            Jdot = (self.closure(plus)[1] - self.closure(minus)[1]) / (2 * epsilon)
            closure.append(np.max(np.linalg.norm(gap.reshape(-1, 3), axis=1)))
            vclosure.append(np.max(np.linalg.norm((J @ v).reshape(-1, 3), axis=1)))
            aclosure.append(np.max(np.linalg.norm((J @ a + Jdot @ v).reshape(-1, 3), axis=1)))
            residuals.append(equivalent - u)
            energies.append(energy)
            powers.append((self.B @ u + friction + external) @ v)
            condition.append(projection_condition)
            marker_errors.append(self.points(pose) - self.points(truth[k]))
            predicted = Rotation.from_quat(pose[[4, 5, 6, 3]])
            reference = Rotation.from_quat(truth[k, [4, 5, 6, 3]])
            orientation.append((reference.inv() * predicted).magnitude())
        # Exclude the observed prefix and the three-frame differentiation edge.
        interior = slice(3, len(time) - 3)
        future = slice(3, len(time))
        torque = np.asarray(residuals)[interior]
        energy = np.asarray(energies)[interior]
        work = cumulative_trapezoid(np.asarray(powers)[interior], time[interior], initial=0)
        energy_defect = energy - energy[0] - work
        g, v, a = [float(np.max(np.asarray(x)[interior])) for x in [closure, vclosure, aclosure]]
        projection_defined = bool(np.isfinite(torque).all())
        effort_rmse = float(np.sqrt(np.mean(torque**2))) if projection_defined else None
        e = float(np.max(np.abs(energy_defect)))
        # Engineering scales: 5% rated actuator effort; 1% commanded actuator work scale.
        energy_tolerance = .01 * self.limit * float(np.sum(np.ptp(self.command_p, axis=0)))
        point_rmse = float(np.sqrt(np.mean(np.sum(np.asarray(marker_errors)[future]**2, axis=-1))))
        tests = dict(geometry=g <= .001, velocity=v <= .01, acceleration=a <= .1,
                     effort=projection_defined and effort_rmse <= .05 * self.limit, energy=e <= energy_tolerance,
                     prediction=point_rmse <= .01)
        admissible = tests["geometry"] and tests["velocity"] and tests["acceleration"] and projection_defined
        return dict(finite=True, geometry_max_mm=1000 * g, velocity_constraint_max_m_s=v,
                    acceleration_constraint_max_m_s2=a, point_rmse_mm=1000 * point_rmse,
                    platform_position_rmse_mm=float(np.sqrt(np.mean(np.sum((q[future, :3] - truth[future, :3])**2, axis=1))) * 1000),
                    platform_rotation_rmse_deg=float(np.sqrt(np.mean(np.asarray(orientation)[future]**2)) * 180 / np.pi),
                    raw_equivalent_effort_rmse=effort_rmse, effort_unit=self.unit,
                    raw_work_energy_max_J=e, mechanical_coverage=admissible,
                    effort_rmse=effort_rmse if admissible else None,
                    work_energy_max_J=e if admissible else None,
                    actuator_projection_defined=projection_defined,
                    actuator_projection_condition_max=float(np.nanmax(condition)) if np.isfinite(condition).any() else None,
                    effort_tolerance=.05 * self.limit, energy_tolerance_J=energy_tolerance,
                    component_pass=tests, physical_pass=all(tests.values()))


def evaluate_case(task):
    row, mode, *locations = task
    prediction_root = Path(locations[0]) if locations else DESTINATION
    audit = Audit(row)
    directory = ROOT / row["evaluator_directory"]
    with np.load(directory / "physics.npz") as data:
        indices = np.arange(120) * 8
        truth = data["qpos"][indices]
        v, a = data["qvel"][indices], data["qacc"][indices]
        times = data["time"][indices]
    answer = {key: row[key] for key in ["video_id", "assembly_id", "category", "condition", "split"]}
    if mode == "calibrate":
        answer["reference_exact"] = audit.measure(times, truth, truth, (v, a))
        answer["reference_30Hz"] = audit.measure(times, truth, truth)
        projected = np.stack([audit.project(q) for q in truth])
        answer["intervention_reference_max_coordinate_change_mm"] = float(max(
            np.linalg.norm(audit.points(p) - audit.points(q), axis=1).max()
            for p, q in zip(projected, truth)) * 1000)
        return answer
    prediction = prediction_root / "predictions" / row["category"] / (row["video_id"] + ".npz")
    if not prediction.exists():
        answer["status"] = "missing_prediction"
        return answer
    with np.load(prediction) as data:
        q = data["qpos"].astype(float)
        if not np.allclose(data["time"], times, atol=1e-9):
            raise ValueError("Prediction timestamps differ from the registered reference")
    answer["native"] = audit.measure(times, q, truth)
    # Hold the last observed predicted pose: a matched readout-persistence control.
    persistent = q.copy()
    persistent[3:] = q[2]
    answer["readout_persistence"] = audit.measure(times, persistent, truth)
    try:
        projected = np.stack([audit.project(pose) for pose in q])
        answer["kinematic_intervention"] = audit.measure(times, projected, truth)
        answer["coordinate_correction_rmse_mm"] = float(np.sqrt(np.mean([
            np.sum((audit.points(p) - audit.points(raw))**2, axis=1)
            for p, raw in zip(projected[3:], q[3:])])) * 1000)
        np.savez_compressed(prediction.with_name(prediction.stem + "_projected.npz"), time=times, qpos=projected)
    except (ValueError, np.linalg.LinAlgError) as error:
        answer["kinematic_intervention"] = {"finite": False, "physical_pass": False, "error": str(error)}
    answer["status"] = "evaluated"
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["calibrate", "score"])
    parser.add_argument("--category", choices=["H-A", "H-B"], required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--prediction-root", type=Path, default=DESTINATION,
                        help="Separate forecast and score destination; frozen manifest remains shared")
    args = parser.parse_args()
    rows = json.loads((DESTINATION / "dataset_manifest.json").read_text())["videos"]
    split = "validation" if args.mode == "calibrate" else "test"
    rows = [r for r in rows if r["category"] == args.category and r["split"] == split and r["view"] == "front"]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for item in pool.map(evaluate_case, [(r, args.mode, args.prediction_root) for r in rows]):
            results.append(item)
            write_json(args.prediction_root / (args.category + "_" + args.mode + ".json"), results)
            print(item["video_id"], len(results), "/", len(rows), flush=True)


if __name__ == "__main__":
    main()
