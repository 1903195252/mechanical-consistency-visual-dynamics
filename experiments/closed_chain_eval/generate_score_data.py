"""New paired RGB inputs and dense MuJoCo truth for scored KeyCLD evaluation."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
from scipy.integrate import solve_ivp

from .evaluate import write_json
from .import_keycld import ROOT
from .planar import physics


def generate(task):
    condition, seed, config, root = task
    root = Path(root)
    import mujoco
    from extended_dataset import xml_model
    from mechanics import TREE_TO_ABSOLUTE
    rng = np.random.RandomState(seed)
    action = rng.uniform(-2., 2., 3)
    action[1:] = 0.
    if rng.rand() < .2:
        action[:] = 0.
    c = copy.deepcopy(config)
    c["initial_crank_angle_rad"] = float(rng.uniform(-np.pi, np.pi))
    c["initial_crank_velocity_rad_s"] = float(rng.uniform(-.1, .1))
    c["drive"] = {"type": "constant_open_loop_torque", "torque_Nm": float(action[0]), "torque_limit_Nm": 2.}
    xml, q0 = xml_model(c)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    S = TREE_TO_ABSOLUTE
    data.ctrl[0] = action[0]
    renderer = mujoco.Renderer(model, 64, 64)
    options = mujoco.MjvOption()
    options.sitegroup[:] = 0
    times = np.arange(1, 801, dtype=float) / 1600
    def rhs(t, state):
        return [state[1], physics.acceleration(state[0], state[1], float(action[0]), c)]
    solution = solve_ivp(rhs, [0., .5], [c["initial_crank_angle_rad"], c["initial_crank_velocity_rad_s"]],
                         method="DOP853", t_eval=times, rtol=1e-10, atol=1e-12, max_step=.002)
    if not solution.success:
        raise RuntimeError(solution.message)
    images, positions, velocities, accelerations = [], [], [], []
    max_closure = max_force = max_mass = max_bias = 0.
    dense_mass = np.zeros((3, 3))
    for k, (theta, omega) in enumerate(solution.y.T):
        q = physics.inverse(theta, c)
        gamma, second = physics.derivatives(q, c)
        alpha = physics.acceleration(theta, omega, float(action[0]), c)
        v, a = gamma * omega, gamma * alpha + second * omega**2
        positions.append(q.copy())
        velocities.append(v.copy())
        accelerations.append(a.copy())
        if (k + 1) % 16 == 0:
            data.qpos[:] = np.linalg.solve(S, q - q0)
            data.qvel[:] = np.linalg.solve(S, v)
            data.qfrc_applied[:] = S.T @ (physics.external(q, c) + physics.friction(v, c))
            mujoco.mj_forward(model, data)
            M, C, G = physics.mass_bias(q, v, c)
            gamma, _ = physics.derivatives(q, c)
            value = gamma @ (M @ a + C + G - [action[0], 0., 0.]
                             - physics.external(q, c) - physics.friction(v, c))
            max_force = max(max_force, abs(float(value)))
            max_closure = max(max_closure, float(np.linalg.norm(physics.closure(q, c))))
            mujoco.mj_fullM(model, dense_mass, data.qM)
            max_mass = max(max_mass, float(np.max(np.abs(dense_mass - S.T @ M @ S))))
            max_bias = max(max_bias, float(np.max(np.abs(data.qfrc_bias - S.T @ (C + G)))))
            renderer.update_scene(data, camera="main", scene_option=options)
            images.append(renderer.render().copy())
    renderer.close()
    assert max_closure < 1e-12 and max_force < 1e-8 and max_mass < 1e-10 and max_bias < 1e-8, (condition, seed, max_closure, max_force, max_mass, max_bias)
    observation = root / "observations" / condition / ("run_%04d.npz" % seed)
    truth = root / "truth" / condition / ("run_%04d.npz" % seed)
    observation.parent.mkdir(parents=True, exist_ok=True)
    truth.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(observation, x=np.asarray(images), t=np.arange(1, 51, dtype=np.float32) / 100,
                        action=action.astype(np.float32))
    np.savez_compressed(truth, time=np.arange(1, 801, dtype=float) / 1600,
                        q=np.asarray(positions), qdot=np.asarray(velocities), qddot=np.asarray(accelerations),
                        exact_action=action)
    write_json(truth.with_suffix(".json"), c)
    return {"condition": condition, "seed": seed, "torque_Nm": float(action[0]),
            "initial_angle_rad": c["initial_crank_angle_rad"],
            "initial_speed_rad_s": c["initial_crank_velocity_rad_s"],
            "closure_max_m": max_closure, "force_residual_max_Nm": max_force,
            "independent_mujoco_mass_max_abs": max_mass, "independent_mujoco_bias_max_abs": max_bias,
            "reference_solver_nfev": solution.nfev,
            "observation_sha256": hashlib.sha256(observation.read_bytes()).hexdigest(),
            "truth_sha256": hashlib.sha256(truth.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repair-generation", action="store_true")
    args = parser.parse_args()
    root = ROOT / "output/keycld_fourbar/scored_evaluation"
    root.mkdir(parents=True, exist_ok=True)
    if (root / "dataset_manifest.json").exists():
        failed = root / "calibration.json"
        if not args.repair_generation or not failed.exists() or json.loads(failed.read_text())["measurement_floor_acceptable"]:
            raise RuntimeError("Completed scored data already exists; preserve it")
        previous = json.loads(failed.read_text())
        previous["investigation"] = "MuJoCo implicitfast finite-step trajectories passed instantaneous force checks but accumulated work-energy and acceleration-constraint errors at high speed. Replaced only this new score dataset with ideal DOP853 trajectories before any new model inference; limits unchanged."
        write_json(root / "integrator_audit.json", previous)
        failed.unlink()
    source = Path(__file__).with_name("keycld_score_protocol.json")
    protocol = json.loads(source.read_text())
    write_json(root / "protocol.json", protocol)
    configs = json.loads((ROOT / "output/keycld_fourbar/extended_dynamics/condition_configs.json").read_text())
    tasks = []
    for split in ("calibration", "test"):
        low, high = protocol[split + "_seeds"]
        tasks += [(name, seed, configs[name], str(root)) for name in protocol["conditions"] for seed in range(low, high + 1)]
    start = time.monotonic()
    rows = []
    with ProcessPoolExecutor(max_workers=6, mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(generate, task) for task in tasks]):
            rows.append(future.result())
            if len(rows) % 16 == 0:
                status = {"stage": "generating", "completed": len(rows), "total": len(tasks), "seconds": time.monotonic() - start}
                write_json(root / "status.json", status)
                print(json.dumps(status), flush=True)
    rows.sort(key=lambda row: (row["condition"], row["seed"]))
    for seed in sorted({row["seed"] for row in rows}):
        paired = [row for row in rows if row["seed"] == seed]
        assert len(paired) == 8
        for key in ("torque_Nm", "initial_angle_rad", "initial_speed_rad_s"):
            assert all(row[key] == paired[0][key] for row in paired)
    write_json(root / "dataset_manifest.json", {"rows": rows, "count": len(rows), "dense_hz": 1600,
               "protocol_sha256": hashlib.sha256((root / "protocol.json").read_bytes()).hexdigest(),
               "new_test_seeds": True, "elapsed_seconds": time.monotonic() - start})
    write_json(root / "status.json", {"stage": "data_complete", "completed": len(rows)})


if __name__ == "__main__":
    main()
