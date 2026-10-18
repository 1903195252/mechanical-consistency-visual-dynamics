"""Bounded full-objective timing, or explicitly requested complete training.

Benchmark artifacts are separate from scientific runs and cannot be resumed as
trained models. Training always uses all 813 sequences and exactly 40 epochs.
"""
import argparse
import importlib.metadata
import json
import os
import pickle
import signal
import time
import numpy as np
from .common import ROOT, OUT, dataset, training_records, representative_records, load_keycld, sha, write
from .keycld import ResolutionKeyCLD, make_step, jax, jnp

STOP = False


def stop(*unused):
    global STOP
    STOP = True


def save(path, value):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as handle:
        pickle.dump(jax.device_get(value), handle, protocol=4)
    temporary.replace(path)


def run(args):
    devices = jax.devices()
    if not args.allow_cpu and all(d.platform != "gpu" for d in devices):
        raise RuntimeError("GPU benchmark requested but JAX has no GPU; CPU timing is not a GPU estimate")
    data = dataset(args.width)
    records = training_records(data)
    assert len(records) == 813
    folder = OUT / ("preflight" if args.mode == "benchmark" else "runs") / f"keycld_{args.width}_seed{args.seed}"
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder / ("benchmark.json" if args.mode == "benchmark" else "training_report.json")
    if report_path.exists():
        raise FileExistsError(f"Finished artifact already exists: {report_path}")
    started = time.monotonic()
    def status(stage, **extra):
        write(folder / "status.json", dict(stage=stage, pid=os.getpid(), mode=args.mode,
              width=args.width, seed=args.seed, elapsed_s=time.monotonic()-started, **extra))
    status("initializing")
    first = load_keycld(data, records[0])
    tx, step = make_step(args.width, args.frame_chunk)
    params = ResolutionKeyCLD(jnp.asarray(first["geometry"]), args.width).init(jax.random.PRNGKey(args.seed))
    state = tx.init(params)
    generator = np.random.RandomState(args.seed)
    count = sum(x.size for x in jax.tree.leaves(params))
    identity = dict(protocol_sha256=sha(ROOT / "experiments/planar_resolution/protocol.json"),
        frame_chunk=args.frame_chunk,
        dataset_manifest_sha256=sha(data.root / "dataset_manifest.json"),
        source_sha256={name: sha(ROOT / "experiments/planar_resolution" / name)
                       for name in ("keycld.py", "run_keycld.py", "common.py")},
        upstream_losses_sha256=sha(ROOT / "output/keycld_corpus/runtime/keycld/losses.py"),
        upstream_models_sha256=sha(ROOT / "output/keycld_corpus/runtime/keycld/models.py"))
    packages = {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "flax", "optax", "numpy")}
    samples, history, cursor, epoch, updates, order = [], [], 0, 0, 0, None
    checkpoint = folder / "resume.p"
    if args.mode == "train" and checkpoint.exists():
        with checkpoint.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["identity"] != identity:
            raise ValueError("Checkpoint source or registered data changed")
        params, state = saved["params"], saved["opt_state"]
        generator.set_state(saved["rng"])
        epoch, cursor, updates, order, history = (saved[k] for k in ("epoch", "cursor", "updates", "order", "history"))
    def checkpoint_now():
        save(checkpoint, dict(params=params, opt_state=state, rng=generator.get_state(),
             epoch=epoch, cursor=cursor, updates=updates, order=order, history=history, identity=identity))
    if args.mode == "train" and not checkpoint.exists():
        save(folder / "initial_params.p", params)
    targets = representative_records(data) if args.mode == "benchmark" else records
    max_updates = args.warmup + args.steps if args.mode == "benchmark" else 40 * len(records)
    while updates < max_updates:
        if STOP:
            if args.mode == "train":
                checkpoint_now()
            status("stopped", updates=updates)
            return
        if args.mode == "train":
            if order is None:
                order = generator.permutation(len(records))
            record = records[int(order[cursor])]
        else:
            record = targets[updates % len(targets)]
        tick = time.monotonic()
        item = load_keycld(data, record)
        load_s = time.monotonic() - tick
        augmentation = generator.randint(0, 8, 50).astype(np.int32)
        status("compiling" if updates == 0 else "updating", updates=updates, case_id=record["case_id"])
        tick = time.monotonic()
        params, state, loss, aux, finite = step(params, state, item, augmentation)
        jax.block_until_ready((params, state, loss, aux, finite))
        step_s = time.monotonic() - tick
        if not bool(finite) or not np.isfinite(float(loss)):
            status("failed", updates=updates, reason="Nonfinite full loss or gradient")
            raise FloatingPointError("Nonfinite full objective or gradients; no sequences skipped")
        row = dict(update=updates+1, case_id=record["case_id"], load_s=load_s, update_s=step_s,
                   loss=float(loss), components={k:float(v) for k,v in aux.items()})
        updates += 1
        if args.mode == "benchmark":
            samples.append(row)
            write(folder / "timing_partial.json", samples)
            print(json.dumps(row), flush=True)
        else:
            cursor += 1
            if cursor == len(records):
                epoch += 1
                history.append(dict(epoch=epoch, updates=updates, last_loss=float(loss), elapsed_s=time.monotonic()-started))
                cursor, order = 0, None
                checkpoint_now()
            elif updates % 100 == 0:
                checkpoint_now()
    report = dict(mode=args.mode, width=args.width, seed=args.seed, parameters=count, identity=identity,
        devices=[str(d) for d in devices], packages=packages, full_objective=True,
        frames_per_sequence=50, reconstruction_scale=(64/args.width)**2,
        precision="FP32", framewise_rematerialization=True, frame_chunk=args.frame_chunk,
        elapsed_s=time.monotonic()-started)
    if args.mode == "benchmark":
        timings = [r["load_s"] + r["update_s"] for r in samples[args.warmup:]]
        report.update(scientific_training_result=False, warmup_updates=args.warmup, timed_updates=args.steps,
            first_compile_and_update_s=samples[0]["update_s"], samples=samples,
            steady_mean_s=float(np.mean(timings)), steady_median_s=float(np.median(timings)),
            steady_p10_p90_s=np.quantile(timings, [.1,.9]).tolist(),
            training_hours_per_seed=float(np.mean(timings)*32520/3600),
            three_seed_training_hours=float(np.mean(timings)*32520*3/3600),
            estimate_excludes=["reader training and calibration", "model inference and SI scoring",
                               "checkpoint I/O", "sustained laptop thermal/power changes"])
        memory = devices[0].memory_stats()
        report["jax_device_memory_stats"] = memory
    else:
        save(folder / "params.p", params)
        report.update(scientific_training_result=True, epochs=40, optimizer_updates=updates,
            history=history, checkpoint_sha256=sha(folder / "params.p"), checkpoint_selection="final epoch 40")
    write(report_path, report)
    status("benchmark_complete" if args.mode == "benchmark" else "training_complete", updates=updates)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("benchmark", "train"), required=True)
    parser.add_argument("--width", type=int, choices=(256,512), required=True)
    parser.add_argument("--seed", type=int, choices=(42,43,44), default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=9)
    parser.add_argument("--frame-chunk", type=int, default=1,
                        help="Independent frames evaluated together; all 50 frames still contribute")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    if args.steps < 3 or args.warmup < 1:
        parser.error("Use at least 1 warmup and 3 measured complete updates")
    if not 1 <= args.frame_chunk <= 50:
        parser.error("Frame chunk must be between 1 and 50")
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    run(args)
