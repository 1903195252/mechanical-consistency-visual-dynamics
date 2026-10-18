"""Validate paired data and training loss equivalence without fitting a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import numpy as np
from .common import ROOT, OUT, dataset, sha, write


def verify_data():
    original = json.loads((ROOT / "output/keycld_corpus/dataset_manifest.json").read_text())
    reference = {r["case_id"]: r for r in original["records"]}
    paired = json.loads((ROOT / "experiments/planar_resolution/protocol.json").read_text())["dataset"]["paired_fields"]
    reports = []
    for width in (256,512):
        data = dataset(width)
        assert {r["case_id"] for r in data.records} == set(reference)
        split_ids = {}
        total_bytes = 0
        started = time.monotonic()
        for i, record in enumerate(data.records):
            for field in paired:
                assert record[field] == reference[record["case_id"]][field], (width, record["case_id"], field)
            split_ids.setdefault(record["split"], set()).add(record["assembly_id"])
            camera = (ROOT / record["source_video"]).with_suffix(".inputs.json")
            assert sha(camera) == record["camera_sha256"]
            for field in ("images", "observations", "inputs"):
                path = data.root / record[field]
                assert sha(path) == record[field + "_sha256"], (width, record["case_id"], field)
                total_bytes += path.stat().st_size
            if (i+1) % 200 == 0:
                print(json.dumps(dict(width=width, verified=i+1, total=len(data.records))), flush=True)
        for a, ids in split_ids.items():
            assert all(not ids.intersection(other) for b, other in split_ids.items() if a != b)
        reports.append(dict(width=width, sequences=len(data.records),
            split_sequences=dict(Counter(r["split"] for r in data.records)),
            split_assemblies={key:len(value) for key,value in split_ids.items()},
            sha256_all_observation_files_verified=True, physical_references_identical=True,
            bytes_verified=total_bytes, seconds=time.monotonic()-started))
    write(OUT / "preflight/data_integrity.json", dict(passed=True, resolutions=reports))


def verify_objective(frame_chunk=1, report_path=None):
    from .keycld import ResolutionKeyCLD, objective, jax, jnp
    manifest = json.loads((ROOT / "output/keycld_corpus/dataset_manifest.json").read_text())
    record = next(r for r in manifest["records"] if r["split"] == "train")
    with np.load(ROOT / "output/keycld_corpus" / record["observations"]) as z:
        item = {k: z[k] for k in ("x", "t", "geometry", "action")}
    item["x"] = item["x"].astype(np.float32)/255
    params = ResolutionKeyCLD(item["geometry"], 64).init(jax.random.PRNGKey(42))
    augmentation = np.random.RandomState(42).randint(0,8,50).astype(np.int32)
    result = []
    for stream in (False, True):
        fn = jax.jit(jax.value_and_grad(lambda p: objective(p, item, augmentation, 64, stream, frame_chunk), has_aux=True))
        (loss, aux), grad = fn(params)
        jax.block_until_ready((loss, aux, grad))
        result.append((float(loss), jax.device_get(aux), jax.device_get(grad)))
    a, b = result
    relative_loss = abs(a[0]-b[0])/max(1.,abs(a[0]))
    numerator = sum(np.sum((x.astype(float)-y)**2) for x,y in zip(jax.tree.leaves(a[2]),jax.tree.leaves(b[2])))
    denominator = sum(np.sum(x.astype(float)**2) for x in jax.tree.leaves(a[2]))
    relative_grad = float(np.sqrt(numerator/max(denominator,1e-30)))
    report = dict(native_loss=a[0], streamed_loss=b[0], relative_loss=relative_loss,
        relative_gradient_l2=relative_grad, tolerance=1e-3, frames=50, frame_chunk=frame_chunk,
        real_training_case=record["case_id"], all_three_loss_terms=True,
        devices=[str(d) for d in jax.devices()], passed=relative_loss < 1e-3 and relative_grad < 1e-3)
    write(report_path or OUT / "preflight/objective_equivalence.json", report)
    if not report["passed"]:
        raise AssertionError(report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "objective"), required=True)
    parser.add_argument("--frame-chunk", type=int, default=1)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    verify_data() if args.stage == "data" else verify_objective(args.frame_chunk, args.report)
