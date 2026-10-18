"""Rebuild registered RGB arrays from compact source videos, checking exact hashes.

This changes transport only. Camera calibration, frame support, warping and
the pre-existing dataset manifests remain fixed. A mismatched array is never
installed as a registered observation.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .common import ROOT, OUT, dataset, representative_records, sha, training_records, write
from experiments.closed_chain_eval.prepare_planar_resolutions import pixel_map, warp


def rebuild_case(work):
    records, destination = work
    cv2.setNumThreads(1)
    row = records[0]
    video = ROOT / row["source_video"]
    camera_path = video.with_suffix(".inputs.json")
    if sha(video) != row["source_video_sha256"] or sha(camera_path) != row["camera_sha256"]:
        raise ValueError("Source video or calibration changed: " + row["case_id"])
    camera = json.loads(camera_path.read_text())["observed_camera_frames"][0]
    pending = []
    for record in records:
        width = record["resolution"]
        root = OUT / str(width)
        for field in ("observations", "inputs"):
            if sha(root / record[field]) != record[field + "_sha256"]:
                raise ValueError("Registered metadata changed: " + record["case_id"])
        target = Path(destination) / str(width) / record["images"]
        if target.exists() and sha(target) == record["images_sha256"]:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        count = 50 if record["split"] == "train" else 90
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8,
                                        shape=(count, width, width, 3))
        pending.append((record, target, temporary, array, pixel_map(camera, width)))
    if pending:
        cap = cv2.VideoCapture(str(video))
        count = len(pending[0][3])
        try:
            for index in range(30 + count):
                ok, frame = cap.read()
                if not ok:
                    raise ValueError("Incomplete source video: " + str(video))
                if index >= 30:
                    for record, _, _, array, mapping in pending:
                        array[index - 30] = warp(frame, mapping, record["resolution"])
        finally:
            cap.release()
        for record, target, temporary, array, _ in pending:
            array.flush()
            if sha(temporary) != record["images_sha256"]:
                raise ValueError("Rebuilt RGB differs from registered data: " + str(temporary))
            temporary.replace(target)
    return dict(case_id=row["case_id"], widths=[r["resolution"] for r in records], exact_hashes=True)


def run(args):
    data = dataset(256)
    selected = data.records
    if args.bootstrap:
        train = training_records(data)
        indices = np.random.RandomState(42).permutation(len(train))[:80]
        identifiers = {train[int(i)]["case_id"] for i in indices}
        identifiers.update(r["case_id"] for r in representative_records(data))
        identifiers.add(train[0]["case_id"])
        selected = [r for r in selected if r["case_id"] in identifiers]
    if args.case_id:
        selected = [r for r in selected if r["case_id"] in args.case_id]
        if {r["case_id"] for r in selected} != set(args.case_id):
            raise ValueError("Requested case is outside the selected registered set")
    manifests = {width: {r["case_id"]: r for r in dataset(width).records} for width in args.widths}
    work = [([manifests[w][r["case_id"]] for w in args.widths], str(args.destination)) for r in selected]
    started = time.monotonic()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(rebuild_case, item) for item in work]):
            results.append(future.result())
            if len(results) % 30 == 0 or len(results) == len(work):
                print(json.dumps(dict(verified=len(results), total=len(work), seconds=time.monotonic()-started)), flush=True)
    write(args.report, dict(passed=True, transport="original source video; hash-identical RGB reconstruction",
          cases=len(results), widths=args.widths, bootstrap_only=args.bootstrap,
          destination=str(args.destination), opencv=cv2.__version__, seconds=time.monotonic()-started,
          dataset_sha256={str(w): sha(OUT/str(w)/"dataset_manifest.json") for w in args.widths},
          results=sorted(results, key=lambda x:x["case_id"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--widths", type=int, choices=(256, 512), nargs="+", default=[256, 512])
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--destination", type=Path, default=OUT)
    parser.add_argument("--report", type=Path, default=OUT/"validation/rebuilt_images.json")
    run(parser.parse_args())
