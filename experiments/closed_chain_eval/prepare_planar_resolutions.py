"""Recover matched 256/512 observations directly from the original 640px RGB.

Physical trajectories, assembly partitions, FOV, pixel-centre convention and
the registered frame support are identical to the existing 64px experiment.
Images are memory-mappable uint8 arrays; training need not load the corpus RAM.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

from .visual_calibration import ROOT, sha, write


SOURCE = ROOT / "output/keycld_corpus"
DIRECTORY = ROOT / "output/planar_resolution"
WIDTHS = (256, 512)
CENTER = np.array([.2, .35])
SCALE = .65


def pixel_map(camera, width):
    R = np.asarray(camera["R_world_to_camera"])
    t = np.asarray(camera["t_world_to_camera_m"])
    K = np.asarray(camera["K_affine_pixels"])
    world_to_source = K @ np.array([[R[0, 0], R[0, 2], t[0]], [R[1, 0], R[1, 2], t[1]], [0, 0, 1.]])
    pixel_to_world = np.array([[2*SCALE/(width-1), 0, CENTER[0]-SCALE],
                              [0, -2*SCALE/(width-1), CENTER[1]+SCALE], [0, 0, 1.]])
    return world_to_source @ pixel_to_world


def warp(frame, mapping, width):
    bgr = cv2.warpPerspective(frame, mapping, (width, width), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def link_or_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def prepare_case(row):
    cv2.setNumThreads(1)
    destinations = [DIRECTORY/str(n)/"records"/(row["case_id"]+".json") for n in WIDTHS]
    if all(p.exists() for p in destinations):
        saved = [json.loads(p.read_text()) for p in destinations]
        for r in saved:
            if not (DIRECTORY/str(r["resolution"])/r["images"]).exists():
                raise FileNotFoundError(r["images"])
        return saved
    video = ROOT/row["source_video"]
    if sha(video) != row["source_video_sha256"]:
        raise ValueError("Source RGB changed: "+row["case_id"])
    camera = json.loads(video.with_suffix(".inputs.json").read_text())["observed_camera_frames"][0]
    maps = {n: pixel_map(camera, n) for n in WIDTHS}
    # All new grids sample a finer original rendering; no 64px image is read.
    if min(np.linalg.svd(m[:2, :2], compute_uv=False).min() for m in maps.values()) < 1:
        raise ValueError("Source rendering is too coarse for a requested grid")
    count = 50 if row["split"] == "train" else 90
    arrays, paths = {}, {}
    for n in WIDTHS:
        path = DIRECTORY/str(n)/"observations"/row["split"]/(row["case_id"]+".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        paths[n] = path
        arrays[n] = np.lib.format.open_memmap(str(path)+".partial", mode="w+", dtype=np.uint8,
                                              shape=(count, n, n, 3))
    cap = cv2.VideoCapture(str(video))
    source_width, source_height = int(cap.get(3)), int(cap.get(4))
    frames = 0
    for index in range(30+count):
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Incomplete source video: "+str(video))
        if index < 30:
            continue
        for n in WIDTHS:
            arrays[n][frames] = warp(frame, maps[n], n)
        frames += 1
    cap.release()
    records = []
    for n in WIDTHS:
        arrays[n].flush()
        del arrays[n]
        Path(str(paths[n])+".partial").replace(paths[n])
        root = DIRECTORY/str(n)
        with np.load(SOURCE/row["observations"]) as previous:
            metadata = {k: previous[k] for k in previous.files if k != "x"}
        observed = np.load(paths[n], mmap_mode="r")
        inputs = root/row["inputs"]
        inputs.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(inputs, x=observed[:3], t=np.arange(90, dtype=np.float32)/30,
                            geometry=metadata["geometry"], action=metadata["action"])
        meta = paths[n].with_suffix(".npz")
        np.savez_compressed(meta, **metadata)
        link_or_copy(SOURCE/row["state_labels"], root/row["state_labels"])
        r = dict(row, resolution=n, observations=str(meta.relative_to(root)), observations_sha256=sha(meta),
                 images=str(paths[n].relative_to(root)), images_sha256=sha(paths[n]),
                 inputs_sha256=sha(inputs), source_resolution=[source_width, source_height],
                 native_rendering_downsampled=True, reused_64px_images=False,
                 camera_sha256=sha(video.with_suffix(".inputs.json")))
        write(root/"records"/(row["case_id"]+".json"), r)
        records.append(r)
    return records


def run(workers):
    original = json.loads((SOURCE/"dataset_manifest.json").read_text())
    spec = dict(resolutions=[64, *WIDTHS], new_widths=list(WIDTHS), source_resolution=[640, 640],
                source_manifest_sha256=sha(SOURCE/"dataset_manifest.json"), source_sha256=sha(__file__),
                camera_center_m=CENTER.tolist(), half_fov_m=SCALE, convention="pixel_centers",
                sampling="bilinear calibrated warp from original RGB; identical to 64px preparation",
                splits=original["assemblies_by_split"], training_frames=50, evaluation_frames=90,
                observed_frames=3, fps=30, source_start_frame=30, physics_and_labels_unchanged=True,
                upsampled_64px_images=False)
    path = DIRECTORY/"protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Resolution protocol changed")
    write(path, spec)
    # A changed grid must reproduce the old one at N=64 before new data is made.
    first = original["records"][0]
    cam = json.loads((ROOT/first["source_video"]).with_suffix(".inputs.json").read_text())["observed_camera_frames"][0]
    cap = cv2.VideoCapture(str(ROOT/first["source_video"]))
    for _ in range(31):
        ok, frame = cap.read()
    cap.release()
    with np.load(SOURCE/first["observations"]) as d:
        if not ok or not np.array_equal(warp(frame, pixel_map(cam, 64), 64), d["x"][0]):
            raise ValueError("Resolution preparation does not reproduce the original 64px mapping")
    records = {n: [] for n in WIDTHS}
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_case, row) for row in original["records"]]
        for count, future in enumerate(as_completed(futures), 1):
            for row in future.result():
                records[row["resolution"]].append(row)
            if count%30 == 0 or count == len(futures):
                status = dict(stage="preparing", completed=count, total=len(futures), resolutions=list(WIDTHS),
                              elapsed_s=round(time.monotonic()-started, 1))
                write(DIRECTORY/"status.json", status)
                print(json.dumps(status), flush=True)
    for n in WIDTHS:
        ordered = sorted(records[n], key=lambda r: r["case_id"])
        write(DIRECTORY/str(n)/"dataset_manifest.json", dict(records=ordered, counts=original["counts"],
            assemblies_by_split=original["assemblies_by_split"], resolution=n,
            protocol_sha256=sha(path), images_format="memory-mapped uint8 NPY; metadata in NPZ",
            physical_reference_sha256_unchanged=True, model_training_completed=False))
    write(DIRECTORY/"status.json", dict(stage="data_complete", sequences_per_resolution=len(original["records"]),
        completed=len(original["records"]), total=len(original["records"]), resolutions=list(WIDTHS),
        model_training_completed=False, elapsed_s=round(time.monotonic()-started, 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    run(parser.parse_args().workers)
