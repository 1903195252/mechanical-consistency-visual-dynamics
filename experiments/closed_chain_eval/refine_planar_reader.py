"""Reduce convolutional sampling bias with development-selected shift averaging.

Integer image shifts are undone in physical coordinates before averaging. No
geometry or trajectory constraints are imposed. Only original validation
identities select the ensemble; calibration/verification identities are held out.
"""
import json
from itertools import product

import numpy as np
import torch
from torch.nn import functional as F

from .planar_visual_reader import CoordinateReader, DATA, DIRECTORY
from .visual_calibration import OUT, ROOT, sha, write, load_protocol


DESTINATION = OUT / "optimization/planar_reader"
SHIFTS = {"single": [(0, 0)], "shift4": list(product((0, 1), repeat=2)),
          "shift16": list(product((-2, -1, 0, 1), repeat=2))}


def shifted_points(net, images, settings):
    predictions = {}
    unique = sorted(set(shift for name in settings for shift in SHIFTS[name]))
    for dx, dy in unique:
        pieces = []
        for source in np.array_split(images, max(1, int(np.ceil(len(images) / 45)))):
            x = torch.from_numpy(source).cuda().permute(0, 3, 1, 2).float() / 255
            x = F.pad(x, (2, 2, 2, 2), value=128/255)[:, :, 2-dy:66-dy, 2-dx:66-dx]
            with torch.no_grad():
                points, _ = net(x)
            pieces.append(points.cpu().numpy() - np.array([dx, -dy]) / 31.5)
        predictions[dx, dy] = np.concatenate(pieces)
    return {name: np.mean([predictions[s] for s in SHIFTS[name]], axis=0) * .65 + [.2, .35] for name in settings}


def run():
    torch.set_num_threads(4)
    checkpoint = DIRECTORY / "final.pt"
    state = torch.load(checkpoint, map_location="cpu")
    net = CoordinateReader().cuda().eval()
    net.load_state_dict(state["model"])
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    development = [r for r in manifest["records"] if r["split"] == "validation"]
    spec = dict(checkpoint_sha256=sha(checkpoint), shifts=SHIFTS, source_sha256=sha(__file__),
                selection="mean development point Euclidean RMSE; family and condition counts balanced",
                selection_ids=[r["case_id"] for r in development], geometry_projection=False)
    write(DESTINATION / "specification.json", spec)
    results = []
    for i, row in enumerate(development):
        cache = DESTINATION / "development" / (row["case_id"] + ".npz")
        if cache.exists():
            with np.load(cache) as data:
                predicted = {key: data[key] for key in SHIFTS}
        else:
            with np.load(DATA / row["observations"]) as data:
                predicted = shifted_points(net, data["x"], SHIFTS)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, **predicted, reader_sha256=np.array(sha(checkpoint)))
        with np.load(DATA / row["state_labels"]) as data:
            truth = data["x"].astype(float) * .65 + [.2, .35]
        result = dict(case_id=row["case_id"], assembly_id=row["assembly_id"], category=row["category"],
            point_rmse_mm={key: 1000 * float(np.sqrt(np.mean(np.sum((value[3:] - truth[3:]) ** 2, axis=-1))))
                           for key, value in predicted.items()})
        results.append(result)
        if (i + 1) % 9 == 0:
            print(json.dumps(dict(stage="development", completed=i + 1, total=len(development))), flush=True)
    objective = {key: float(np.mean([r["point_rmse_mm"][key] for r in results])) for key in SHIFTS}
    selected = min(objective, key=objective.get)
    selection = dict(selected=selected, objective=objective, rows=results,
                     specification_sha256=sha(DESTINATION / "specification.json"))
    write(DESTINATION / "selection.json", selection)
    print(json.dumps(dict(selected=selected, development_point_rmse_mm=objective)), flush=True)
    for i, row in enumerate(load_protocol()["planar_records"]):
        cache = DESTINATION / "held_out" / (row["case_id"] + ".npz")
        if cache.exists():
            continue
        with np.load(DATA / row["observations"]) as data:
            images, times = data["x"], data["t"]
        predicted = shifted_points(net, images, [selected])[selected]
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, points=predicted, time=times.astype(float),
                            reader_sha256=np.array(sha(checkpoint)), selection_sha256=np.array(sha(DESTINATION / "selection.json")))
        if (i + 1) % 9 == 0:
            print(json.dumps(dict(stage="held_out", completed=i + 1, total=54)), flush=True)


if __name__ == "__main__":
    run()
