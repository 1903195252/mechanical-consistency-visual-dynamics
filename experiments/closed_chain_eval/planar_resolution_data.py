"""Stream genuine high-resolution observations without opening physical truth.

Training access is restricted to the original training identities. Forecast
inputs contain only the three observed frames, public geometry and commands.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class PlanarResolutionData:
    def __init__(self, width):
        if width not in (256, 512):
            raise ValueError("The registered resolution extension contains 256 and 512 pixels")
        self.width = width
        self.root = ROOT / "output/planar_resolution" / str(width)
        self.manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        self.records = self.manifest["records"]

    def load(self, record, *, purpose, model):
        if purpose not in ("training", "forecast") or model not in ("KeyCLD", "CON"):
            raise ValueError("Declare training/forecast access and the model's native image convention")
        if purpose == "training" and record["split"] != "train":
            raise ValueError("Training cannot open held-out full sequences")
        if purpose == "forecast":
            with np.load(self.root / record["inputs"], allow_pickle=False) as z:
                images, times, geometry, action = (z[k] for k in ("x", "t", "geometry", "action"))
            assert len(images) == 3
        else:
            images = np.load(self.root / record["images"], mmap_mode="r", allow_pickle=False)
            with np.load(self.root / record["observations"], allow_pickle=False) as z:
                times, geometry, action = (z[k] for k in ("t", "geometry", "action"))
            assert len(images) == len(times) == 50
        assert images.shape[1:] == (self.width, self.width, 3) and images.dtype == np.uint8
        rgb = np.asarray(images, dtype=np.float32)
        if model == "KeyCLD":
            return dict(x=rgb/255, t=times, geometry=geometry, action=action)
        command = np.r_[action, geometry].astype(np.float32)
        command /= np.asarray([5, 12, 12, 1, 1] + [1]*9, dtype=np.float32)
        return dict(rendering_ts=rgb/127.5-1, tau=command, t=times)


def verify():
    original = json.loads((ROOT / "output/keycld_corpus/dataset_manifest.json").read_text())
    source = {r["case_id"]: r for r in original["records"]}
    summary = []
    for width in (256, 512):
        dataset = PlanarResolutionData(width)
        assert {r["case_id"] for r in dataset.records} == set(source)
        counts = {}
        for record in dataset.records:
            old = source[record["case_id"]]
            for field in ("assembly_id", "split", "category", "condition", "state_labels_sha256",
                          "physics_reference_sha256", "physical_parameters_sha256", "source_video_sha256"):
                assert record[field] == old[field], (width, record["case_id"], field)
            images = np.load(dataset.root / record["images"], mmap_mode="r", allow_pickle=False)
            count = 50 if record["split"] == "train" else 90
            assert images.shape == (count, width, width, 3)
            with np.load(dataset.root / record["inputs"], allow_pickle=False) as prefix:
                assert np.array_equal(prefix["x"], images[:3])
            counts[record["split"]] = counts.get(record["split"], 0) + 1
        first = next(r for r in dataset.records if r["split"] == "train")
        for model in ("KeyCLD", "CON"):
            dataset.load(first, purpose="training", model=model)
            dataset.load(first, purpose="forecast", model=model)
        heldout = next(r for r in dataset.records if r["split"] == "test")
        try:
            dataset.load(heldout, purpose="training", model="KeyCLD")
        except ValueError:
            pass
        else:
            raise AssertionError("Held-out training access was accepted")
        summary.append(dict(resolution=width, sequences=len(dataset.records), split_counts=counts,
                            shapes_and_prefixes_verified=True, same_physics_and_partitions=True,
                            heldout_training_access_rejected=True))
    result = dict(stage="data_verified", model_training_completed=False, results=summary)
    (ROOT / "output/planar_resolution/verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    verify()
