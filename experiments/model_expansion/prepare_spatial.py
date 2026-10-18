"""Create an assembly-level snapshot of verified spatial videos."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3


ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "output/mechanism_corpus"
DESTINATION = ROOT / "output/model_expansion/spatial"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(destination=DESTINATION):
    connection = sqlite3.connect("file:%s?mode=ro" % (CORPUS / "catalog.sqlite"))
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        "SELECT a.assembly_id, a.source_assembly_id, a.category, a.split, "
        "v.video_id, v.condition, v.view, v.path "
        "FROM assemblies a JOIN videos v USING(assembly_id) "
        "WHERE a.category IN ('H-A','H-B') AND a.status='verified' "
        "AND v.status='verified' ORDER BY a.category,a.assembly_id,v.video_index"
    ).fetchall()
    connection.close()
    videos, identities, cases = [], {}, set()
    for row in rows:
        video = CORPUS / row["path"]
        sidecar = video.with_suffix(".inputs.json")
        public = json.loads(sidecar.read_text())
        commands = (sidecar.parent / public["commanded_actions"]).resolve()
        condition = video.parent.parent / "conditions" / row["condition"]
        for path in (video, commands, condition / "physics_validation.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
        validation = json.loads((condition / "physics_validation.json").read_text())
        if not validation["passed"]:
            raise ValueError("Failed physics reference: %s" % condition)
        record = dict(row)
        record.update(
            video=str(video.relative_to(ROOT)),
            public_inputs=str(sidecar.relative_to(ROOT)),
            public_inputs_sha256=sha256(sidecar),
            commands=str(commands.relative_to(ROOT)),
            commands_sha256=sha256(commands),
            camera_projection=public["camera_intrinsics"]["projection"],
            observed_prefix_frames=public["observed_prefix_frames"],
            evaluator_directory=str(condition.relative_to(ROOT)),
            depth_available=False,
        )
        videos.append(record)
        identities[row["assembly_id"]] = (row["category"], row["split"])
        cases.add((row["assembly_id"], row["condition"]))
    summary = {}
    for category in ("H-A", "H-B"):
        counts = Counter(split for cat, split in identities.values() if cat == category)
        summary[category] = {
            "assemblies": sum(counts.values()),
            "assembly_splits": dict(sorted(counts.items())),
            "physical_trajectories": sum(a.startswith(category) for a, _ in cases),
            "videos": sum(row["category"] == category for row in videos),
        }
    manifest = {
        "schema_version": 1,
        "purpose": "Available-data snapshot for spatial baseline development",
        "independent_sampling_unit": "assembly; views share a physical trajectory",
        "split_policy": "Preserve existing assembly-level train/validation/test splits",
        "reference_access": "Evaluator paths are not model input; training labels require train split",
        "summary": summary,
        "videos": videos,
    }
    write_json(Path(destination) / "dataset_manifest.json", manifest)
    print(json.dumps(summary, ensure_ascii=False))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DESTINATION)
    prepare(parser.parse_args().output)
