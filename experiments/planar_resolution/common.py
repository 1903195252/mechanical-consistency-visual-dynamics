"""Runtime isolation, streaming data and atomic experiment artifacts."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output/planar_resolution"
RUNTIME = OUT / "runtime"
for name, value in {
    "TMPDIR": RUNTIME / "tmp", "CUDA_CACHE_PATH": RUNTIME / "cuda_cache",
    "JAX_COMPILATION_CACHE_DIR": RUNTIME / "jax_cache", "MPLCONFIGDIR": RUNTIME / "mpl",
}.items():
    Path(value).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(name, str(value))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("WANDB_MODE", "disabled")
sys.path[:0] = [str(RUNTIME / "packages"), str(RUNTIME / "con_native"), str(ROOT / "official"), str(ROOT / "output/keycld_corpus/runtime"),
               str(ROOT / "output/con_corpus/runtime")]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def dataset(width):
    from experiments.closed_chain_eval.planar_resolution_data import PlanarResolutionData
    return PlanarResolutionData(width)


def training_records(data):
    return [r for r in data.records if r["split"] == "train"]


def representative_records(data):
    """One training sequence per family/condition; never read held-out labels."""
    selected = {}
    for record in training_records(data):
        selected.setdefault((record["category"], record["condition"]), record)
    return list(selected.values())


def load_keycld(data, record):
    item = data.load(record, purpose="training", model="KeyCLD")
    return {k: v for k, v in item.items() if k in ("x", "t", "geometry", "action")}
