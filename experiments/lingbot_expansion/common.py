"""Frozen scope and paths for the spatial learning-curve experiment."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/lingbot_expansion'
CORPUS = OUT / 'corpus'
OLD_CORPUS = ROOT / 'output/mechanism_corpus'
OLD_MANIFEST = ROOT / 'output/model_expansion/spatial/dataset_manifest.json'
OLD_RUN = ROOT / 'output/model_expansion/lingbot_spatial'
FAMILIES = ('H-A', 'H-B')
CONDITIONS = ('low_speed', 'rated', 'limit_or_load')
SEEDS = (42, 43, 44)
STEPS = (0, 50)
TEST_ASSEMBLIES_PER_FAMILY = 50
NEW_ASSEMBLIES_PER_FAMILY = TEST_ASSEMBLIES_PER_FAMILY - 5
NEW_ASSEMBLIES = len(FAMILIES) * NEW_ASSEMBLIES_PER_FAMILY
TEST_CASES = len(FAMILIES) * TEST_ASSEMBLIES_PER_FAMILY * len(CONDITIONS)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.%d.partial' % os.getpid())
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def frozen(path, data):
    if path.exists():
        if json.loads(path.read_text()) != data:
            raise ValueError('Refusing to change frozen definition: ' + str(path))
    else:
        write(path, data)


def generation_seed(seed, video_id):
    # Independent of manifest order, GPU allocation and training checkpoint.
    value = 'lingbot60:%d:%s' % (seed, video_id)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'big') & 0x7fffffff


def records(split=None, require_ready=True):
    path = OUT / ('dataset_manifest.json' if require_ready else 'planned_dataset.json')
    data = json.loads(path.read_text())
    if require_ready and not data.get('ready'):
        raise ValueError('Dataset production and acceptance must finish before training')
    return [r for r in data['videos'] if split is None or r['split'] == split]


def training_records():
    """Verified training inputs can be cached while held-out videos are rendered."""
    data=json.loads((OUT/'training_manifest.json').read_text())
    if not data.get('ready') or data.get('scope')!='training-only':raise ValueError('Training inputs are not verified')
    rows=data['videos']
    if len(rows)!=270 or any(r['split']!='train' for r in rows):raise ValueError('Training partition changed')
    return rows


def assert_split_isolation(rows):
    assignments = {}
    cases = set()
    for row in rows:
        identity = row['assembly_id']
        prior = assignments.setdefault(identity, row['split'])
        if prior != row['split']:
            raise ValueError('Assembly crosses data partitions: ' + identity)
        if row['video_id'] in cases:
            raise ValueError('Duplicate video identity: ' + row['video_id'])
        cases.add(row['video_id'])
    for family in FAMILIES:
        tests = [r for r in rows if r['category'] == family and r['split'] == 'test']
        identities = {r['assembly_id'] for r in tests}
        if len(identities) != TEST_ASSEMBLIES_PER_FAMILY or len(tests) != TEST_ASSEMBLIES_PER_FAMILY * len(CONDITIONS):
            raise ValueError('Require 50 test assemblies and 150 trajectories per family')
        for identity in identities:
            subset = [r for r in tests if r['assembly_id'] == identity]
            if {r['condition'] for r in subset} != set(CONDITIONS) or any(r['view'] != 'front' for r in subset):
                raise ValueError('Require exactly three front-view conditions: ' + identity)
