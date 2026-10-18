"""The current, shared RGB measurement sources for spatial audit artifacts."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CURRENT = ROOT / 'output/closed_chain_eval/spatial_rgb_measurement'


def score_path(mode, category):
    if mode not in ('reference', 'zero_shot', 'adapted') or category not in ('H-A', 'H-B'):
        raise ValueError('Unregistered spatial measurement source')
    return CURRENT / mode / (category + '_score.json')


def prediction_path(mode, row):
    protocol = json.loads((CURRENT / 'protocol.json').read_text())
    return ROOT / protocol['prediction_roots'][row['category']][mode] / (row['video_id'] + '.npz')


def reference_metrics(category, video_id):
    rows = json.loads(score_path('reference', category).read_text())
    return next(r['kinematic_intervention'] for r in rows if r['video_id'] == video_id)


def finite_cases(rows):
    return [r for r in rows if r.get('native', {}).get('finite', False)]
