"""Verify the released SI evaluator and its reference calibration bindings."""
import json
from .common import ROOT, sha

DESTINATION = ROOT/'configs/evaluator_binding.json'


def verify():
    binding = json.loads(DESTINATION.read_text())
    for group in ('source_hashes','calibration_hashes'):
        for path, expected in binding[group].items():
            if sha(ROOT/path) != expected:
                raise ValueError('Evaluator or calibration changed: '+path)
    return binding


def prepare():
    return verify()


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify-only',action='store_true')
    p.parse_args()
    verify()
    print('Released SI evaluator and reference calibration verified.')
