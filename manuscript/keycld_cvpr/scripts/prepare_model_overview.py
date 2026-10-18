"""Four visual-pipeline profiles from the canonical continuous score records."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
METRICS = ("geometry", "constraint_velocity", "constraint_acceleration",
           "dynamics", "energy", "prediction")


def collect():
    paths = [ROOT / "output/closed_chain_eval/continuous/scores.json",
             ROOT / "output/closed_chain_eval/continuous/case_scores.csv"]
    scores = json.loads(paths[0].read_text())
    assert scores["scoring_version"] == "continuous-si-1"
    with paths[1].open() as stream:
        cases = list(csv.DictReader(stream))
    profiles = []
    for model, arm, suite, native, corrected in (
            ("KeyCLD", "video_euler", "planar-servo", "raw", "conditioned"),
            ("CON", "video_con_iae", "planar-servo", "raw", "conditioned"),
            ("V-JEPA", "AC", "spatial-position", "native", "kinematic_intervention")):
        def selected(track):
            return [r for r in scores["summaries"] if r["model"] == model and
                    r["arm"] == arm and r["suite"] == suite and r["track"] == track]
        raw, reconstructed = selected(native), selected(corrected)
        assert len(raw) == len(reconstructed) == (1 if suite == "planar-servo" else 2)
        measurements = [r for r in cases if r["model"] == model and r["arm"] == arm
                        and r["suite"] == suite and r["track"] == native]
        profile = {"model": model, "arm": arm, "suite": suite,
            "native_track": native, "reconstructed_track": corrected,
            "registered_cases": sum(r["expected_cases"] for r in raw),
            "components_100": {k: float(np.mean([r["components_100"][k] for r in raw])) for k in METRICS},
            "reconstructed_components_100": {k: float(np.mean([r["components_100"][k] for r in reconstructed])) for k in METRICS},
            "measured_counts": {k: sum(bool(r[k + "_SI"]) for r in measurements) for k in METRICS},
            "native_score_100": float(np.mean([r["continuous_score_100"] for r in raw])),
            "reconstructed_score_100": float(np.mean([r["continuous_score_100"] for r in reconstructed]))}
        np.testing.assert_allclose(np.mean(list(profile["components_100"].values())), profile["native_score_100"])
        np.testing.assert_allclose(np.mean(list(profile["reconstructed_components_100"].values())), profile["reconstructed_score_100"])
        corrected_cases = [r for r in cases if r["model"] == model and r["arm"] == arm
                           and r["suite"] == suite and r["track"] == corrected]
        assert len(corrected_cases) == profile["registered_cases"]
        assert all(bool(r["dynamics_ratio"]) == bool(r["energy_ratio"]) for r in corrected_cases)
        available = [r for r in corrected_cases if r["dynamics_ratio"]]
        records = [{**{key: r[key] for key in ("case_id", "seed", "family", "condition")},
                    **{key + "_ratio": float(r[key + "_ratio"]) for key in ("dynamics", "energy")}}
                   for r in available]
        distributions = {}
        for metric in ("dynamics", "energy"):
            values = np.array([r[metric + "_ratio"] for r in records])
            assert np.all(np.isfinite(values)) and np.all(values > 0)
            distributions[metric] = {"quantiles": np.quantile(values, [0, .25, .5, .75, 1]).tolist(),
                                     "above_tolerance": int(np.sum(values > 1))}
        profile["reconstructed_mechanics"] = {"measurable_cases": len(records), "cases": records,
                                             "distributions": distributions}
        profiles.append(profile)
    # Both resolutions have equal visual weight; spatial profiles retain their
    # family-balanced definitions rather than being pooled with planar cases.
    from prepare_resolution_artifacts import collect as resolution_data
    resolution = resolution_data()
    paired = []
    for run in resolution['runs']:
        raw, reconstructed = run['native'], run['reconstructed']
        records = [dict(case_id=r['case_id'], seed=str(r['seed']), family=r['family'], condition=r['condition'],
                        dynamics_ratio=r['dynamics_ratio'], energy_ratio=r['energy_ratio'])
                   for r in reconstructed['rows'] if r['complete']]
        distributions = {k: dict(quantiles=np.quantile([r[k+'_ratio'] for r in records], [0,.25,.5,.75,1]).tolist(),
                                  above_tolerance=sum(r[k+'_ratio'] > 1 for r in records)) for k in ('dynamics','energy')}
        paired.append(dict(model=run['model'], label=f"{run['model']} {run['width']}", width=run['width'],
            arm=run['arm'], suite='planar-servo', native_track='raw', reconstructed_track='conditioned',
            registered_cases=324, components_100=raw['components'], reconstructed_components_100=reconstructed['components'],
            measured_counts={k:sum(r[k] is not None for r in raw['rows']) for k in METRICS},
            native_score_100=raw['score'], reconstructed_score_100=reconstructed['score'],
            reconstructed_mechanics=dict(measurable_cases=len(records),cases=records,distributions=distributions)))
    from prepare_spatial_expansion import collect as expanded_data, overview_profile
    profiles = paired + profiles[2:] + [overview_profile()]
    result = {"scoring_version": scores["scoring_version"], "profiles": profiles,
        "aggregation": "Composite scores balance planar seeds, families and conditions, or equally average Stewart and Delta. Upper distributions pool measurable reconstructed cases within each pipeline; boxes show unweighted case quartiles and full-range whiskers. Counts give measurable / registered cases.",
        "display": "Upper plots show post-reconstruction effort and energy error/tolerance ratios, all individual cases, quartiles and ranges on labeled log axes. Lower paired contribution stacks retain all six criterion scores divided by six. Directed links join hollow native markers to filled reconstructed markers, with both composite values labeled on the common 0--100 axis. Native effort/energy are unavailable and contribute zero. The 50-point line marks the maximum contribution of closure and its two differentiated constraints. Scores use each task's engineering targets; 50 is not a joint-qualification threshold.",
        "plot": {"score_axis": [0, 100],
                 "composite_divisor": 6, "reference_score": 50,
                 "upper_mechanics_layout": {"row_y": [78,97,116,135,157,178], "axis_y": 189,
                     "axes": {"dynamics": {"left": 214, "width": 80, "limits": [.7,200], "ticks": [1,10,100]},
                              "energy": {"left": 320, "width": 83, "limits": [.005,10000], "ticks": [.01,1,100,10000]}},
                     "point_diameter": 2.0, "point_opacity_percent": 32, "jitter_half_width": 4.1,
                     "jitter": "Stable SHA256 of seed/case_id; vertical displacement only, shared between metrics.",
                     "box_height": 6},
                 "contribution_layout": {"left": 112, "width": 272,
                                         "row_y": [292,321,350,379,408,437], "track_y_offsets": [-7,5]},
                 "reference_label": "Maximum contribution of the three kinematic criteria",
                 "criterion_colors": dict(zip(METRICS,
                     ("#0072B2", "#56B4E9", "#76B7B2", "#D55E00", "#009E73", "#596773")))},
        "sources": dict({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, **resolution['sources'], **expanded_data()['sources'])}
    return result


def export():
    evidence = collect()
    (PAPER / "sources/model_overview.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence
