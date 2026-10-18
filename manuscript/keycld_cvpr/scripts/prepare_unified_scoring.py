"""Export the one current score and measured instrument ablations to LaTeX.

Raw prediction files and historical pass-rate tables remain measurement sources.
This exporter changes only the composite columns and the current observer results.
"""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
SCORES = ROOT / "output/closed_chain_eval/continuous"
CAL = ROOT / "output/closed_chain_eval/visual_calibration"
sys.path.insert(0, str(ROOT))
SOURCES = {}
PLANAR_ARMS = ["video_euler", "state_euler", "state_rk4", "video_con_iae", "state_con_iae",
               "state_euler_con_iae", "state_rk4_con_iae"]


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def fmt(value):
    return "N/A" if value is None else f"{value:.2f}"


def mean(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def texrow(*cells):
    return " & ".join(map(str, cells)) + r" \\"


def continuous_estimate(rows):
    """Cluster resampling: seeds and assembly IDs, preserving all conditions."""
    seeds = sorted({r["seed"] for r in rows})
    families = sorted({r["family"] for r in rows})
    arrays = []
    for family in families:
        selected = [r for r in rows if r["family"] == family]
        assemblies = sorted({r["case_id"].split("__")[0] for r in selected})
        values = np.empty((len(seeds), len(assemblies)))
        for i, seed in enumerate(seeds):
            for j, assembly in enumerate(assemblies):
                cases = [r for r in selected if r["seed"] == seed and
                         r["case_id"].split("__")[0] == assembly]
                assert len(cases) == len({r["condition"] for r in cases}) == 3
                values[i, j] = np.mean([float(r["continuous_score_100"]) for r in cases])
        arrays.append(values)
    rng = np.random.default_rng(20261016)
    draws = []
    for _ in range(2000):
        si = rng.integers(len(seeds), size=len(seeds))
        draws.append(np.mean([a[np.ix_(si, rng.integers(a.shape[1], size=a.shape[1]))].mean()
                              for a in arrays]))
    return dict(mean=float(np.mean([a.mean() for a in arrays])),
                ci95=np.percentile(draws, [2.5, 97.5]).tolist(),
                by_seed={s: float(np.mean([a[i].mean() for a in arrays]))
                         for i, s in enumerate(seeds)})


def replace_column(name, column, values):
    path = PAPER / "tables" / name
    lines = path.read_text().splitlines()
    indices = [i for i, line in enumerate(lines) if " & " in line and
               not line.startswith("Image persistence")]
    assert len(indices) == len(values), (name, indices, values)
    for i, value in zip(indices, values):
        cells = lines[i].rstrip("\\ ").split(" & ")
        cells[column] = value
        lines[i] = texrow(*cells)
    path.write_text("\n".join(lines) + "\n")


def planar_estimates():
    """Canonical continuous composites for every manuscript table exporter."""
    summary = read(SCORES / "scores.json")
    assert summary["scoring_version"] == "continuous-si-1"
    path = SCORES / "case_scores.csv"
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    estimates = {}
    for arm in PLANAR_ARMS:
        estimates[arm] = continuous_estimate([r for r in rows if r["suite"] == "planar-servo"
                                             and r["arm"] == arm and r["track"] == "raw"])
        expected = next(r["continuous_score_100"] for r in summary["summaries"]
                        if r["suite"] == "planar-servo" and r["arm"] == arm and r["track"] == "raw")
        np.testing.assert_allclose(estimates[arm]["mean"], expected)
    return estimates








def main():
    summary = read(SCORES / "scores.json")
    assert summary["scoring_version"] == "continuous-si-1"
    arms = PLANAR_ARMS
    estimates = planar_estimates()
    # The resolution exporter owns the main table, including shared state rows.
    for file, selected in (("corpus_score_rows.tex", arms[:3]), ("con_score_rows.tex", arms[3:5])):
        replace_column(file, -1, [fmt(estimates[a]["mean"]) + " [" +
            ", ".join(map(fmt, estimates[a]["ci95"])) + "]" for a in selected])
    replace_column("corpus_seed_rows.tex", 5,
                   [fmt(estimates[a]["by_seed"][str(s)]) for a in arms[:3] for s in (42, 43, 44)])
    replace_column("con_seed_rows.tex", 4,
                   [fmt(estimates[a]["by_seed"][str(s)]) for a in arms[3:5] for s in (42, 43, 44)])

    from prepare_spatial_expansion import export_tables
    spatial = export_tables()
    SOURCES.update(spatial['sources'])

    stress = read(CAL / "optimization/robust_planar_reader/stress.json")
    quality_rows = []
    for quality, label in [("clean", "Clean"), ("mild_blur", r"Blur $\sigma=0.5$"),
                            ("blur", r"Blur $\sigma=1$"), ("blur_low_contrast", r"Blur + $0.6\times$ contrast")]:
        values = [mean([r["point_rmse_m"] * 1000 for r in stress["rows"] if
                       r["quality"] == quality and r["reader"] == reader and
                       r["calibration_role"] == "verification"]) for reader in ("baseline", "robust")]
        assert all(v is not None for v in values), quality
        quality_rows.append(texrow(label, *map(fmt, values)))
    (PAPER / "tables/reader_quality_rows.tex").write_text("\n".join(quality_rows) + "\n")
    paired = read(CAL / "paired_robust_planar_rgb/summary.json")
    phase = read(CAL / "paired_constrained_visual_phase/summary.json")
    phase_by_model = {r["model"]: r for r in phase["summary"]}
    paired_rows = []
    for r in paired["summary"]:
        a = phase_by_model[r["model"]]
        label = r["model"].replace("reference", "Reference RGB").replace("keycld_seed", "KeyCLD, ").replace("con_seed", "CON, ")
        paired_rows.append(texrow(label, f'{r["finite"]}/{r["total"]}', fmt(r["point_rmse_mean_mm"]),
                                  fmt(a["force_mean_Nm"]), fmt(a["energy_mean_J"])))
    (PAPER / "tables/reader_paired_rows.tex").write_text("\n".join(paired_rows) + "\n")
    calibration = read(CAL / "summary.json")
    spatial_rows = []
    for reader, label in [("shared_spatial_256", "Absolute-coordinate head"),
                           ("geometry_spatial_256", "CAD-coordinate head"),
                           ("reset_spatial_256", "+ known-pose alignment")]:
        rr = [next(r for r in calibration["summaries"] if r["reader"] == reader and r["category"] == c)
              for c in ("H-A", "H-B")]
        spatial_rows.append(texrow(label, *[fmt(r["verification_point_rmse_mean_mm"]) for r in rr],
                                  *[f'{r["intervention_qualified"]}/6' for r in rr]))
    (PAPER / "tables/observer_ablation_rows.tex").write_text("\n".join(spatial_rows) + "\n")
    instrument = ROOT / "output/lingbot_expansion/delta_instrument/no_coordinate_smoothing"
    verified = read(instrument / "optical/verification_summary.json")["cases"]
    ablation = read(instrument / "component_ablation.json")["cases"]
    delta_rows = []
    for label, cases, branch in [
            ("Exact sampled states", verified, "numerical"),
            ("Relative feature head", ablation, "relative_head_without_optical"),
            ("+ image-plane tracking", verified, "updated"),
            ("Held CAD reset", ablation, "held_CAD_reset_negative_control")]:
        delta_rows.append(texrow(label, *[fmt(mean([r[branch][key] for r in cases])) for key in
            ("point_rmse_mm", "effort_rmse", "work_energy_max_J")]))
    (PAPER / "tables/delta_measurement_rows.tex").write_text("\n".join(delta_rows) + "\n")
    replay = {k: g['paired'] for k, g in spatial['groups'].items()}
    shutil.copyfile(CAL / "calibration_overview.pdf", PAPER / "figures/calibration_overview.pdf")
    evidence = dict(scoring_version=summary["scoring_version"], planar=estimates,
                    legacy=[r for r in summary["summaries"] if r["suite"] == "torque-excitation"],
                    lingbot=spatial['groups'], matched_spatial_reference=replay,
                    bootstrap=dict(draws=2000, seed=20261016,
                        units="training seeds and assemblies within family; conditions retained together"),
                    sources=SOURCES)
    (PAPER / "sources/unified_scoring.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"planar": estimates, "lingbot": evidence["lingbot"]}, indent=2))


if __name__ == "__main__":
    main()
