"""Verify frozen sources and record measurement roles across experimental suites."""
import json
import platform
from pathlib import Path
from .visual_calibration import ROOT,OUT,load_protocol,sha,write


def run():
    p=load_protocol();checks=[]
    for reader in p["readers"]:
        if reader["kind"]=="planar":
            path=ROOT/reader["folder"]
            if sha(path/"params.p")!=reader["checkpoint_sha256"]:raise ValueError("Original model weights changed")
            mapping="scores.json" if reader["model"]=="keycld" else "readout.json"
            if sha(path/mapping)!=reader["mapping_sha256"]:raise ValueError("Original readout mapping changed")
            checks.append(dict(reader=reader["id"],original_weights_and_mapping_unchanged=True))
    spatial=[r for r in p["readers"] if r["kind"]=="spatial"][0]
    if sha(ROOT/"output/model_expansion/weights/vjepa2_1_vitb_dist_vitG_384.pt")!=spatial["encoder_sha256"]:
        raise ValueError("Frozen encoder changed")
    for family,expected in spatial["readout_sha256"].items():
        if sha(ROOT/"output/model_expansion/spatial"/(family+"_readout.pt"))!=expected:
            raise ValueError("Original spatial head changed")
    original_sources=json.loads((ROOT/"output/closed_chain_eval/continuous/scores.json").read_text())["sources"]
    for name,expected in original_sources.items():
        if sha(ROOT/name)!=expected:raise ValueError("Original measurement changed: "+name)
    for kind in ("planar","spatial"):
        records=p[kind+"_records"]
        a={r["assembly_id"] for r in records if r["calibration_role"]=="calibration"}
        b={r["assembly_id"] for r in records if r["calibration_role"]=="verification"}
        if a&b:raise ValueError("Calibration/verification identity overlap")
    phase_controls=json.loads((OUT/"optimization/constrained_visual_phase/sensitivity.json").read_text())
    control=phase_controls["summary"]
    if control["correct"]["passed"]!=control["correct"]["total"]:
        raise ValueError("Selected visual derivative damages exact reference controls")
    for name in ("time_scale_0_8","phase_1Hz_0_1rad","wrong_actuator_plus_0_5Nm"):
        if control[name]["passed"]:
            raise ValueError("Selected derivative hides a registered physical defect")
    phase_rows=json.loads((OUT/"paired_constrained_visual_phase/summary.json").read_text())["rows"]
    if len(phase_rows)!=756 or sum(not r["finite"] for r in phase_rows)!=13:
        raise ValueError("Paired phase measurement changed the registered denominator")
    for row in phase_rows:
        original=json.loads((OUT/"paired_robust_planar_rgb/measurements"/row["model"]/(row["case_id"]+".json")).read_text())
        if row["raw_geometry_max_m"]!=original.get("native_metrics",{}).get("geometry"):
            raise ValueError("Phase intervention overwrote raw geometric closure")
        if row["raw_point_euclidean_rmse_m"]!=original.get("point_euclidean_rmse_m"):
            raise ValueError("Phase intervention overwrote raw point errors")
    archive={}
    source_paths=list((ROOT/"experiments/closed_chain_eval").glob("*.py"))+[
        ROOT/"experiments/keycld_corpus/physics_score.py",ROOT/"experiments/keycld_corpus/prepare.py",
        ROOT/"experiments/con_corpus/crank_readout.py",ROOT/"experiments/model_expansion/evaluate_spatial.py"]
    for path in source_paths:
        digest=sha(path);target=OUT/"source_archive"/(digest+".py")
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():target.write_bytes(path.read_bytes())
        archive[str(path.relative_to(ROOT))]=digest
    write(OUT/"source_archive/index.json",archive)
    registry=[
        dict(suite="legacy torque-excitation / KeyCLD",measurement="native model keypoints and ODE derivatives; direct RGB metrics",
             existing_controls=["output/keycld_fourbar/diagnosis/report.json","output/keycld_fourbar/scored_evaluation/calibration.json"],
             rule="native encoder errors belong to the declared model pipeline; do not borrow a 30Hz/shared-FOV visual noise profile"),
        dict(suite="shared planar / KeyCLD and CON native state or latent readout",measurement="native coordinates/derivatives and registered mappings",
             controls="protocol.json readers keycld_64_seed* and con_64_seed*",
             rule="separate encoder/affine-interface error from dynamics; model-owned readouts cannot calibrate away their own errors"),
        dict(suite="shared planar / common RGB instrument",measurement="fixed external RGB-to-landmark model plus declared differentiation",
             controls="paired_robust_planar_rgb/calibration.json",paired_remeasurement="paired_robust_planar_rgb/summary.json",
             rule="same observer for all seeds/models and correct-video controls; raw geometry remains distinct from reconstruction"),
        dict(suite="shared planar / filtered phase intervention",measurement="fixed phase fit plus explicit geometric reconstruction",
             controls="optimization/constrained_visual_phase/verification.json; optimization/constrained_visual_phase/sensitivity.json",
             paired_remeasurement="paired_constrained_visual_phase/summary.json",
             rule="exact-state acceptance gate and fault controls passed; retain raw geometry and unfiltered companions, report frequency response and remaining reference torque/energy error"),
        dict(suite="spatial / V-JEPA AC",measurement="released encoder, trained physical head and autoregressive predictor",
             controls="shared_spatial_384 reference replay",
             rule="the physical head also supplies autoregressive state feedback; replacing it changes the forecaster unless an independent measurement head is used only on outputs"),
        dict(suite="spatial / LingBot",measurement="external RGB observer",
             controls="geometry_spatial_* / reset_spatial_*; spatial_confirmation/summary.json",
             paired_remeasurement="paired_lingbot/summary.json",
             rule="geometry-coordinate and known-reset setups remain separate; do not promote every observer-produced violation to a model-only failure")]
    for entry in registry:
        for path in entry.get("existing_controls",[]):
            if not (ROOT/path).exists():raise FileNotFoundError(path)
    write(OUT/"measurement_registry.json",dict(suites=registry,engineering_limits_unchanged=True,
        readout_errors_trigger_optimization=True,rmse_subtraction_for_model_attribution=False,
        status="calibration and optimization evidence recorded; full model-only attribution is not assumed"))
    write(OUT/"verification.json",dict(protocol_sha256=sha(OUT/"protocol.json"),model_checks=checks,
        original_spatial_encoder_and_heads_unchanged=True,original_measurement_sources_verified=len(original_sources),
        calibration_verification_identity_disjoint=True,source_files_archived=len(archive),python=platform.python_version(),
        visual_phase_exact_state_controls_passed=control["correct"]["passed"],
        paired_phase_rows_with_raw_errors_preserved=len(phase_rows),nonfinite_predictions_retained=13,
        complete_visual_force_energy_acceptance=False,
        model_forecasts_reused_not_retrained=True,cloud_compute_used=False))
    print(json.dumps(dict(original_measurement_sources_verified=len(original_sources),source_files_archived=len(archive))))


if __name__=="__main__":run()
