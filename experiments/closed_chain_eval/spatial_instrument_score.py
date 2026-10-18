"""Score external pose instruments with matching observed-prefix calibration."""
import json
from pathlib import Path
import numpy as np
from .visual_calibration import ROOT,OUT,readout_path,error_statistics,write,sha


def measure_reader(task,source):
    from experiments.model_expansion.evaluate_spatial import Audit
    from .rescore_continuous import spatial_values
    row,reader=task
    audit=Audit(row);path=readout_path(reader,row)
    with np.load(path) as d:q,t=d["qpos"].astype(float),d["time"]
    if reader["native_resolution"]==256:
        prefix_id=reader["id"].rsplit("_",1)[0]+"_384"
        with np.load(OUT/"readouts"/prefix_id/path.name) as d:q[:3]=d["qpos"][:3]
    with np.load(ROOT/row["evaluator_directory"]/"physics.npz") as d:
        ix=np.arange(120)*8
        truth=d["qpos"][ix]
        true_v,true_a=d["qvel"][ix],d["qacc"][ix]
    m=audit.measure(t,q,truth)
    points,ref=np.stack([audit.points(v) for v in q]),np.stack([audit.points(v) for v in truth])
    original_path=OUT/"measurements/shared_spatial_384"/(row["video_id"]+".json")
    if original_path.exists():
        original=json.loads(original_path.read_text())
    else:
        original=dict(numerical_reference_exact=spatial_values(audit.measure(t,truth,truth,(true_v,true_a))),
                      numerical_reference_sampled=spatial_values(audit.measure(t,truth,truth)))
    result=dict(reader_id=reader["id"],kind="spatial",category=row["category"],assembly_id=row["assembly_id"],
        case_id=row["video_id"],calibration_role=row["calibration_role"],
        identity=dict(protocol_sha256=sha(OUT/"protocol.json"),scorer_sha256=sha(source),readout_sha256=sha(path)),
        localization=error_statistics(points[3:]-ref[3:]),native_metrics=spatial_values(m),
        native_qualified=m.get("mechanical_coverage",False),
        numerical_reference_exact=original["numerical_reference_exact"],numerical_reference_sampled=original["numerical_reference_sampled"],
        native_raw_equation_diagnostics=dict(dynamics=m.get("raw_equivalent_effort_rmse"),energy=m.get("raw_work_energy_max_J")))
    try:
        projected=np.stack([audit.project(v) for v in q]); pm=audit.measure(t,projected,truth)
        pp=np.stack([audit.points(v) for v in projected])
        result.update(intervention_metrics=spatial_values(pm),intervention_reachable=True,
            intervention_qualified=pm.get("mechanical_coverage",False),
            correction_point_rms_m=float(np.sqrt(np.mean(np.sum((pp[3:]-points[3:])**2,axis=-1)))))
    except ValueError as error:
        result.update(intervention_metrics={},intervention_reachable=False,intervention_qualified=False,intervention_failure=str(error))
    write(OUT/"measurements"/reader["id"]/(row["video_id"]+".json"),result)
    return result
