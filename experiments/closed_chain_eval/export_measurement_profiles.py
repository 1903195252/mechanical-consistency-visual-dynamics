"""Export unit-checked empirical reference-error profiles for exact pipelines.

Profiles describe instrument errors; model-score normalization is disabled.
Missing physical terms stay missing instead of fitting on successful subsets.
"""
from collections import defaultdict
import hashlib
import json
from .core import CORE_METRICS
from .continuous import calibrate_noise_profile
from .visual_calibration import ROOT,OUT,sha,write,load_protocol
from .summarize_visual_calibration import verify_bound


def export_group(identity,rows,track,role):
    calibration=[r for r in rows if r["calibration_role"]=="calibration"]
    check=[r for r in rows if r["calibration_role"]=="verification"]
    units={"geometry":"m","constraint_velocity":"m/s","constraint_acceleration":"m/s^2",
           "dynamics":"N" if rows[0]["category"]=="H-A" else "N m","energy":"J","prediction":"m"}
    key=track+"_metrics"
    incomplete=[r["case_id"] for r in calibration if any(r.get(key,{}).get(m) is None for m in CORE_METRICS)]
    entry=dict(measurement_id=identity,readout_role=role,calibration_cases=len(calibration),
        calibration_assemblies=len({r["assembly_id"] for r in calibration}),model_score_normalization_enabled=False)
    if incomplete:
        return dict(entry,status="incomplete_physical_qualification",profile=None,incomplete_cases=incomplete,
                    required_action="continue improving the reader/measurement; never silently drop these references")
    cards=[dict(metrics={m:dict(complete=True,units=[units[m]],raw_per_channel=[r[key][m]]) for m in CORE_METRICS})
           for r in calibration]
    source_id=hashlib.sha256(json.dumps(calibration,sort_keys=True,allow_nan=False).encode()).hexdigest()
    profile=calibrate_noise_profile(cards,profile_id=identity,measurement_id=identity,source_id=source_id,split="calibration")
    verification={m:verify_bound(check,lambda r,k=m:r.get(key,{}).get(k),profile["metrics"][m]["floor_per_channel"][0])
                  for m in CORE_METRICS}
    return dict(entry,status="measured_reference_errors",profile=profile,verification=verification,
                interpretation="empirical upper quantiles, not confidence bounds; reconstructed tracks remain interventions")


def run():
    s=json.loads((OUT/"summary.json").read_text())
    protocol=load_protocol();entries=[]
    for item in s["summaries"]:
        records=protocol[item["kind"]+"_records"]
        rows=[]
        for r in records:
            if r["category"]!=item["category"]:continue
            key=r.get("case_id",r.get("video_id"))
            rows.append(json.loads((OUT/"measurements"/item["reader"]/(key+".json")).read_text()))
        for track in ("native","intervention"):
            identity="/".join((item["reader"],item["category"],track))
            entries.append(export_group(identity,rows,track,item["role"]))
    path=OUT/"paired_robust_planar_rgb/calibration.json"
    if path.exists():
        groups=defaultdict(list)
        for r in json.loads(path.read_text())["rows"]:groups[(r["model"],r["category"])].append(r)
        for (quality,family),rows in sorted(groups.items()):
            for track in ("native","intervention"):
                entries.append(export_group("/".join(("robust_planar_64",quality,family,track)),rows,track,
                                            "external RGB instrument; fixed blur/contrast calibration condition"))
    write(OUT/"reference_error_profiles.json",dict(schema="independent-measurement-errors/1",entries=entries,
        complete_profiles=sum(e["profile"] is not None for e in entries),
        engineering_thresholds_changed=False,model_score_normalization_enabled=False,
        no_rmse_subtraction=True,source_sha256=sha(__file__)))
    print(json.dumps(dict(profiles=len(entries),complete=sum(e["profile"] is not None for e in entries))))


if __name__=="__main__":run()
