"""Calibrate each instrument once at the mechanism's known CAD reset pose.

Use the low-speed sequence's first observed image as a calibration exposure.
The reference pose is model.qpos0 from the fixed CAD, not future trajectory
labels. Freeze this offset for all three conditions of that assembly. This is
an explicit calibrated-setup track, separate from the original uncalibrated
observer. Only the initial pose is anchored; future closure is unconstrained.
"""
from concurrent.futures import ProcessPoolExecutor
import json

import numpy as np
from scipy.spatial.transform import Rotation

from .visual_calibration import ROOT, OUT, load_protocol, write, sha
from .spatial_instrument_score import measure_reader


DIRECTORY=OUT/"optimization/reset_calibration"


def align(q,observed_reset,known_reset):
    q=np.asarray(q,dtype=float).copy()
    q[:,:3]+=known_reset[:3]-observed_reset[:3]
    rotation=Rotation.from_quat(known_reset[[4,5,6,3]])*Rotation.from_quat(observed_reset[[4,5,6,3]]).inv()
    quaternion=(rotation*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()
    q[:,3:7]=quaternion[:,[3,0,1,2]]
    q[:,7:]+=known_reset[7:]-observed_reset[7:]
    return q


def score(task):return measure_reader(task,source=__file__)


def run():
    from .geometry_spatial_reader import nominal
    protocol=load_protocol();rows=protocol["spatial_records"]
    calibration={r["assembly_id"]:r for r in rows if r["condition"]=="low_speed"}
    offsets={}
    for key,row in calibration.items():
        path=OUT/"readouts/geometry_spatial_384"/(row["video_id"]+".npz")
        with np.load(path) as data:observed=data["qpos"][0].astype(float)
        known=nominal(row).astype(float)
        # This assertion checks the dataset's registered reset condition. Its
        # trajectory labels do not determine any calibration parameter.
        with np.load(ROOT/row["evaluator_directory"]/"physics.npz") as data:
            if not np.allclose(data["qpos"][0],known,atol=1e-7,rtol=0):
                raise ValueError("This dataset does not begin at the known CAD reset")
        offsets[key]=dict(observed_reset=observed.tolist(),known_reset=known.tolist(),
                         calibration_image=row["video_id"],source_readout_sha256=sha(path))
    write(DIRECTORY/"calibration.json",dict(source_sha256=sha(__file__),offsets=offsets,
        one_calibration_exposure_per_assembly=True,uses_future_truth_to_fit=False,
        information="explicit known CAD reset pose and one observed reference frame; frozen for all conditions",
        applicable_only_when_known_reset_is_available=True))
    readers=[]
    for width in (384,256):
        reader=dict(id="reset_spatial_"+str(width),kind="spatial",native_resolution=width,encoder_resolution=384,
            role="external instrument; calibrated at known reset before forecasting",
            calibration_sha256=sha(DIRECTORY/"calibration.json"))
        readers.append(reader)
        for row in rows:
            old=OUT/"readouts"/("geometry_spatial_"+str(width))/(row["video_id"]+".npz")
            with np.load(old) as data:q,t=data["qpos"],data["time"]
            c=offsets[row["assembly_id"]]
            q=align(q,np.array(c["observed_reset"]),np.array(c["known_reset"]))
            path=OUT/"readouts"/reader["id"]/old.name;path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,qpos=q,time=t,calibration_sha256=np.array(sha(DIRECTORY/"calibration.json")))
    with ProcessPoolExecutor(max_workers=6) as pool:
        measured=list(pool.map(score,[(row,r) for r in readers for row in rows]))
    write(DIRECTORY/"readers.json",readers)
    summary=[]
    for reader in readers:
        for family in ("H-A","H-B"):
            group=[r for r in measured if r["reader_id"]==reader["id"] and r["category"]==family and r["calibration_role"]=="verification"]
            summary.append(dict(reader=reader["id"],family=family,total=len(group),
                point_rmse_mm=1000*float(np.mean([r["localization"]["point_euclidean_rmse"] for r in group])),
                geometry_mm=1000*float(np.mean([r["native_metrics"]["geometry"] for r in group])),
                qualified=sum(r["native_qualified"] for r in group),
                projected_qualified=sum(r["intervention_qualified"] for r in group)))
    write(DIRECTORY/"summary.json",summary);print(json.dumps(summary),flush=True)


if __name__=="__main__":run()
