"""Repeat paired RGB measurement with the frozen quality-robust instrument."""
from concurrent.futures import ProcessPoolExecutor
import json
import numpy as np
from .visual_calibration import OUT,load_protocol,sha,write
from . import remeasure_planar_rgb as pipeline


def run():
    pipeline.DIRECTORY=OUT/"paired_robust_planar_rgb"
    pipeline.READER_DIRECTORY=OUT/"optimization/robust_planar_reader"
    pipeline.run()
    rows=load_protocol()["planar_records"]
    qualities=("clean","mild_blur","blur","blur_low_contrast")
    tasks=[]
    for quality in qualities:
        name="calibration_"+quality
        for row in rows:
            source=pipeline.READER_DIRECTORY/"stress_readouts/robust"/quality/(row["case_id"]+".npz")
            with np.load(source) as data:points=data["points"]
            target=pipeline.DIRECTORY/"readouts"/name/source.name;target.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(target,points=points,time=np.arange(90)/30,source_sha256=np.array(sha(source)))
            tasks.append((row,name))
    with ProcessPoolExecutor(max_workers=6) as pool:results=list(pool.map(pipeline.measure_case,tasks))
    roles={r["case_id"]:r["calibration_role"] for r in rows}
    for r in results:r["calibration_role"]=roles[r["case_id"]]
    write(pipeline.DIRECTORY/"calibration.json",dict(rows=results,qualities=qualities,
        reader_sha256=sha(pipeline.READER_DIRECTORY/"final.pt"),test_used_for_calibration=False,
        all_model_forecasts_unchanged=True,source_sha256=sha(__file__)))
    print(json.dumps(dict(quality_calibration_measurements=len(results))),flush=True)


if __name__=="__main__":run()
