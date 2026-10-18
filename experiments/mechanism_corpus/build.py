"""Build supported corpus templates; unknown categories remain explicitly pending."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

from .catalog import OUTPUT, connect, summary, write_json
from . import fourbar,slider_crank,double_crank,stewart,gear_pair,cam_follower,delta
from . import physics_planar,physics_stewart,physics_gear,physics_cam,physics_delta
from .physics_fourbar import simulate,source_digest


def build_one(entry, simulate_physics=True):
    mechanisms={"fourbar":fourbar,"slider_crank":slider_crank,"parallelogram":double_crank,"stewart":stewart,"gear_pair":gear_pair,"cam_follower":cam_follower,"delta":delta}
    if entry["builder"] not in mechanisms:
        raise NotImplementedError("Template is not implemented yet: "+entry["category"])
    mechanism=mechanisms[entry["builder"]]
    output=OUTPUT/"assemblies"/entry["category"]/entry["assembly_id"]
    output.mkdir(parents=True,exist_ok=True)
    assembly=mechanism.build(entry)
    write_json(output/"assembly.json",assembly)
    xml,_=mechanism.mjcf(assembly["config"],assembly["bodies"])
    (output/"model.xml").write_text(xml)
    write_json(output/"assembly_validation.json",assembly["assembly_audit"])
    db=connect()
    db.execute("UPDATE assemblies SET status='built',geometry_digest=?,manifest_path=?,last_error=NULL WHERE assembly_id=?",
               (assembly["geometry_digest"],str((output/"assembly.json").relative_to(OUTPUT)),entry["assembly_id"]))
    db.commit()
    if simulate_physics:
        fingerprint=(source_digest() if mechanism is fourbar else physics_stewart.source_digest() if mechanism is stewart
                     else physics_gear.source_digest() if mechanism is gear_pair
                     else physics_cam.source_digest() if mechanism is cam_follower
                     else physics_delta.source_digest() if mechanism is delta
                     else physics_planar.source_digest(mechanism))
        for name in ["low_speed","rated","limit_or_load"]:
            directory=output/"conditions"/name
            result=directory/"physics_validation.json"
            previous=json.loads(result.read_text()) if result.exists() else {}
            if previous.get("passed") and previous.get("assembly_physical_digest")==assembly["physical_digest"] \
                    and previous.get("generator_source_digest")==fingerprint:
                print("Verified existing physics",entry["assembly_id"],name,flush=True)
                continue
            report=(simulate(assembly,name,directory) if mechanism is fourbar else
                    physics_stewart.simulate(assembly,name,directory) if mechanism is stewart else
                    physics_gear.simulate(assembly,name,directory) if mechanism is gear_pair else
                    physics_cam.simulate(assembly,name,directory) if mechanism is cam_follower else
                    physics_delta.simulate(assembly,name,directory) if mechanism is delta else
                    physics_planar.simulate(assembly,name,directory,mechanism))
            print(json.dumps({"assembly":entry["assembly_id"],"condition":name,"physics":report["metrics"]},ensure_ascii=False),flush=True)
        db.execute("UPDATE assemblies SET status='physics_validated' WHERE assembly_id=?",(entry["assembly_id"],))
        db.commit()
    summary(db);db.close()
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category",default="S-A")
    parser.add_argument("--start",type=int,default=1)
    parser.add_argument("--count",type=int,default=1)
    parser.add_argument("--geometry-only",action="store_true")
    parser.add_argument("--workers",type=int,default=1)
    args=parser.parse_args()
    db=connect()
    entries=[dict(row) for row in db.execute(
        "SELECT * FROM assemblies WHERE category=? AND ordinal>=? AND ordinal<? ORDER BY ordinal",
        (args.category,args.start,args.start+args.count))]
    if len(entries)!=args.count:raise ValueError("Requested assembly range differs from catalog")
    pending=[]
    for entry in entries:
        if entry["status"]=="verified":
            print("Preserving completed assembly",entry["assembly_id"]);continue
        pending.append(entry)
    if args.workers>1:
        db.close()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
            futures={pool.submit(build_one,entry,not args.geometry_only):entry for entry in pending}
            for completed,future in enumerate(as_completed(futures),1):
                entry=futures[future]
                try:future.result()
                except Exception as error:
                    current=connect();current.execute("UPDATE assemblies SET last_error=? WHERE assembly_id=?",
                                                     (str(error),entry["assembly_id"]));current.commit();current.close()
                    raise
                print(json.dumps({"completed_assemblies_in_job":completed,"total":len(pending),
                                  "assembly_id":entry["assembly_id"]}),flush=True)
        current=connect();summary(current);current.close()
        return
    for entry in pending:
        try:
            build_one(entry,not args.geometry_only)
        except Exception as error:
            db.execute("UPDATE assemblies SET last_error=? WHERE assembly_id=?",(str(error),entry["assembly_id"]))
            db.commit();summary(db)
            raise
    db.close()


if __name__=="__main__":
    main()
