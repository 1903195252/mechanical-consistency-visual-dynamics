"""Count only actual, fully validated assembly/video artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from .catalog import OUTPUT,connect,summary,write_json
from .naming import canonical_entry, video_aliases


def verify_assembly(entry):
    lookup = connect()
    entry = canonical_entry(lookup, entry)
    lookup.close()
    folder=OUTPUT/"assemblies"/entry["category"]/entry["assembly_id"]
    assembly=json.loads((folder/"assembly.json").read_text())
    video_aliases(folder, assembly["assembly_id"], entry["assembly_id"])
    reasons=[];videos=[];db=connect();physics_valid=True
    for condition in ["low_speed","rated","limit_or_load"]:
        path=folder/"conditions"/condition/"physics_validation.json"
        if not path.exists():
            reasons.append("Missing physics: "+condition);physics_valid=False;continue
        result=json.loads(path.read_text())
        if not result.get("passed") or result.get("assembly_physical_digest")!=assembly["physical_digest"]:
            reasons.append("Invalid/stale physics: "+condition)
            physics_valid=False
        target={"low_speed":1.,"rated":3.,"limit_or_load":5.}[condition]
        if "achieved_mean_crank_speed_rad_s" in result.get("metrics",{}):
            if not .65*target<result["metrics"]["achieved_mean_crank_speed_rad_s"]<1.35*target:
                physics_valid=False;reasons.append("Operating speed not attained: "+condition)
    for row in list(db.execute("SELECT * FROM videos WHERE assembly_id=? ORDER BY video_index",(entry["assembly_id"],))):
        path=folder/"videos"/(row["video_id"]+".mp4")
        sidecar=path.with_suffix(".json")
        if not path.exists() or not sidecar.exists():
            reasons.append("Missing video: "+row["video_id"])
            db.execute("UPDATE videos SET status='planned',last_error='Missing file or sidecar' WHERE video_id=?",(row["video_id"],))
            continue
        try:
            metadata=json.loads(sidecar.read_text())
            if metadata["benchmark_only"] or metadata["assembly_physical_digest"]!=assembly["physical_digest"]:
                raise ValueError("Benchmark or stale simulation")
            if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata["sha256"]:
                raise ValueError("Video hash mismatch")
            info=json.loads(subprocess.check_output(["ffprobe","-v","error","-select_streams","v:0",
                              "-count_frames","-show_entries","stream=width,height,nb_read_frames,r_frame_rate,codec_name",
                              "-of","json",str(path)],text=True))["streams"][0]
            if (info["width"],info["height"],int(info["nb_read_frames"]),info["r_frame_rate"])!=(640,640,120,"30/1"):
                raise ValueError("Unexpected video dimensions, frame count or frame rate")
            if info["codec_name"]!="h264":raise ValueError("Unexpected codec")
            if len(metadata["camera_frames"])!=120 or len(metadata["frame_times_s"])!=120:
                raise ValueError("Missing camera/time metadata")
            if (metadata["pose_check"]["maximum_position_error_m"]>3e-7
                    or metadata["pose_check"]["maximum_quaternion_error"]>3e-6):
                raise ValueError("Pose mismatch")
            if not np.allclose(metadata["frame_times_s"],np.arange(120)/30,atol=1e-8,rtol=0):
                raise ValueError("Video/physics timestamps differ")
            for camera in metadata["camera_frames"]:
                rotation=np.asarray(camera["R_world_to_camera"])
                if (not np.isfinite(rotation).all() or not np.isfinite(camera["t_world_to_camera_m"]).all()
                        or not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-5)
                        or abs(np.linalg.det(rotation)-1)>1e-5 or camera["scale_m"]<=0):
                    raise ValueError("Invalid camera calibration")
            first_camera=metadata['camera_frames'][0]
            if any(not np.allclose(camera['R_world_to_camera'],first_camera['R_world_to_camera'],atol=1e-9,rtol=0)
                   or not np.allclose(camera['t_world_to_camera_m'],first_camera['t_world_to_camera_m'],atol=1e-9,rtol=0)
                   for camera in metadata['camera_frames']):
                raise ValueError('Released views require fixed camera extrinsics')
            if metadata["condition"]!=row["condition"] or metadata["view"]!=row["view"]:
                raise ValueError("Wrong condition/view assigned to video ID")
            for reference,hash_key in [(metadata["physical_trajectory"],"physics_trajectory_sha256"),
                                       (metadata["commands"],"commands_sha256")]:
                referenced=folder/reference
                if not referenced.is_file():raise ValueError("Missing referenced trajectory")
                if hash_key in metadata and hashlib.sha256(referenced.read_bytes()).hexdigest()!=metadata[hash_key]:
                    raise ValueError("Referenced trajectory changed since rendering")
            safe_inputs=json.loads(path.with_suffix(".inputs.json").read_text())
            if len(safe_inputs["observed_camera_frames"])!=3:raise ValueError("Unexpected public camera prefix")
            for key in ["actual_motor_torque_Nm","actual_actuator_force_N","ball_constraint_force_N","camera_frames","q","qdot","qddot"]:
                if key in safe_inputs:raise ValueError("Future reference field in public inputs: "+key)
            if not (path.parent/safe_inputs["commanded_actions"]).is_file():
                raise ValueError("Missing public command trajectory")
            record={"video_id":row["video_id"],"passed":True,"video":str(path.relative_to(OUTPUT)),
                    "decoded":info,"bytes":path.stat().st_size,"sha256":metadata["sha256"],
                    "scope":"Video decoding and metadata. Assembly acceptance requires completion_audit.json."}
            validation=path.with_suffix(".validation.json")
            # Keep legacy signed validation bytes; new acceptance lives in its own sidecar.
            if validation.is_symlink():
                validation=path.with_suffix(".canonical_validation.json")
            write_json(validation,record);videos.append(record)
            db.execute("UPDATE videos SET status='decoded_validated',path=?,validation_path=?,last_error=NULL WHERE video_id=?",
                       (str(path.relative_to(OUTPUT)),str(validation.relative_to(OUTPUT)),row["video_id"]))
        except Exception as error:
            reasons.append(row["video_id"]+": "+str(error))
            db.execute("UPDATE videos SET status='failed',last_error=? WHERE video_id=?",(str(error),row["video_id"]))
    solid_path=folder/"blender_solid_validation.json"
    solid=json.loads(solid_path.read_text()) if solid_path.exists() else {}
    if (not solid.get("passed") or solid.get("assembly_physical_digest")!=assembly["physical_digest"]
            or solid.get("parts_checked")!=len(assembly["parts"])):
        reasons.append("Blender solid audit is missing or failed")
    if solid.get("blend_sha256") and (folder/"assembly.blend").is_file():
        if hashlib.sha256((folder/"assembly.blend").read_bytes()).hexdigest()!=solid["blend_sha256"]:
            reasons.append("Blender scene changed since solid audit")
    visual_path=folder/"visual_review.json"
    visual=json.loads(visual_path.read_text()) if visual_path.exists() else {}
    if not visual.get("passed") or visual.get("assembly_physical_digest")!=assembly["physical_digest"]:
        reasons.append("Template/assembly visual review is missing")
    clearance_path=folder/"blender_clearance_validation.json"
    clearance=json.loads(clearance_path.read_text()) if clearance_path.is_file() else {}
    expected={(condition,frame) for condition in ['low_speed','rated','limit_or_load'] for frame in range(120)}
    checked={(row['condition'],row['frame']) for row in clearance.get('frames_checked',[]) if row['failures']==0}
    if (not clearance.get('passed') or clearance.get('assembly_physical_digest')!=assembly['physical_digest']
            or checked!=expected):
        reasons.append('Complete 360-pose mesh interference audit missing or failed')
    if not clearance.get('nominal_mesh_digest') or clearance.get('nominal_mesh_digest')!=solid.get('nominal_mesh_digest'):
        reasons.append('Clearance and mass checks refer to different meshes')
    for condition,checksum in clearance.get('render_pose_sha256',{}).items():
        if hashlib.sha256((folder/'conditions'/condition/'render_poses.npz').read_bytes()).hexdigest()!=checksum:
            reasons.append('Poses changed after clearance audit: '+condition)
    if assembly.get('permitted_contacts'):
        rules=assembly['permitted_contacts']
        if clearance.get('contact_rules_sha256')!=hashlib.sha256(json.dumps(rules,sort_keys=True).encode()).hexdigest():
            reasons.append('Declared-contact rule changed after audit')
        expected_contacts={(condition,frame,frozenset([rule['cylinder_part'],rule['box_part']]))
                           for condition in ['low_speed','rated','limit_or_load'] for frame in range(120) for rule in rules}
        checked_contacts={(r['condition'],r['frame'],frozenset(r['parts']))
                          for r in clearance.get('declared_contact_checks',[]) if r.get('passed')}
        if checked_contacts!=expected_contacts:reasons.append('Incomplete declared-contact geometry/force checks')
        for condition in ['low_speed','rated','limit_or_load']:
            checksum=hashlib.sha256((folder/'conditions'/condition/'physics.npz').read_bytes()).hexdigest()
            if clearance.get('contact_reference_sha256',{}).get(condition)!=checksum:
                reasons.append('Contact-force reference changed after audit: '+condition)
    if visual.get("mode")=="reviewed_template_with_instance_checks":
        reference=OUTPUT/visual["template_review"]
        if not reference.is_file() or hashlib.sha256(reference.read_bytes()).hexdigest()!=visual["template_review_sha256"]:
            reasons.append("Template review reference changed")
        for record in videos:
            meta=json.loads((OUTPUT/record["video"]).with_suffix(".json").read_text())
            if not meta.get("framing_check",{}).get("passed"):
                reasons.append("Missing per-instance framing audit: "+record["video_id"])
    duplicates=db.execute("SELECT assembly_id FROM assemblies WHERE geometry_digest=? AND assembly_id!=?",
                          (assembly["geometry_digest"],entry["assembly_id"])).fetchall()
    if duplicates:reasons.append("Duplicate geometric assembly")
    passed=not reasons and len(videos)==15 and (folder/"assembly.blend").exists()
    if passed:
        db.execute("UPDATE videos SET status='verified',last_error=NULL WHERE assembly_id=?",(entry['assembly_id'],))
    result={"assembly_id":entry["assembly_id"],"passed":passed,"verified_videos":len(videos) if passed else 0,
            "decoded_video_files":len(videos),
            "reasons":reasons,"geometry_digest":assembly["geometry_digest"],"parts":len(assembly["parts"])}
    write_json(folder/"completion_audit.json",result)
    db.execute("UPDATE assemblies SET status=?,last_error=? WHERE assembly_id=?",
               ("verified" if passed else ("rendered" if physics_valid and len(videos)==15 else "physics_validated" if physics_valid else "built"),
                None if passed else "; ".join(reasons),entry["assembly_id"]))
    db.commit();summary(db);db.close()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--assembly",required=True)
    args=parser.parse_args()
    db=connect();row=db.execute("SELECT * FROM assemblies WHERE assembly_id=?",(args.assembly,)).fetchone();db.close()
    if row is None:raise ValueError("Assembly not in catalog")
    result=verify_assembly(dict(row));print(json.dumps(result,ensure_ascii=False,indent=2))
    if not result["passed"]:raise SystemExit(1)


if __name__=="__main__":main()
