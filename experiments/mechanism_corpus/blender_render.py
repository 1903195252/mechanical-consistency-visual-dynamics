"""Build the shared mechanical scene and replay exported MuJoCo body poses."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

RENDERER_SOURCE_BYTES=Path(__file__).read_bytes()
RENDERER_SOURCE_SHA256=hashlib.sha256(RENDERER_SOURCE_BYTES).hexdigest()

import bpy
from mathutils import Matrix,Quaternion,Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments/fourbar"))
from experiments.mechanism_corpus.solids import MATERIALS
from experiments.mechanism_corpus.catalog import write_json
from geometry import link_mesh
import visual_parts as vp


def annular_hex(name,position,outer,inner,depth,material,parent):
    n=96;vertices=[];faces=[]
    for i in range(n):
        angle=2*math.pi*i/n
        radius=outer*math.cos(math.pi/6)/math.cos(angle%(math.pi/3)-math.pi/6)
        for r,y in [(radius,-depth/2),(radius,depth/2),(inner,-depth/2),(inner,depth/2)]:
            vertices.append((r*math.cos(angle),y,r*math.sin(angle)))
    for i in range(n):
        a,b=4*i,4*((i+1)%n)
        faces.extend([(a+1,b+1,b,a),(b+2,b+3,a+3,a+2),(b,b+2,a+2,a),(a+3,b+3,b+1,a+1)])
    mesh=bpy.data.meshes.new(name+"_mesh");mesh.from_pydata(vertices,[],faces);mesh.update()
    obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
    obj.parent=parent;obj.location=position;mesh.materials.append(material)
    return vp.finish(obj,.0001)


def create_part(item,parents,materials):
    shape=item["shape"];dims=item["dimensions_m"];name=item["name"]
    position=item["position_m"];parent=parents[item["body"]];mat=materials[item["material"]]
    if shape in ("box","bearing_block","rectangular_sleeve"):
        obj=vp.box(name,position,dims["size"],mat,parent,bevel=min(.001,min(dims["size"])/8))
        if shape=="bearing_block":
            cutter=vp.cylinder(name+"_bore_cutter",position,dims["bore_radius"],dims["size"][1]+.01,mat,parent)
            vp.cut_fixture(obj,cutter)
        elif shape=="rectangular_sleeve":
            cutter=vp.box(name+"_guide_cutter",position,[dims["size"][0]+.01,*dims["opening_yz"]],mat,parent,bevel=0)
            vp.cut_fixture(obj,cutter)
    elif shape=="link":
        vertices,faces=link_mesh(dims["length"],dims["width"],dims["thickness"],dims["bore_radius"])
        mesh=bpy.data.meshes.new(name+"_mesh");mesh.from_pydata(vertices,[],faces);mesh.update()
        for face in mesh.polygons:face.use_smooth=abs(face.normal.y)<.1
        obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
        obj.parent=parent;obj.location=position;mesh.materials.append(mat);vp.finish(obj,.00022)
    elif shape=="ring":
        obj=vp.ring(name,position,dims["radius"],dims["inner_radius"],dims["depth"],mat,parent)
        if item["axis"]=="Z":obj.rotation_euler=(-math.pi/2,0,0)
    elif shape=="cylinder":
        obj=vp.cylinder(name,position,dims["radius"],dims["depth"],mat,parent)
        if item["axis"]=="Z":obj.rotation_euler=(0,0,0)
    elif shape=="socket_head":
        obj=vp.socket_head(name,position,dims["radius"],dims["depth"],mat,parent,axis=item["axis"])
    elif shape=="hex_nut":
        obj=annular_hex(name,position,dims["radius"],dims["inner_radius"],dims["depth"],mat,parent)
    elif shape=="axial_profile":
        from experiments.mechanism_corpus.spatial_solids import profile_mesh
        vertices,faces=profile_mesh(dims)
        mesh=bpy.data.meshes.new(name+"_mesh");mesh.from_pydata(vertices,[],faces);mesh.update()
        obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
        obj.parent=parent;obj.location=position;mesh.materials.append(mat)
        for face in mesh.polygons:face.use_smooth=len(face.vertices)==4
    elif shape=="eccentric_disk":
        center=list(position);center[0]+=dims['eccentricity']
        obj=vp.cylinder(name,center,dims['radius'],dims['depth'],mat,parent)
        cutter=vp.cylinder(name+'_bore_cutter',position,dims['bore_radius'],dims['depth']+.01,mat,parent)
        vp.cut_fixture(obj,cutter)
    elif shape=="involute_gear":
        from experiments.mechanism_corpus.gear_geometry import mesh as gear_mesh
        vertices,faces=gear_mesh(dims)
        mesh=bpy.data.meshes.new(name+"_mesh");mesh.from_pydata(vertices,[],faces);mesh.update()
        obj=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(obj)
        obj.parent=parent;obj.location=position;mesh.materials.append(mat)
        for face in mesh.polygons:face.use_smooth=len(face.vertices)==4
    else:raise ValueError("Unimplemented Blender shape: "+shape)
    if "rotation_matrix" in item:
        rotation=Matrix(item["rotation_matrix"])@obj.rotation_euler.to_matrix()
        obj.rotation_euler=rotation.to_euler()
    obj["part_id"]=name;obj["rigid_body_owner"]=item["body"]
    obj["mass_kg"]=item["mass_kg"];obj["material_density_kg_m3"]=item["density_kg_m3"]
    obj["physics_role"]="Included nominal physical solid; ownership avoids double counting"
    return obj


def read_poses(path):
    with np.load(path,allow_pickle=False) as data:return {key:data[key] for key in data.files}


def apply_poses(parents,poses,frame):
    names=list(poses["body_names"])
    for name,parent in parents.items():
        if name=="ground":continue
        index=names.index(name)
        parent.location=poses["body_position_m"][frame,index]
        parent.rotation_mode="QUATERNION"
        parent.rotation_quaternion=Quaternion(poses["body_quaternion_wxyz"][frame,index])


def camera_record(camera,scene):
    R=np.diag([1.,-1.,-1.])@np.asarray(camera.matrix_world.to_3x3()).T
    translation=-R@np.asarray(camera.location)
    scale=camera.data.ortho_scale
    fx=scene.render.resolution_x/scale
    fy=fx
    K=np.array([[fx,0,scene.render.resolution_x/2],[0,fy,scene.render.resolution_y/2],[0,0,1.]])
    return {"projection":"orthographic","scale_m":scale,"K_affine_pixels":K.tolist(),
            "R_world_to_camera":R.tolist(),"t_world_to_camera_m":translation.tolist(),
            "position_m":list(camera.location),"quaternion_wxyz":list(camera.rotation_euler.to_quaternion())}


def scene_setup(assembly,poses_by_condition):
    bpy.ops.object.select_all(action="SELECT");bpy.ops.object.delete(use_global=False)
    bpy.context.preferences.filepaths.save_version=0
    scene=bpy.context.scene;scene.unit_settings.system="METRIC";scene.unit_settings.scale_length=1.
    scene.render.engine="BLENDER_EEVEE";scene.eevee.taa_render_samples=32
    scene.render.resolution_x=640;scene.render.resolution_y=640;scene.render.resolution_percentage=100
    scene.render.fps=30;scene.frame_start=1;scene.frame_end=120
    scene.view_settings.view_transform="AgX"
    scene.view_settings.exposure=-.45
    scene.world.use_nodes=True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value=(.72,.76,.81,1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value=.35
    vp.SEGMENTS=96;vp.BEVEL_SEGMENTS=3
    materials={key:vp.material(key,properties["rgb"],properties["metallic"],properties["roughness"],brushed=key!="frame")
               for key,properties in MATERIALS.items()}
    parents={}
    for name in ["ground",*assembly["bodies"]]:
        parent=bpy.data.objects.new("body_"+name,None);scene.collection.objects.link(parent);parents[name]=parent
        if name in assembly["bodies"]:
            parent["physical_inertia"]=json.dumps(assembly["bodies"][name])
    parts=[create_part(item,parents,materials) for item in assembly["parts"]]
    c=assembly["config"];d=c.get("ground_length_m",0.);h=c.get("pivot_height_m",c.get("platform_height_m",.4))
    presentation=assembly.get("presentation",{})
    for body,code,i in presentation.get("link_marks",[["crank","AB",0],["coupler","BC",1],["rocker","DC",2]]):
        vp.stamp("mark_"+body,code+" "+str(round(c["lengths_m"][i]*1000)),
                 [c["lengths_m"][i]/2,c["visual_and_com_y_m"][i]-c["bar_depth_m"]/2-.00005,0.],
                 .0055,materials["steel"],parents[body])
    base=next((item for item in assembly["parts"] if item["name"]=="machined_base"),None)
    default_id_position=[d/2,base["position_m"][1]-base["dimensions_m"]["size"][1]/2-.00005 if base else -.0635,.025]
    id_mark=vp.stamp("assembly_id",assembly["assembly_id"],presentation.get("id_mark_position_m",default_id_position),
                     .0058,materials["graphite"],parents["ground"])
    if presentation.get('id_mark_rotation_matrix'):
        id_mark.rotation_euler=Matrix(presentation['id_mark_rotation_matrix']).to_euler()
    floor=vp.material("studio_floor",(.62,.66,.71),0,.65)
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.0002))
    ground=bpy.context.object;ground.name="studio_floor";ground.data.materials.append(floor)
    for name,position,power,size in [
        ("key",(-.15,-.65,1.15),42,.65),("fill",(.75,-.35,.75),20,.6),("rim",(.3,.4,1.1),36,.5)]:
        bpy.ops.object.light_add(type="AREA",location=position)
        light=bpy.context.object;light.name=name;light.data.energy=power;light.data.shape="DISK";light.data.size=size
        vp.look_at(light,(d/2,0,h/2))
    # Bound all moving physical parts over all three recorded conditions,
    # then use the same static-camera calibration for each condition.
    bounds_by_condition={}
    for condition,poses in poses_by_condition.items():
        frames=[]
        for frame in range(len(poses["time"])):
            apply_poses(parents,poses,frame);bpy.context.view_layer.update()
            frames.append([[np.asarray(obj.matrix_world@Vector(corner)) for corner in obj.bound_box] for obj in parts])
        bounds_by_condition[condition]=np.asarray(frames)
    bounds=np.concatenate([value.reshape(-1,3) for value in bounds_by_condition.values()])
    lower,upper=bounds.min(axis=0),bounds.max(axis=0);center=(lower+upper)/2
    extent=float(np.linalg.norm(upper-lower))
    cameras={}
    views={"front":[0,-1,0],"side":[1,0,0],"top":[0,0,1],"isometric":[1,-1,.8],"tracking_closeup":[.45,-1,.35]}
    for name,offset in views.items():
        direction=np.asarray(offset,dtype=float);direction/=np.linalg.norm(direction)
        target=center.copy()
        bpy.ops.object.camera_add(location=target+direction*max(2.,extent*3))
        camera=bpy.context.object;camera.name="camera_"+name;vp.look_at(camera,target)
        camera.data.type="ORTHO";camera.data.clip_start=.001;camera.data.clip_end=100
        camera.data.ortho_scale=extent*1.2
        if name!="tracking_closeup":
            R=np.asarray(camera.rotation_euler.to_matrix()).T
            projected=(bounds-target)@R.T
            span=np.ptp(projected[:,:2],axis=0)
            camera.data.ortho_scale=float(max(span)*1.13)
            shift=np.r_[(projected[:,:2].max(axis=0)+projected[:,:2].min(axis=0))/2,0.]
            camera.location+=camera.rotation_euler.to_matrix()@Vector(shift)
        else:
            focus_bodies=presentation.get('focus_bodies',[presentation.get('focus_body','coupler')])
            selected=[i for i,obj in enumerate(parts) if obj['rigid_body_owner'] in focus_bodies]
            focus_bounds=np.concatenate([value[:,selected].reshape(-1,3) for value in bounds_by_condition.values()])
            R=np.asarray(camera.rotation_euler.to_matrix()).T
            projected=(focus_bounds-target)@R.T
            camera.data.ortho_scale=float(max(np.ptp(projected[:,:2],axis=0))*1.18)
            shift=np.r_[(projected[:,:2].max(axis=0)+projected[:,:2].min(axis=0))/2,0.]
            camera.location+=camera.rotation_euler.to_matrix()@Vector(shift)
        cameras[name]=camera
    scene["assembly_id"]=assembly["assembly_id"]
    scene["physics_source"]="MuJoCo raw body transforms; no IK/pose repair in Blender"
    return scene,parents,parts,cameras,bounds_by_condition


def framing_check(bounds,calibration,parts,view,focus_body="coupler"):
    focus_bodies=[focus_body] if isinstance(focus_body,str) else focus_body
    selected=[i for i,obj in enumerate(parts) if view!="tracking_closeup" or obj["rigid_body_owner"] in focus_bodies]
    if not selected:raise ValueError("No objects selected for framing check")
    margin=1.
    for points,record in zip(bounds[:,selected],calibration):
        xyz=points.reshape(-1,3)@np.asarray(record["R_world_to_camera"]).T+record["t_world_to_camera_m"]
        if xyz[:,2].min()<=.001:raise ValueError("Object behind camera")
        normalized=xyz[:,:2]/record["scale_m"]+.5
        margin=min(margin,float(normalized.min()),float(1-normalized.max()))
    if margin<.005:raise ValueError("Required parts leave camera frame: "+view+" "+str(margin))
    return {"passed":True,"frames":len(calibration),"minimum_fractional_margin":margin,
            "scope":"focus bodies: "+', '.join(focus_bodies) if view=="tracking_closeup" else "all physical parts"}


def set_animation(scene,parents,poses):
    for parent in parents.values():parent.animation_data_clear()
    for frame in range(len(poses["time"])):
        apply_poses(parents,poses,frame)
        for name,parent in parents.items():
            if name=="ground":continue
            parent.keyframe_insert("location",frame=frame+1)
            parent.keyframe_insert("rotation_quaternion",frame=frame+1)
    scene.frame_start=1;scene.frame_end=len(poses["time"])
    max_position=max_rotation=0.
    names=list(poses["body_names"])
    for frame in range(len(poses["time"])):
        scene.frame_set(frame+1);bpy.context.view_layer.update()
        for name,parent in parents.items():
            if name=="ground":continue
            index=names.index(name)
            max_position=max(max_position,float(np.linalg.norm(np.asarray(parent.location)-poses["body_position_m"][frame,index])))
            current=np.asarray(parent.rotation_quaternion)
            target=poses["body_quaternion_wxyz"][frame,index]
            max_rotation=max(max_rotation,float(min(np.linalg.norm(current-target),np.linalg.norm(current+target))))
    if max_position>3e-7 or max_rotation>3e-6:raise RuntimeError("Blender/MuJoCo pose disagreement")
    return {"maximum_position_error_m":max_position,"maximum_quaternion_error":max_rotation,"frames":len(poses["time"])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assembly-dir",type=Path,required=True)
    parser.add_argument("--preview-only",action="store_true")
    parser.add_argument("--conditions",nargs="+",default=["low_speed","rated","limit_or_load"])
    parser.add_argument("--views",nargs="+",default=["front","side","top","isometric","tracking_closeup"])
    parser.add_argument("--benchmark-frames",type=int,default=0)
    parser.add_argument("--resume",action="store_true")
    argv=sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
    args=parser.parse_args(argv);folder=args.assembly_dir.resolve()
    assembly=json.loads((folder/"assembly.json").read_text())
    condition_names=["low_speed","rated","limit_or_load"]
    view_names=["front","side","top","isometric","tracking_closeup"]
    poses_by_condition={}
    for condition in condition_names:
        report=json.loads((folder/"conditions"/condition/"physics_validation.json").read_text())
        if not report["passed"] or report.get('assembly_physical_digest')!=assembly['physical_digest']:
            raise RuntimeError("Unvalidated or stale physics")
        poses_by_condition[condition]=read_poses(folder/"conditions"/condition/"render_poses.npz")
    scene,parents,parts,cameras,bounds_by_condition=scene_setup(assembly,poses_by_condition)
    output=folder/"videos";output.mkdir(exist_ok=True)
    inspections=folder/"previews";inspections.mkdir(exist_ok=True)
    renderer_sha=RENDERER_SOURCE_SHA256
    snapshot=ROOT/"output/mechanism_corpus/source_versions"/(renderer_sha+".py")
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    if not snapshot.exists():snapshot.write_bytes(RENDERER_SOURCE_BYTES)
    artifact_rows=[];start=time.monotonic()
    for condition in args.conditions:
        poses=poses_by_condition[condition];pose_check=set_animation(scene,parents,poses)
        physical=json.loads((folder/"conditions"/condition/"physical_parameters.json").read_text())
        for name,parent in parents.items():
            if name in physical["bodies"]:parent["physical_inertia"]=json.dumps(physical["bodies"][name])
        mass_lookup={p["name"]:p["mass_kg"] for p in physical["parts"]}
        for obj in parts:obj["mass_kg"]=mass_lookup[obj["part_id"]]
        scene["operating_condition"]=condition
        for view in args.views:
            number=condition_names.index(condition)*5+view_names.index(view)+1
            identity=assembly["assembly_id"]+"-V%02d"%number
            path=output/(identity+".mp4")
            previous_path=path.with_suffix(".json")
            if args.resume and not args.preview_only and path.exists() and previous_path.exists():
                previous=json.loads(previous_path.read_text())
                references_current=all(previous.get(key)==hashlib.sha256((folder/'conditions'/condition/name).read_bytes()).hexdigest()
                                       for key,name in [('physics_trajectory_sha256','physics.npz'),('commands_sha256','commands.npz')])
                if (not previous.get("benchmark_only") and previous.get("assembly_physical_digest")==assembly["physical_digest"]
                        and references_current
                        and previous.get("frames")==120 and previous.get("framing_check",{}).get("passed")
                        and (view!="tracking_closeup" or previous.get("camera_policy_revision")=="fixed_workspace_closeup_v1")
                        and path.with_suffix(".inputs.json").exists()
                        and previous.get("sha256")==hashlib.sha256(path.read_bytes()).hexdigest()):
                    artifact_rows.append(previous)
                    print(json.dumps({"reused_verified_hash":identity}),flush=True)
                    continue
            camera=cameras[view];camera.animation_data_clear();scene.camera=camera
            calibration=[]
            for frame in range(len(poses["time"])):
                scene.frame_set(frame+1);bpy.context.view_layer.update()
                calibration.append(camera_record(camera,scene))
            framing=framing_check(bounds_by_condition[condition],calibration,parts,view,
                                  assembly.get("presentation",{}).get("focus_bodies",assembly.get("presentation",{}).get("focus_body","coupler")))
            scene.frame_set(1)
            if condition=="rated" and view=="isometric":
                bpy.ops.wm.save_as_mainfile(filepath=str(folder/"assembly.blend"),compress=True)
            scene.render.image_settings.media_type="IMAGE"
            scene.render.image_settings.file_format="PNG"
            scene.render.filepath=str(inspections/(identity+".png"))
            bpy.ops.render.render(write_still=True)
            if args.preview_only:continue
            scene.render.image_settings.media_type="VIDEO"
            scene.render.image_settings.file_format="FFMPEG"
            scene.render.ffmpeg.format="MPEG4";scene.render.ffmpeg.codec="H264"
            scene.render.ffmpeg.constant_rate_factor="CUSTOM";scene.render.ffmpeg.custom_constant_rate_factor=18
            scene.render.ffmpeg.use_lossless_output=False;scene.render.ffmpeg.ffmpeg_preset="GOOD"
            scene.render.ffmpeg.gopsize=30
            scene.render.filepath=str(path)
            scene.frame_end=args.benchmark_frames or len(poses["time"])
            bpy.ops.render.render(animation=True)
            if not path.exists():
                candidates=list(output.glob(identity+"*.mp4"))
                if len(candidates)!=1:
                    raise RuntimeError("Cannot identify Blender movie output: "+identity)
                candidates[0].replace(path)
            manifest={"video_id":identity,"assembly_id":assembly["assembly_id"],"condition":condition,"view":view,
                      "frames":scene.frame_end,"fps":30,"width":640,"height":640,
                      "frame_times_s":poses["time"][:scene.frame_end].tolist(),
                      "camera_frames":calibration[:scene.frame_end],"pose_check":pose_check,
                      "framing_check":framing,"path_base":"assembly directory",
                      "physical_trajectory":"conditions/"+condition+"/physics.npz",
                      "physics_trajectory_sha256":hashlib.sha256((folder/"conditions"/condition/"physics.npz").read_bytes()).hexdigest(),
                      "commands_sha256":hashlib.sha256((folder/"conditions"/condition/"commands.npz").read_bytes()).hexdigest(),
                      "commands":"conditions/"+condition+"/commands.npz","blender_version":bpy.app.version_string,
                      "engine":scene.render.engine,"samples":scene.eevee.taa_render_samples,
                      "codec":"h264","crf":18,"assembly_physical_digest":assembly["physical_digest"],
                      "renderer_source_sha256":renderer_sha,
                      "camera_policy_revision":"fixed_workspace_closeup_v1",
                      "camera_policy":"Fixed extrinsics throughout every clip; view 5 is a fixed workspace closeup.",
                      "benchmark_only":bool(args.benchmark_frames),
                      "sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
            write_json(path.with_suffix(".json"),manifest);artifact_rows.append(manifest)
            public={"video_id":identity,"description":assembly.get("presentation",{}).get("description",
                    "Rigid closed-chain four-bar with a driven crank, coupler, rocker and fixed supports."),
                    "condition":condition,"view":view,"video_file":path.name,
                    "path_base":"directory containing this sidecar",
                    "observed_prefix_frames":3,
                    "model_observation_policy":"Use a selected video prefix; this sidecar contains no future object states or measured torques.",
                    "camera_intrinsics":{key:value for key,value in calibration[0].items()
                                         if key in ["projection","scale_m","K_affine_pixels"]},
                    "observed_camera_frames":calibration[:3],
                    "camera_motion_policy":"Fixed camera framing the moving-part workspace" if view=="tracking_closeup" else "Fixed camera",
                    "commanded_actions":"../conditions/"+condition+"/commands.npz",
                    "controller_calibration":assembly["config"]["actuator"],
                    "actuator_effort_policy":"Realized feedback effort is reference data, not a future action input.",
                    "full_camera_and_physical_reference":"Evaluator manifest only"}
            if "external_force_xz_N" in assembly["config"]:
                public["known_external_force_world_xz_N"]=assembly["config"]["external_force_xz_N"]
            controller_file=folder/"conditions"/condition/"controller_inputs.json"
            if controller_file.exists():public.update(json.loads(controller_file.read_text()))
            write_json(path.with_suffix(".inputs.json"),public)
            print(json.dumps({"rendered":identity,"frames":scene.frame_end,"elapsed_seconds":time.monotonic()-start}),flush=True)
    # Always leave a reproducible nominal scene, including after a partial
    # render or a resume where all requested video files were reused.
    set_animation(scene,parents,poses_by_condition['rated'])
    for name,parent in parents.items():
        if name in assembly['bodies']:parent['physical_inertia']=json.dumps(assembly['bodies'][name])
    nominal_mass={part['name']:part['mass_kg'] for part in assembly['parts']}
    for obj in parts:obj['mass_kg']=nominal_mass[obj['part_id']]
    scene['operating_condition']='rated';scene.camera=cameras['isometric'];scene.frame_set(1)
    bpy.ops.wm.save_as_mainfile(filepath=str(folder/'assembly.blend'),compress=True)
    summary_name="preview_manifest.json" if args.preview_only else "render_manifest.json"
    if not args.preview_only:
        artifact_rows=[]
        for metadata_file in sorted(output.glob('*-V[0-9][0-9].json')):
            row=json.loads(metadata_file.read_text())
            if row.get('assembly_physical_digest')==assembly['physical_digest'] and metadata_file.with_suffix('.mp4').is_file():
                artifact_rows.append(row)
    write_json(folder/summary_name,{"rendered_videos":len(artifact_rows),"rows":artifact_rows,
               "preview_only":args.preview_only,"part_count":len(parts),"seconds":time.monotonic()-start,
               "assembly_id":assembly["assembly_id"]})


if __name__=="__main__":
    main()
