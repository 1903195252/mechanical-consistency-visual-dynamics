"""Organize editable scenes and record appearance/physics provenance."""

import hashlib
import json
from pathlib import Path

import bpy
import numpy as np


def organize(scene):
    names=['01_Fixed_frame','02_Moving_links','03_Joints_and_hardware',
           '04_Motor','05_Markings','06_Cameras','07_Studio']
    groups={}
    for name in names:
        groups[name]=bpy.data.collections.new(name)
        scene.collection.children.link(groups[name])
    for obj in list(scene.objects):
        ancestry=[];parent=obj
        while parent:
            ancestry.append(parent.name);parent=parent.parent
        if obj.type=='CAMERA':group='06_Cameras'
        elif obj.type=='LIGHT' or obj.name in ['Floor','Studio cyclorama']:group='07_Studio'
        elif obj.type=='FONT' or 'identification' in obj.name.lower():group='05_Markings'
        elif 'motor' in obj.name.lower():group='04_Motor'
        elif any(name.startswith('Joint ') for name in ancestry) or 'connector' in obj.get('physics_role','') or obj.get('asset_type')=='socket_head':group='03_Joints_and_hardware'
        elif any(name.endswith('rigid-body pose') for name in ancestry):group='02_Moving_links'
        else:group='01_Fixed_frame'
        for collection in list(obj.users_collection):collection.objects.unlink(obj)
        groups[group].objects.link(obj)


def export(scene,source,output,config,trajectory,links,style_name,appearance,appearance_path):
    # Compare animation to raw recorded states at every frame, never to projected IK.
    max_position,max_angle=0.,0.
    for i,q in enumerate(trajectory['q']):
        scene.frame_set(i+1)
        A=np.array([0,0,config['pivot_height_m']])
        B=A+config['lengths_m'][0]*np.array([np.cos(q[0]),0,np.sin(q[0])])
        D=A+np.array([config['ground_length_m'],0,0])
        for obj,origin,angle in zip(links,[A,B,D],q):
            max_position=max(max_position,float(np.linalg.norm(np.asarray(obj.location)-origin)))
            max_angle=max(max_angle,abs(float(obj.rotation_euler.y)+angle))
    if max_position>2e-7 or max_angle>3e-6:
        raise RuntimeError('Blender poses differ from the recorded simulation')
    scene.frame_set(1);bpy.context.view_layer.update()
    parts=[]
    for obj in sorted(scene.objects,key=lambda obj:obj.name):
        if obj.type in ('MESH','EMPTY','FONT'):
            parts.append({'name':obj.name,'type':obj.type,'parent':obj.parent.name if obj.parent else None,
                          'collection':obj.users_collection[0].name,'physics_role':obj.get('physics_role','pose transform or presentation geometry'),
                          'mesh_asset':obj.data.name if obj.type=='MESH' else None,
                          'materials':[slot.material.name for slot in obj.material_slots if slot.material],
                          'local_position_m':list(obj.location)})
    inputs={name:hashlib.sha256((source/name).read_bytes()).hexdigest()
            for name in ['physical_metadata.json','video_trajectory.npz','ground_truth.npz','actions.npz']}
    fasteners=[obj for obj in scene.objects if obj.get('asset_type')=='socket_head']
    manifest={'schema_version':1,'style':style_name,'appearance':appearance,
              'builder_source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                       for name in ['blender_scene.py','visual_parts.py','scene_manifest.py','geometry.py']},
              'appearance_file_sha256':hashlib.sha256(Path(appearance_path).read_bytes()).hexdigest(),
              'source_directory':str(source),'physics_input_sha256':inputs,
              'asset_stats':{'socket_head_instances':len(fasteners),'socket_head_unique_meshes':len({obj.data.name for obj in fasteners}),
                             'object_count':len(scene.objects)},
              'checks':{'animation_matches_recorded_states':True,'frame_count':len(trajectory['time']),
                        'pose_position_max_error_m':max_position,'pose_angle_max_error_rad':max_angle},
              'physical_scope':'Moving plate solids and cuboid payload retain the input mass/inertia. Fixtures are fixed; connector refinements are massless idealized attachments.',
              'cameras':[{'name':obj.name,'role':obj.get('role','calibrated dataset camera'),
                          'type':obj.data.type,'position_m':list(obj.location),
                          'rotation_euler_rad':list(obj.rotation_euler),'ortho_scale_m':obj.data.ortho_scale if obj.data.type=='ORTHO' else None}
                         for obj in scene.objects if obj.type=='CAMERA'],
              'parts':parts}
    (output/'appearance_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    scene['appearance_style']=style_name;scene['appearance_manifest']='appearance_manifest.json'
    return manifest
