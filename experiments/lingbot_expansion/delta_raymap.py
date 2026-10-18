"""Blender helper: visible CAD surface coordinates at the registered reset.

Only static CAD, public camera calibration and nominal body transforms are read.
No simulated future pose, command or dynamics enters the map.
"""
import json
import sys
from pathlib import Path
import bpy
import numpy as np
from mathutils import Vector, Quaternion


def run(path):
    task=json.loads(Path(path).read_text())
    bpy.ops.wm.open_mainfile(filepath=task['blend'])
    for name,pose in task['nominal_bodies'].items():
        obj=bpy.data.objects.get('body_'+name)
        if obj:
            # Saved render scenes may contain replay animation. Reset it first.
            obj.animation_data_clear()
            obj.location=pose['position'];obj.rotation_mode='QUATERNION'
            obj.rotation_quaternion=Quaternion(pose['quaternion'])
    bpy.context.view_layer.update()
    graph=bpy.context.evaluated_depsgraph_get();scene=bpy.context.scene
    camera=task['camera'];R=np.array(camera['R_world_to_camera']);t=np.array(camera['t_world_to_camera_m'])
    width=task['width'];scale=camera['scale_m'];direction=Vector(R.T@np.array([0.,0.,1.]))
    bodies=['platform','upper_0','upper_1','upper_2']
    owner=np.full((width,width),-1,dtype=np.int8);local=np.full((width,width,3),np.nan,dtype=np.float32)
    for iy in range(width):
        for ix in range(width):
            origin=R.T@(np.array([(ix+.5-width/2)*scale/width,(iy+.5-width/2)*scale/width,-10.])-t)
            hit,point,normal,index,obj,matrix=scene.ray_cast(graph,Vector(origin),direction,distance=30.)
            if not hit:continue
            body=obj.get('rigid_body_owner','')
            if body not in bodies:continue
            owner[iy,ix]=bodies.index(body)
            parent=bpy.data.objects['body_'+body]
            local[iy,ix]=parent.matrix_world.inverted()@point
    np.savez_compressed(task['output'],owner=owner,local=local)
    print(json.dumps(dict(stage='CAD raymap',output=task['output'],pixels={b:int((owner==i).sum()) for i,b in enumerate(bodies)})),flush=True)


if __name__=='__main__':run(sys.argv[sys.argv.index('--')+1])
