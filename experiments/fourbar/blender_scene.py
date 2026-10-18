"""Build an editable Blender scene driven exclusively by recorded MuJoCo states."""

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from geometry import link_mesh
import visual_parts
import scene_manifest
from visual_parts import material,finish,box,cylinder,ring,socket_head,stamp,look_at,cut_fixture,fit_assembly_camera


def joint_assembly(name,front,back,steel,bronze,dark,shaft_radius):
    root=bpy.data.objects.new('Joint '+name+' assembly',None)
    bpy.context.scene.collection.objects.link(root)
    root['physics_role']='ideal revolute connector; hardware mass excluded'
    # Terminate the shoulder shaft at the cap's rear face; it must not fill
    # the blind socket cavity. Contact faces of caps/washers/nuts coincide.
    shaft_start,shaft_end=front-.002,back+.0085
    cylinder(name+' precision shoulder axle',(0,(shaft_start+shaft_end)/2,0),shaft_radius,shaft_end-shaft_start,steel,root)
    for label,y in [('front',front-.001),('rear',back+.001)]:
        ring(name+' '+label+' washer',(0,y,0),.009,shaft_radius,.002,steel,root)
        ring(name+' '+label+' turned washer rim',(0,y-.0008,0),.0087,.0077,.00035,steel,root)
    socket_head(name+' socket cap',(0,front-.0045,0),.006,.005,steel,root)
    cylinder(name+' retaining nut',(0,back+.0045,0),.006,.005,steel,root,vertices=6)
    ring(name+' rear locking collar',(0,back+.0075,0),.0057,shaft_radius,.001,bronze,root)
    return root


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',default='nominal')
    parser.add_argument('--render',action='store_true')
    parser.add_argument('--preview-only',action='store_true')
    parser.add_argument('--input-dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--appearance',type=Path,default=Path(__file__).with_name('appearance.json'))
    parser.add_argument('--style',default=None)
    parser.add_argument('--beauty',action='store_true',help='Also render a high-resolution Cycles assembly view')
    arguments=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    args=parser.parse_args(arguments)
    bpy.context.preferences.filepaths.save_version=0
    root=Path(__file__).resolve().parents[2]
    source=(args.input_dir or root/'output'/'fourbar'/args.case).resolve()
    output=(args.output_dir or source).resolve();output.mkdir(parents=True,exist_ok=True)
    config=json.loads((source/'physical_metadata.json').read_text())
    trajectory=np.load(source/'video_trajectory.npz')
    appearance=json.loads(args.appearance.read_text())
    style_name=args.style or appearance['default_style']
    style=appearance['styles'][style_name]
    visual_parts.SEGMENTS=appearance['geometry']['circular_segments']
    visual_parts.BEVEL_SEGMENTS=appearance['geometry']['bevel_segments']
    shaft_radius=config['joint_bore_radius_m']*appearance['geometry']['shaft_to_bore_ratio']
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    scene=bpy.context.scene
    scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
    scene['physics_source']='MuJoCo torque-driven simulation; Blender only replays exported states'
    scene['physical_metadata']=json.dumps(config)
    scene['validation_report']='validation.json (independent Lagrangian and power-balance checks)'
    engines={item.identifier for item in scene.render.bl_rna.properties['engine'].enum_items}
    if 'BLENDER_EEVEE' in engines:scene.render.engine='BLENDER_EEVEE'
    elif 'BLENDER_EEVEE_NEXT' in engines:scene.render.engine='BLENDER_EEVEE_NEXT'
    else:scene.render.engine='CYCLES';scene.cycles.samples=24
    scene.render.resolution_x=config['simulation']['width'];scene.render.resolution_y=config['simulation']['height']
    scene.render.resolution_percentage=100;scene.render.fps=config['simulation']['video_fps']
    scene.frame_start=1;scene.frame_end=len(trajectory['time'])
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(*appearance['studio']['background_rgb'],1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=appearance['studio']['world_strength']
    scene.view_settings.view_transform='AgX'
    scene.view_settings.exposure=appearance['studio'].get('exposure',0.)
    blue,orange,green=[material(name+' anodized aluminum',rgb,.72,.3,brushed=True,coat=.12)
                       for name,rgb in zip(['Crank','Coupler','Rocker'],style['link_rgb'])]
    steel=material('Turned stainless steel',(.48,.54,.6),.9,.22,brushed=True)
    bronze=material('Bronze bearing sleeves',(.42,.25,.08),.82,.28,brushed=True)
    base=material('Coated frame',style['frame_rgb'],.4,.33,coat=.12)
    aluminum=material('Machined aluminum supports',style['aluminum_rgb'],.82,.3,brushed=True)
    dark=material('Black oxide fasteners / motor',(.022,.028,.035),.5,.3)
    lettering=material('Laser etched marks',style['lettering_rgb'],.2,.45)
    floor=material('Studio floor',style['floor_rgb'],0,.65)
    d,h=config['ground_length_m'],config['pivot_height_m']
    box('Fixed ground tie beam',(d/2,.08,h),(d+.075,.04,.034),base)
    box('Machined base plate',(d/2,.055,.022),(d+.2,.24,.024),aluminum,bevel=.002)
    box('Flush graphite deck insert',(d/2,.065,.0345),(max(.08,d-.12),.14,.001),base,bevel=.0004)
    box('Base front identification plate',(d/2,-.0655,.022),(.18,.001,.013),base,bevel=.0004)
    stamp('Base engraved designation','PLANAR 4R  /  DYNAMICS RIG',(d/2,-.0661,.022),.0045,lettering)
    # Anchoring hardware has a consistent pitch and head style.
    for x in [-.07,d+.07]:
        for y in [-.035,.145]:
            washer=ring('Base anchor washer',(x,y,.035),.008,.003,.0015,steel)
            washer.rotation_euler=(math.pi/2,0,0)
            socket_head('Base anchor M6',(x,y,.0385),.006,.006,dark,axis='Z')
    for x in [-.06,d+.06]:
        box('Isolation foot',(x,.055,.008),(.04,.16,.012),dark,bevel=.003)
    for i,x in enumerate([0,d]):
        name=['A','D'][i]
        box(name+' bolted support foot',(x,.075,.043),(.105,.115,.018),aluminum)
        column=box(name+' support column',(x,.077,(h+.044)/2),(.052,.048,h-.044),base)
        for dx in [-.015,0,.015]:
            cutter=box('Column flute cutter',(x+dx,.0525,(h+.045)/2),(.0032,.006,h-.12),base,bevel=0)
            cut_fixture(column,cutter)
        block=box(name+' machined bearing block',(x,.066,h),(.07,.052,.07),aluminum,bevel=.002)
        cutter=cylinder('Bearing seat cutter',(x,.066,h),.0202,.060,steel)
        cut_fixture(block,cutter)
        for dx in [-.027,.027]:
            cut_fixture(block,box('Split bearing seam cutter',(x+dx,.0397,h),(.016,.002,.0005),base,bevel=0))
        ring(name+' fixed bearing outer race',(x,.066,h),.0202,.016,.052,steel)
        ring(name+' fixed bearing inner race',(x,.066,h),.009,shaft_radius,.052,steel)
        ring(name+' fixed bearing face race',(x,.038,h),.0202,.017,.004,steel)
        ring(name+' fixed bearing seal',(x,.037,h),.017,.0085,.003,dark)
        ring(name+' bearing face inner ring',(x,.036,h),.0085,shaft_radius,.003,steel)
        for dx in [-.024,.024]:
            for dz in [-.024,.024]:
                socket_head(name+' bearing block screw',(x+dx,.037,h+dz),.0035,.003,dark)
        for dx in [-.038,.038]:
            for y in [.035,.115]:
                socket_head(name+' support foot screw',(x+dx,y,.055),.0045,.005,dark,axis='Z')
        box(name+' pillar identification tab',(x,.052,.115),(.022,.002,.018),aluminum,bevel=.0005)
        stamp(name+' support marking',name,(x,.0509,.115),.01,dark)
    box('Frame identification plate',(d/2,.058,h),(.115,.003,.026),aluminum)
    stamp('Frame marking','4R  /  '+str(round(d*1000))+' mm',(d/2,.056,h),.008,dark)
    for dx in [-.049,.049]:
        socket_head('Identification plate screw',(d/2+dx,.055,h),.0017,.0014,dark)
    bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,.01));bpy.context.object.name='Floor';bpy.context.object.data.materials.append(floor)
    # A curved studio backdrop removes the artificial floor/world seam.
    backdrop=[]
    for angle in np.linspace(0,math.pi/2,25):
        y,z=1.+.5*math.sin(angle),.01+.5*(1-math.cos(angle))
        backdrop.extend([(-3,y,z),(3,y,z)])
    backdrop.extend([(-3,1.5,3),(3,1.5,3)])
    mesh=bpy.data.meshes.new('Continuous studio backdrop')
    mesh.from_pydata(backdrop,[],[(2*i,2*i+2,2*i+3,2*i+1) for i in range(len(backdrop)//2-1)])
    mesh.update();obj=bpy.data.objects.new('Studio cyclorama',mesh);scene.collection.objects.link(obj)
    mesh.materials.append(floor)
    for face in mesh.polygons:face.use_smooth=True
    cylinder('A motor housing',(0,.132,h),.038,.078,dark)
    cylinder('A motor front flange',(0,.093,h),.044,.008,aluminum)
    cylinder('A motor rear cover',(0,.173,h),.035,.006,aluminum)
    for angle in np.linspace(0,2*math.pi,9)[:-1]:
        x,z=.039*math.cos(angle),h+.039*math.sin(angle)
        socket_head('Motor flange screw',(x,.087,z),.0025,.003,dark)
    for y in [.106,.117,.128,.139,.15,.161]:
        cylinder('Motor cooling rib',(0,y,h),.040,.002,dark)
    links=[]
    for i,(name,mat) in enumerate(zip(['Crank','Coupler','Rocker'],[blue,orange,green])):
        pivot=bpy.data.objects.new(name+' rigid-body pose',None);scene.collection.objects.link(pivot)
        pivot['mass_kg']=config['moving_bar_mass_kg'][i]
        pivot['COM_local_m']=[config['lengths_m'][i]/2,config['visual_and_com_y_m'][i],0]
        pivot['diagonal_inertia_kg_m2']=config['moving_bar_diagonal_inertia_kg_m2'][i]
        pivot['joint_friction_model']='see physical_metadata.json: all four joints including closure C'
        length,layer=config['lengths_m'][i],config['visual_and_com_y_m'][i]
        vertices,faces=link_mesh(length,config['bar_height_m'],config['bar_depth_m'],config['joint_bore_radius_m'])
        mesh=bpy.data.meshes.new(name+' drilled capsule plate');mesh.from_pydata(vertices,[],faces);mesh.update()
        for face in mesh.polygons:face.use_smooth=abs(face.normal.y)<.1
        obj=bpy.data.objects.new(name+' machined link',mesh);scene.collection.objects.link(obj)
        obj.parent=pivot;obj.location=(0,layer,0);mesh.materials.append(mat)
        finish(obj,.00025)
        obj['physics_role']='uniform drilled capsule solid; analytic moments in physical_metadata.json'
        obj['pivot_spacing_m']=length;obj['overall_length_m']=length+config['bar_height_m']
        for endpoint in [0,length]:
            ring(name+' bronze sleeve',(endpoint,layer,0),config['joint_bore_radius_m'],shaft_radius,config['bar_depth_m'],bronze,pivot)
            ring(name+' sleeve flange',(endpoint,layer-config['bar_depth_m']/2-.0005,0),.0075,shaft_radius,.001,bronze,pivot)
        stamp(name+' engraved part ID',['AB','BC','DC'][i]+'  '+str(round(length*1000)),
              (length/2,layer-config['bar_depth_m']/2-.00006,0),.0065,
              dark if np.dot(style['link_rgb'][i],[.2126,.7152,.0722])>.45 else lettering,pivot)
        links.append(pivot)
    # A low-profile satin block sits flush on a thin mounting plate.
    size=np.asarray(config['payload_size_m']);along=config['payload_along_coupler_m']
    normal=config['payload_normal_offset_m'];depth=config['payload_mount_plate_thickness_m']
    link_face=config['visual_and_com_y_m'][1]-config['bar_depth_m']/2
    payload_rear=config['payload_y_m']+size[1]/2
    if abs(payload_rear-(link_face-depth))>1e-9:
        raise ValueError('Payload rear face must meet the mounting plate')
    payload_mat=material('Satin graphite removable weight',(.045,.058,.07),.65,.45,brushed=True)
    payload=box('Rigid payload',(along,config['payload_y_m'],normal),tuple(size),payload_mat,links[1],bevel=.0007)
    payload['physics_role']='uniform cuboid rigid payload; nominal solid moments include the full box'
    payload['mass_kg']=config['payload_mass_kg']
    payload['diagonal_inertia_kg_m2']=config['payload_diagonal_inertia_kg_m2']
    plate=box('Payload mounting plate',(along,link_face-depth/2,normal),
              (size[0]+.012,depth,size[2]+.002),aluminum,links[1],bevel=.0003)
    plate['physics_role']='massless visualization of a rigid payload attachment'
    front=config['payload_y_m']-size[1]/2
    for dx in [-size[0]/2+.0045,size[0]/2-.0045]:
        socket_head('Payload mount screw',(along+dx,front-.0013,normal),.0019,.0026,steel,links[1])
    stamp('Payload engraved ID','LOAD',(along,front-.00002,normal),.0045,lettering,links[1])
    # Axle stacks terminate at the correct faces of the layered plates.
    layers=np.asarray(config['visual_and_com_y_m']);half=config['bar_depth_m']/2
    fronts=[layers[0]-half-.001,min(layers[:2])-half-.001,min(layers[1:])-half-.001,layers[2]-half-.001]
    backs=[.093,max(layers[:2])+half+.001,max(layers[1:])+half+.001,.093]
    pins=[joint_assembly(name,front,back,steel,bronze,dark,shaft_radius)
          for name,front,back in zip(['A','B','C','D'],fronts,backs)]
    for name,i,pair in [('B',1,layers[:2]),('C',2,layers[1:])]:
        low,high=sorted(pair);gap=high-low-config['bar_depth_m']
        if gap<=0:raise ValueError('Link layers must have positive axial spacing')
        ring(name+' axial spacer',(0,(low+high)/2,0),.0065,shaft_radius,gap,steel,pins[i])
    for name,i,layer in [('A',0,layers[0]),('D',3,layers[2])]:
        back_face=layer+half;gap=.038-back_face
        if gap<=0:raise ValueError('Moving plate would intersect the fixed bearing face')
        ring(name+' ground standoff',(0,(back_face+.038)/2,0),.008,shaft_radius,gap,steel,pins[i])
    scene['appearance_model']='Machined round-end plates with true through bores; layered bronze sleeves, shoulder pins, washers and socket caps. Fixtures and connector hardware are idealized fixed/massless components.'
    for index,(time,q) in enumerate(zip(trajectory['time'],trajectory['q'])):
        frame=index+1;l1,l2,l3=config['lengths_m']
        A=np.array([0.,0.,h]);D=np.array([d,0.,h])
        B=A+np.array([l1*np.cos(q[0]),0,l1*np.sin(q[0])])
        C=B+np.array([l2*np.cos(q[1]),0,l2*np.sin(q[1])])
        for pivot,origin,angle in zip(links,[A,B,D],q):
            pivot.location=origin;pivot.rotation_euler=(0,-float(angle),0)
            pivot.keyframe_insert('location',frame=frame);pivot.keyframe_insert('rotation_euler',frame=frame)
        for pin,position in zip(pins,[A,B,C,D]):
            pin.location=position;pin.keyframe_insert('location',frame=frame)
    # Blender 5 uses action slots; keyframes are already sampled at every rendered frame.
    bpy.ops.object.camera_add(location=config['camera']['position_m'])
    camera=bpy.context.object;camera.name='Calibrated main camera';look_at(camera,config['camera']['target_m'])
    camera.data.type='PERSP';camera.data.sensor_fit='VERTICAL';camera.data.sensor_height=24
    camera.data.lens=24/(2*math.tan(math.radians(config['camera']['vertical_fov_deg'])/2))
    scene.camera=camera
    for lighting in appearance['studio']['lights']:
        bpy.ops.object.light_add(type='AREA',location=lighting['position_m'])
        light=bpy.context.object;light.name=lighting['name'];light.data.energy=lighting['power_W']
        light.data.shape='DISK';light.data.size=lighting['size_m'];light.data.color=lighting['color_rgb']
        look_at(light,(d/2,0,h))
    scene.frame_set(1)
    focus=Vector(pins[1].location)
    bpy.ops.object.camera_add(location=focus+Vector((.09,-.14,.065)))
    detail_camera=bpy.context.object;detail_camera.name='Joint B assembly inspection camera'
    look_at(detail_camera,focus);detail_camera.data.type='ORTHO';detail_camera.data.ortho_scale=.11
    detail_camera['role']='still inspection view; not the calibrated dataset camera'
    bpy.context.view_layer.update()
    payload_focus=payload.matrix_world.translation.copy()
    offset=links[1].rotation_euler.to_matrix()@Vector((.09,-.20,.075))
    bpy.ops.object.camera_add(location=payload_focus+offset)
    payload_camera=bpy.context.object;payload_camera.name='Payload mounting inspection camera'
    look_at(payload_camera,payload_focus);payload_camera.data.type='ORTHO';payload_camera.data.ortho_scale=.11
    payload_camera['role']='payload mounting still; not the calibrated dataset camera'
    presentation=appearance['presentation']
    target=Vector((d/2,.025,h+.01))
    bpy.ops.object.camera_add(location=target+Vector(presentation['camera_offset_m']))
    beauty_camera=bpy.context.object;beauty_camera.name='Assembly presentation camera'
    look_at(beauty_camera,target);beauty_camera.data.type='ORTHO'
    beauty_camera.data.ortho_scale=max(presentation['ortho_scale_m'],h*2.2,d*1.8)
    beauty_camera['role']='presentation still; not the calibrated dataset camera'
    scene.render.resolution_x=presentation['width'];scene.render.resolution_y=presentation['height']
    fit_assembly_camera(scene,beauty_camera)
    scene.render.resolution_x=config['simulation']['width'];scene.render.resolution_y=config['simulation']['height']
    scene.camera=camera
    fy=scene.render.resolution_y/(2*math.tan(math.radians(config['camera']['vertical_fov_deg'])/2))
    rotation=np.asarray(camera.rotation_euler.to_matrix()).T
    world_to_cv=np.diag([1,-1,-1])@rotation
    camera_metadata={'K_pixels':[[fy,0,scene.render.resolution_x/2],[0,fy,scene.render.resolution_y/2],[0,0,1]],
                     'R_world_to_camera_cv':world_to_cv.tolist(),'t_world_to_camera_cv_m':(-world_to_cv@np.asarray(camera.location)).tolist(),
                     'coordinate_convention':'world X horizontal, Y hinge-axis direction, Z up; camera x right, y down, z forward',
                     'fps':scene.render.fps,'frame_times_s':trajectory['time'].tolist(),'renderer':scene.render.engine,'blender_version':bpy.app.version_string}
    (output/'blender_camera.json').write_text(json.dumps(camera_metadata,indent=2)+'\n')
    scene_manifest.organize(scene)
    manifest=scene_manifest.export(scene,source,output,config,trajectory,links,style_name,appearance,args.appearance)
    print('Appearance asset stats:',manifest['asset_stats'],flush=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'fourbar.blend'))
    if args.render or args.preview_only:
        scene.render.image_settings.file_format='PNG';scene.render.filepath=str(output/'blender_preview.png')
        bpy.ops.render.render(write_still=True)
    if args.render and not args.preview_only:
        frames=output/'blender_frames';frames.mkdir(exist_ok=True)
        scene.render.filepath=str(frames/'frame_');scene.render.image_settings.file_format='PNG'
        bpy.ops.render.render(animation=True)
    if args.render or args.preview_only:
        scene.frame_set(1);scene.camera=detail_camera
        scene.render.filepath=str(output/'joint_detail.png')
        bpy.ops.render.render(write_still=True)
        scene.camera=payload_camera;scene.render.filepath=str(output/'payload_detail.png')
        bpy.ops.render.render(write_still=True)
        scene.camera=camera
    if args.beauty:
        engine=scene.render.engine
        scene.render.engine='CYCLES';scene.cycles.samples=presentation['cycles_samples'];scene.cycles.use_denoising=True
        preferences=bpy.context.preferences.addons['cycles'].preferences
        try:
            preferences.compute_device_type='CUDA';preferences.get_devices()
            gpu=any(device.type=='CUDA' for device in preferences.devices)
            for device in preferences.devices:device.use=device.type=='CUDA'
            scene.cycles.device='GPU' if gpu else 'CPU'
        except (TypeError,RuntimeError):scene.cycles.device='CPU'
        scene.render.resolution_x=presentation['width'];scene.render.resolution_y=presentation['height']
        scene.camera=beauty_camera;scene.frame_set(1)
        scene.render.filepath=str(output/'assembly_beauty.png')
        bpy.ops.render.render(write_still=True)
        scene.camera=camera;scene.render.engine=engine
        scene.render.resolution_x=config['simulation']['width'];scene.render.resolution_y=config['simulation']['height']


if __name__=='__main__':main()
