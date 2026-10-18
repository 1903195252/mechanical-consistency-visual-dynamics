"""Reusable Blender parts; render finishing and idealized connector hardware.

Mesh assets are shared by equal-size fasteners. Dimensions remain SI metres.
No function in this module edits simulator states or physical parameters.
"""

import math
import bpy
from mathutils import Vector

SEGMENTS = 96
BEVEL_SEGMENTS = 4
_SOCKET_MESHES = {}


def material(name, color, metal=0., roughness=.35, brushed=False, coat=0.):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    shader = nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1)
    shader.inputs['Metallic'].default_value = metal
    shader.inputs['Roughness'].default_value = roughness
    shader.inputs['Coat Weight'].default_value = coat
    shader.inputs['Coat Roughness'].default_value = .25
    if brushed:
        shader.inputs['Anisotropic'].default_value = .32
        coordinates = nodes.new('ShaderNodeTexCoord')
        stretch = nodes.new('ShaderNodeVectorMath'); stretch.operation = 'MULTIPLY'
        stretch.inputs[1].default_value = (180, 3200, 3200)
        noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 1
        noise.inputs['Detail'].default_value = 2
        rough = nodes.new('ShaderNodeMapRange')
        rough.inputs['To Min'].default_value = roughness-.025
        rough.inputs['To Max'].default_value = roughness+.025
        bump = nodes.new('ShaderNodeBump')
        bump.inputs['Strength'].default_value = .13
        bump.inputs['Distance'].default_value = .000004
        links.new(coordinates.outputs['Object'], stretch.inputs[0])
        links.new(stretch.outputs['Vector'], noise.inputs['Vector'])
        links.new(noise.outputs['Fac'], rough.inputs['Value'])
        links.new(rough.outputs['Result'], shader.inputs['Roughness'])
        links.new(noise.outputs['Fac'], bump.inputs['Height'])
        links.new(bump.outputs['Normal'], shader.inputs['Normal'])
    return mat


def finish(obj, width=.0005):
    bevel = obj.modifiers.new('Machined edge finishing', 'BEVEL')
    bevel.width = width; bevel.segments = BEVEL_SEGMENTS
    bevel.limit_method = 'ANGLE'; bevel.angle_limit = .5
    bevel.harden_normals = True
    return obj


def box(name, position, dimensions, mat, parent=None, bevel=.001):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object; obj.name = name; obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.parent = parent; obj.location = position
    obj.data.materials.append(mat)
    obj['physics_role'] = 'fixed fixture visualization'
    return finish(obj, bevel) if bevel else obj


def cylinder(name, position, radius, depth, mat, parent=None, vertices=None):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices or SEGMENTS, radius=radius, depth=depth)
    obj = bpy.context.object; obj.name = name
    obj.parent = parent; obj.location = position; obj.rotation_euler = (math.pi/2, 0, 0)
    obj.data.materials.append(mat)
    for face in obj.data.polygons:
        face.use_smooth = len(face.vertices) == 4
    obj['physics_role'] = 'fixed or massless connector visualization'
    return finish(obj, min(.00025, depth/8))


def ring(name, position, outer, inner, depth, mat, parent=None):
    if not 0 < inner < outer or depth <= 0:
        raise ValueError('Invalid annular part dimensions')
    vertices, faces = [], []
    for i in range(SEGMENTS):
        angle = 2*math.pi*i/SEGMENTS
        for radius, y in [(outer, -depth/2), (outer, depth/2),
                          (inner, -depth/2), (inner, depth/2)]:
            vertices.append((radius*math.cos(angle), y, radius*math.sin(angle)))
    for i in range(SEGMENTS):
        a, b = 4*i, 4*((i+1) % SEGMENTS)
        faces.extend([(a+1, b+1, b, a), (b+2, b+3, a+3, a+2),
                      (b, b+2, a+2, a), (a+3, b+3, b+1, a+1)])
    mesh = bpy.data.meshes.new(name); mesh.from_pydata(vertices, [], faces); mesh.update()
    obj = bpy.data.objects.new(name, mesh); bpy.context.scene.collection.objects.link(obj)
    obj.location = position; obj.parent = parent; mesh.materials.append(mat)
    for i, face in enumerate(mesh.polygons):
        face.use_smooth = i % 4 < 2
    obj['physics_role'] = 'massless ideal connector visualization'
    return finish(obj, min(.00012, depth/8))


def _socket_mesh(radius, depth):
    # Closed analytic mesh with a blind hex socket. No scene-dependent Booleans.
    key = (radius, depth, SEGMENTS)
    if key in _SOCKET_MESHES:
        return _SOCKET_MESHES[key]
    vertices, faces = [], []
    n = ((SEGMENTS+5)//6)*6
    for i in range(n):
        angle = 2*math.pi*i/n
        socket_radius = radius*.47*math.cos(math.pi/6)/math.cos(angle % (math.pi/3)-math.pi/6)
        for r, z in [(radius, -depth/2), (radius, depth/2),
                     (socket_radius, depth/2), (socket_radius, -depth*.15)]:
            vertices.append((r*math.cos(angle), r*math.sin(angle), z))
    vertices.extend([(0, 0, -depth/2), (0, 0, -depth*.15)])
    for i in range(n):
        a, b = 4*i, 4*((i+1) % n)
        faces.extend([(a, b, b+1, a+1), (a+1, b+1, b+2, a+2),
                      (a+2, b+2, b+3, a+3), (b, a, 4*n), (a+3, b+3, 4*n+1)])
    mesh = bpy.data.meshes.new('Socket head %.4f x %.4f' % (radius, depth))
    mesh.from_pydata(vertices, [], faces); mesh.update()
    for i, face in enumerate(mesh.polygons):
        face.use_smooth = i % 5 == 0
    _SOCKET_MESHES[key] = mesh
    return mesh


def socket_head(name, position, radius, depth, mat, parent=None, axis='Y'):
    if axis not in ('Y', 'Z'):
        raise ValueError('Fastener axis must be Y or Z')
    mesh = _socket_mesh(radius, depth)
    obj = bpy.data.objects.new(name, mesh); bpy.context.scene.collection.objects.link(obj)
    # Object-linked material slots allow the same mesh to carry different finishes.
    if not mesh.materials:
        mesh.materials.append(mat)
    obj.material_slots[0].link = 'OBJECT'; obj.material_slots[0].material = mat
    obj.parent = parent; obj.location = position
    obj.rotation_euler = (math.pi/2, 0, 0) if axis == 'Y' else (0, 0, 0)
    obj['asset_type'] = 'socket_head'; obj['radius_m'] = radius; obj['depth_m'] = depth
    obj['physics_role'] = 'fixed fastener or massless connector visualization'
    return finish(obj, min(.00015, depth/10))


def cut_fixture(obj, cutter):
    # Apply before finishing the fixed component; nominal moving solids stay intact.
    modifiers = [(m.width, m.segments) for m in obj.modifiers if m.type == 'BEVEL']
    obj.modifiers.clear(); cutter.modifiers.clear()
    bpy.context.view_layer.update()
    boolean = obj.modifiers.new('Machined fixture recess', 'BOOLEAN')
    boolean.operation = 'DIFFERENCE'; boolean.object = cutter
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=boolean.name)
    bpy.data.objects.remove(cutter, do_unlink=True)
    for width, _ in modifiers:
        finish(obj, width)


def stamp(name, text, position, size, mat, parent=None):
    curve = bpy.data.curves.new(name, 'FONT'); curve.body = text; curve.size = size
    curve.align_x = 'CENTER'; curve.align_y = 'CENTER'; curve.extrude = .000008
    obj = bpy.data.objects.new(name, curve); bpy.context.scene.collection.objects.link(obj)
    obj.parent = parent; obj.location = position; obj.rotation_euler = (math.pi/2, 0, 0)
    curve.materials.append(mat)
    obj['physics_role'] = 'surface marking; no mass'
    return obj


def look_at(obj, target):
    obj.rotation_euler = (Vector(target)-obj.location).to_track_quat('-Z', 'Y').to_euler()


def fit_assembly_camera(scene, camera, margin=1.12):
    """Fit a presentation camera to real assembly bounds at the current frame."""
    bpy.context.view_layer.update()
    points = []
    for obj in scene.objects:
        if obj.type not in ('MESH', 'FONT') or obj.name in ('Floor', 'Studio cyclorama'):
            continue
        points.extend(obj.matrix_world @ Vector(corner) for corner in obj.bound_box)
    inverse = camera.rotation_euler.to_matrix().transposed()
    projected = [inverse @ (point-camera.location) for point in points]
    bounds = [(min(point[i] for point in projected), max(point[i] for point in projected)) for i in (0, 1)]
    camera.location += camera.rotation_euler.to_matrix() @ Vector(
        ((bounds[0][0]+bounds[0][1])/2, (bounds[1][0]+bounds[1][1])/2, 0))
    old_scale = camera.data.ortho_scale
    frame = camera.data.view_frame(scene=scene)
    frame_width = max(point.x for point in frame)-min(point.x for point in frame)
    frame_height = max(point.y for point in frame)-min(point.y for point in frame)
    factor = max((bounds[0][1]-bounds[0][0])/frame_width,
                 (bounds[1][1]-bounds[1][0])/frame_height)*margin
    camera.data.ortho_scale = old_scale*factor
    camera['framing'] = 'Assembly bounds at first recorded frame, with 12 percent margin'
