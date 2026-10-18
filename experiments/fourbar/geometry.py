"""Shared render mesh: extruded capsule plate with two axial through bores.

The independent dynamics uses analytic solid moments, not this tessellation.
Coordinates: X along pivot centers, Y thickness, Z plate width.
"""

import math


def link_mesh(length, width, thickness, bore_radius, segments=128):
    radius = width / 2
    if not 0 < bore_radius < radius < length / 2:
        raise ValueError('Require bore radius < end radius < half pivot spacing')
    corner = math.atan2(radius, length / 2)
    angles = sorted(set([2*math.pi*i/segments for i in range(segments)] +
                        [corner, math.pi/2, 3*math.pi/2, 2*math.pi-corner]))
    vertices, faces, lookup = [], [], {}

    def vertex(point):
        # Both half plates share this exact analytic plane. Canonicalize it
        # before rounded-key welding, including decimal-rounding tie cases.
        if abs(point[0] - length / 2) < 1e-10:
            point = (length / 2, point[1], point[2])
        key = tuple(round(value, 12) for value in point)
        if key not in lookup:
            lookup[key] = len(vertices)
            vertices.append(point)
        return lookup[key]

    def triangle(points, reflected):
        indices = [vertex(point) for point in points]
        faces.append(indices[::-1] if reflected else indices)

    for reflected in (False, True):
        rings = []
        for angle in angles:
            c, s = math.cos(angle), math.sin(angle)
            outer = radius if c <= 0 else min(
                radius/max(abs(s), 1e-15), length/(2*max(c, 1e-15)))
            ring = []
            for distance in (bore_radius, outer):
                x, z = distance*c, distance*s
                if reflected:
                    x = length-x
                ring.extend([(x, -thickness/2, z), (x, thickness/2, z)])
            rings.append(ring)
        for i, a in enumerate(rings):
            b = rings[(i+1) % len(rings)]
            # Front and back annulus sectors, then bore wall.
            for points in ([a[0], a[2], b[2]], [a[0], b[2], b[0]],
                           [a[1], b[3], a[3]], [a[1], b[1], b[3]],
                           [a[0], b[0], b[1]], [a[0], b[1], a[1]]):
                triangle(points, reflected)
            # Half plates meet without an internal wall; shared seam vertices weld.
            if not (abs(a[2][0]-length/2) < 1e-10 and
                    abs(b[2][0]-length/2) < 1e-10):
                triangle([a[2], a[3], b[3]], reflected)
                triangle([a[2], b[3], b[2]], reflected)
    return vertices, faces
