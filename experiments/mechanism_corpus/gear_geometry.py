"""Metric involute spur/helical geometry and cross-section mass integrals.

The involute construction was checked against Gabriel Nix Morris's MIT
`mechanism` implementation (retained license in licenses/mechanism_MIT.txt)
and KHK's normal/transverse helical-gear dimension reference. This module
implements a complete, bored gear in SI units rather than a plotted tooth.
"""
import math
from pathlib import Path
import sys
import numpy as np
from .solids import MATERIALS


def dimensions(parameters):
    z=parameters['teeth'];mn=parameters['normal_module_m'];beta=parameters['helix_angle_rad']
    mt=mn/np.cos(beta);alpha=np.arctan(np.tan(parameters['normal_pressure_angle_rad'])/np.cos(beta))
    radius=z*mt/2;base=radius*np.cos(alpha);tip=radius+mn;root=radius-1.25*mn
    return {'pitch_radius_m':float(radius),'base_radius_m':float(base),'tip_radius_m':float(tip),
            'root_radius_m':float(root),'transverse_module_m':float(mt),'transverse_pressure_angle_rad':float(alpha)}


def outline(parameters,flank_samples=32,tip_samples=8,root_samples=12):
    p=dimensions(parameters);z=parameters['teeth'];rb=p['base_radius_m'];rp=p['pitch_radius_m']
    root,tip=p['root_radius_m'],p['tip_radius_m'];pitch_involute=np.tan(p['transverse_pressure_angle_rad'])-p['transverse_pressure_angle_rad']
    t0=np.sqrt(max(0.,(root/rb)**2-1));t1=np.sqrt((tip/rb)**2-1)
    ts=np.linspace(t0,t1,flank_samples+1);radii=rb*np.sqrt(1+ts*ts)
    half=np.pi/(2*z)+pitch_involute-(ts-np.arctan(ts))
    left=np.column_stack([radii*np.cos(-half),radii*np.sin(-half)])
    if root<rb:
        # The active tooth remains an exact involute. Below its base circle,
        # a radial root section meets a tangent circular fillet and root arc.
        fillet=min(.25*parameters['normal_module_m'],.65*(rb-root))
        flank_angle=-half[0];center_radius=root+fillet
        center_angle=flank_angle-np.arcsin(fillet/center_radius)
        center=center_radius*np.array([np.cos(center_angle),np.sin(center_angle)])
        arc_angle=np.linspace(center_angle+np.pi,flank_angle+np.pi/2,9)
        curve=center+fillet*np.column_stack([np.cos(arc_angle),np.sin(arc_angle)])
        left=np.vstack([curve,left])
    left_angles=np.arctan2(left[:,1],left[:,0]);left_radii=np.linalg.norm(left,axis=1)
    root_half=-left_angles[0]
    if half[-1]<=0 or 2*root_half>=2*np.pi/z:raise ValueError('Involute tooth has invalid tip/root thickness')
    angle=[];radius=[]
    for i in range(z):
        center=2*np.pi*i/z
        angle.extend(center+left_angles);radius.extend(left_radii)
        arc=np.linspace(-half[-1],half[-1],tip_samples+1)[1:]
        angle.extend(center+arc);radius.extend(np.full(len(arc),tip))
        angle.extend(center-left_angles[-2::-1]);radius.extend(left_radii[-2::-1])
        gap=np.linspace(center+root_half,center+2*np.pi/z-root_half,root_samples+1)[1:-1]
        angle.extend(gap);radius.extend(np.full(len(gap),root))
    angle=np.asarray(angle);radius=np.asarray(radius)
    if np.any(np.diff(angle)<-1e-12):raise ValueError('Gear profile angular order is invalid')
    return np.column_stack([radius*np.cos(angle),radius*np.sin(angle)]),angle


def section_integrals(parameters):
    points,_=outline(parameters);next_points=np.roll(points,-1,axis=0)
    x,z=points.T;u,v=next_points.T;cross=x*v-u*z
    area=float(cross.sum()/2)-np.pi*parameters['bore_radius_m']**2
    second_x=float(np.sum(cross*(x*x+x*u+u*u))/12)-np.pi*parameters['bore_radius_m']**4/4
    second_z=float(np.sum(cross*(z*z+z*v+v*v))/12)-np.pi*parameters['bore_radius_m']**4/4
    if area<=0 or min(second_x,second_z)<=0:raise ValueError('Nonphysical bored gear section')
    return area,second_x+second_z


def nominal_moments(parameters,density):
    area,polar_area=section_integrals(parameters);depth=parameters['depth_m'];mass=area*depth*density
    # C_z symmetry (z>=20) makes the transverse section second moments equal,
    # including during a helical sweep. The pole moment is sweep invariant.
    polar=polar_area*depth*density;transverse=polar/2+mass*depth*depth/12
    return mass,np.diag([transverse,polar,transverse])


def mesh(parameters,inner_segments=128,helical_slices=16):
    points,angles=outline(parameters);N=len(points);K=inner_segments;p=dimensions(parameters)
    count=helical_slices+1 if abs(parameters['helix_angle_rad'])>1e-12 else 2
    ys=np.linspace(-parameters['depth_m']/2,parameters['depth_m']/2,count)
    vertices=[];outer=[];inner=[];faces=[]
    start=angles[0];inner_angles=start+np.arange(K)*2*np.pi/K
    for y in ys:
        twist=y*np.tan(parameters['helix_angle_rad'])/p['pitch_radius_m'];c,s=np.cos(twist),np.sin(twist)
        rotated=points@np.array([[c,s],[-s,c]])
        outer.append(list(range(len(vertices),len(vertices)+N)))
        vertices.extend((float(x),float(y),float(z)) for x,z in rotated)
        inner.append(list(range(len(vertices),len(vertices)+K)))
        vertices.extend((float(parameters['bore_radius_m']*np.cos(a+twist)),float(y),
                         float(parameters['bore_radius_m']*np.sin(a+twist))) for a in inner_angles)
    for layer in range(count-1):
        for i in range(N):
            j=(i+1)%N;faces.append((outer[layer+1][i],outer[layer+1][j],outer[layer][j],outer[layer][i]))
        for i in range(K):
            j=(i+1)%K;faces.append((inner[layer][i],inner[layer][j],inner[layer+1][j],inner[layer+1][i]))
    try:
        import mapbox_earcut
    except ImportError:
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'output/fourbar/.deps'/('python%d%d'%sys.version_info[:2])))
        import mapbox_earcut
    hole=parameters['bore_radius_m']*np.column_stack([np.cos(inner_angles),np.sin(inner_angles)])
    cap_points=np.vstack([points,hole])
    triangles=mapbox_earcut.triangulate_float64(cap_points,np.asarray([N,N+K],dtype=np.uint32)).reshape(-1,3)
    signed=np.cross(cap_points[triangles[:,1]]-cap_points[triangles[:,0]],cap_points[triangles[:,2]]-cap_points[triangles[:,0]])
    if np.any(abs(signed)<1e-16):raise ValueError('Degenerate gear-face triangulation')
    triangles[signed<0]=triangles[signed<0][:,::-1]
    for layer in [0,count-1]:
        indices=outer[layer]+inner[layer]
        for triangle_indices in triangles:
            triangle=tuple(indices[i] for i in triangle_indices)
            faces.append(triangle if layer==0 else triangle[::-1])
    return vertices,faces


def part(name,body,parameters,material='steel'):
    density=MATERIALS[material]['density_kg_m3'];mass,inertia=nominal_moments(parameters,density)
    return {'name':name,'body':body,'shape':'involute_gear','axis':'Y','position_m':[0.,0.,0.],
            'dimensions_m':parameters,'material':material,'density_kg_m3':density,'mass_kg':float(mass),
            'com_body_m':[0.,0.,0.],'inertia_com_body_kg_m2':inertia.tolist(),
            'note':'Nominal polygonal involute section with a circular shaft bore; signed helix sweep preserves section mass and polar inertia.'}
