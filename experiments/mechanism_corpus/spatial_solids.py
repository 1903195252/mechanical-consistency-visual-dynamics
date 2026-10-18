"""Oriented nominal solids and independently integrated bodies of revolution."""
import math
import numpy as np
from .solids import MATERIALS,moments


def rx(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[1.,0,0],[0,c,-s],[0,s,c]])


def ry(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1.,0],[-s,0,c]])


def rz(a):
    c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1.]])


def profile_radii_squared(y,d):
    y=np.asarray(y)
    if d['profile_type']=='sphere_segment':
        return np.maximum(0.,d['radius']**2-y*y),np.zeros_like(y)
    if d['profile_type']=='ball_socket':
        return np.full_like(y,d['outer_radius']**2),np.maximum(0.,d['cavity_radius']**2-y*y)
    raise ValueError('Unknown axial profile')


def profile_breaks(d):
    points=[d['lower'],d['upper']]
    if d['profile_type']=='ball_socket':
        points.extend(v for v in [-d['cavity_radius'],d['cavity_radius']] if d['lower']<v<d['upper'])
    return sorted(points)


def profile_moments(d,density):
    nodes,weights=np.polynomial.legendre.leggauss(16)
    values=np.zeros(4)
    for lo,hi in zip(profile_breaks(d)[:-1],profile_breaks(d)[1:]):
        y=(lo+hi)/2+nodes*(hi-lo)/2;w=weights*(hi-lo)/2
        out,inside=profile_radii_squared(y,d);area=np.pi*(out-inside)
        values+=np.array([np.dot(w,area),np.dot(w,y*area),np.dot(w,y*y*area),np.dot(w,np.pi*(out*out-inside*inside))])
    volume,first,second,radial=values;mass=volume*density;com=first/volume
    polar=density*radial/2;transverse=density*(radial/4+second)-mass*com*com
    return mass,np.array([0.,com,0.]),np.diag([transverse,polar,transverse])


def profile_mesh(d,segments=128,axial_segments=96):
    lo,hi=d['lower'],d['upper']
    ys=np.unique(np.r_[np.linspace(lo,hi,axial_segments+1),profile_breaks(d)])
    out,inside=profile_radii_squared(ys,d)
    contour=list(zip(np.sqrt(out),ys))+list(zip(np.sqrt(inside[::-1]),ys[::-1]))
    clean=[]
    for pair in contour:
        if not clean or np.linalg.norm(np.asarray(pair)-clean[-1])>1e-12:clean.append(pair)
    if np.linalg.norm(np.asarray(clean[0])-clean[-1])<1e-12:clean.pop()
    vertices=[];rings=[];faces=[]
    for radius,y in clean:
        if radius<1e-9:
            rings.append([len(vertices)]);vertices.append((0.,float(y),0.))
        else:
            rings.append(list(range(len(vertices),len(vertices)+segments)))
            vertices.extend((float(radius*np.cos(2*np.pi*i/segments)),float(y),float(radius*np.sin(2*np.pi*i/segments))) for i in range(segments))
    for current,following in zip(rings,rings[1:]+rings[:1]):
        if len(current)==len(following)==1:continue
        for i in range(segments):
            j=(i+1)%segments
            if len(current)==1:faces.append((following[i],following[j],current[0]))
            elif len(following)==1:faces.append((following[0],current[j],current[i]))
            else:faces.append((following[i],following[j],current[j],current[i]))
    return vertices,faces


def part(name,body,shape,position,dimensions,material,rotation=None,density=None):
    density=MATERIALS[material]['density_kg_m3'] if density is None else float(density)
    mass,center,inertia=(profile_moments(dimensions,density) if shape=='axial_profile' else moments(shape,dimensions,density))
    R=np.eye(3) if rotation is None else np.asarray(rotation)
    if not np.allclose(R.T@R,np.eye(3),atol=1e-10) or np.linalg.det(R)<.999999:raise ValueError('Invalid part rotation')
    center=np.asarray(position)+R@center;inertia=R@inertia@R.T
    if mass<=0 or np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Invalid physical solid')
    return {'name':name,'body':body,'shape':shape,'axis':'Y','rotation_matrix':R.tolist(),
            'position_m':list(position),'dimensions_m':dimensions,'material':material,
            'density_kg_m3':density,'mass_kg':float(mass),'com_body_m':center.tolist(),
            'inertia_com_body_kg_m2':inertia.tolist(),
            'note':'Oriented nominal physical solid; all moving hardware is assigned to exactly one body.'}
