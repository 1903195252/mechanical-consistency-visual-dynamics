"""SI solid moments shared by the part inventory and independent rigid dynamics."""
import math
import numpy as np


MATERIALS = {
    "aluminum": {"density_kg_m3": 2700., "rgb": [.56,.61,.66], "metallic": .8, "roughness": .29},
    "steel": {"density_kg_m3": 7850., "rgb": [.43,.48,.54], "metallic": .9, "roughness": .24},
    "bronze": {"density_kg_m3": 8800., "rgb": [.43,.25,.08], "metallic": .8, "roughness": .30},
    "blue": {"density_kg_m3": 2700., "rgb": [.045,.28,.51], "metallic": .72, "roughness": .3},
    "orange": {"density_kg_m3": 2700., "rgb": [.82,.26,.055], "metallic": .72, "roughness": .3},
    "green": {"density_kg_m3": 2700., "rgb": [.04,.32,.22], "metallic": .72, "roughness": .3},
    "graphite": {"density_kg_m3": 7850., "rgb": [.035,.048,.062], "metallic": .6, "roughness": .4},
    "frame": {"density_kg_m3": 2700., "rgb": [.055,.075,.10], "metallic": .4, "roughness": .35},
}


def shift_tensor(mass, displacement):
    d = np.asarray(displacement, dtype=float)
    return mass * (np.dot(d,d) * np.eye(3) - np.outer(d,d))


def cylinder(radius, depth, density, inner=0.):
    mass = density * math.pi * (radius**2-inner**2) * depth
    polar = .5*mass*(radius**2+inner**2)
    transverse = mass*(3*(radius**2+inner**2)+depth**2)/12
    return mass, np.zeros(3), np.diag([transverse,polar,transverse])


def moments(shape, dimensions, density):
    d = dimensions
    if shape in ("cylinder","ring"):
        return cylinder(d["radius"],d["depth"],density,d.get("inner_radius",0.))
    if shape == "box":
        size=np.asarray(d["size"],dtype=float)
        mass=density*float(np.prod(size))
        return mass,np.zeros(3),np.diag(mass*(np.sum(size**2)-size**2)/12)
    if shape == "bearing_block":
        full,_,tensor=moments("box",d,density)
        hole,_,removed=cylinder(d["bore_radius"],d["size"][1],density)
        return full-hole,np.zeros(3),tensor-removed
    if shape == "rectangular_sleeve":
        full,_,tensor=moments("box",d,density)
        opening={"size":[d["size"][0],*d["opening_yz"]]}
        hole,_,removed=moments("box",opening,density)
        return full-hole,np.zeros(3),tensor-removed
    if shape == "socket_head":
        full,_,tensor=cylinder(d["radius"],d["depth"],density)
        r=.47*d["radius"];h=.65*d["depth"]
        hole=density*(3*math.sqrt(3)/2)*r*r*h
        hole_com=np.array([0.,-.175*d["depth"],0.])
        hole_polar=5*hole*r*r/12
        hole_tensor=np.diag([hole_polar/2+hole*h*h/12,hole_polar,hole_polar/2+hole*h*h/12])
        mass=full-hole
        com=-hole*hole_com/mass
        tensor=tensor-hole_tensor-shift_tensor(hole,hole_com)-shift_tensor(mass,com)
        return mass,com,tensor
    if shape == "hex_nut":
        r,h,ri=d["radius"],d["depth"],d["inner_radius"]
        full=density*(3*math.sqrt(3)/2)*r*r*h
        polar=5*full*r*r/12
        tensor=np.diag([polar/2+full*h*h/12,polar,polar/2+full*h*h/12])
        hole,_,removed=cylinder(ri,h,density)
        return full-hole,np.zeros(3),tensor-removed
    if shape == "link":
        L,w,t,bore=d["length"],d["width"],d["thickness"],d["bore_radius"]
        r=w/2
        area=2*r*L+math.pi*r*r-2*math.pi*bore*bore
        mass=area*t*density
        xx=(2*r*L**3/12+math.pi*r*r*L*L/4+4*L*r**3/3+math.pi*r**4/4
            -math.pi*bore*bore*L*L/2-math.pi*bore**4/2)
        zz=2*L*r**3/3+math.pi*r**4/4-math.pi*bore**4/2
        inertia=np.diag([density*t*zz+mass*t*t/12,density*t*(xx+zz),density*t*xx+mass*t*t/12])
        return mass,np.array([L/2,0.,0.]),inertia
    raise ValueError("Unsupported solid: "+shape)


def part(name, body, shape, position, dimensions, material, *, density=None, note=None, axis="Y"):
    density=MATERIALS[material]["density_kg_m3"] if density is None else density
    mass,local_com,tensor=moments(shape,dimensions,density)
    if axis == "Z":
        rotation=np.array([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]])
        local_com=rotation@local_com;tensor=rotation@tensor@rotation.T
    elif axis != "Y":
        raise ValueError("Unsupported part axis")
    if mass<=0 or np.linalg.eigvalsh(tensor).min()<=0:
        raise ValueError("Nonphysical solid: "+name)
    com=np.asarray(position,dtype=float)+local_com
    return {"name":name,"body":body,"shape":shape,"axis":axis,"position_m":list(position),"dimensions_m":dimensions,
            "material":material,"density_kg_m3":float(density),"mass_kg":float(mass),
            "com_body_m":com.tolist(),"inertia_com_body_kg_m2":tensor.tolist(),
            "note":note or "Nominal solid mass includes this part; visual edge finishing is surface detail"}


def aggregate(parts, body):
    items=[p for p in parts if p["body"]==body]
    total=sum(p["mass_kg"] for p in items)
    if total<=0:raise ValueError("Body has no mass: "+body)
    center=sum(p["mass_kg"]*np.asarray(p["com_body_m"]) for p in items)/total
    tensor=sum(np.asarray(p["inertia_com_body_kg_m2"])+shift_tensor(p["mass_kg"],np.asarray(p["com_body_m"])-center)
               for p in items)
    eigen=np.linalg.eigvalsh(tensor)
    if eigen.min()<=0 or eigen.max()>eigen.sum()/2+1e-12:
        raise ValueError("Invalid aggregate inertia: "+body)
    return {"name":body,"mass_kg":total,"com_local_m":center.tolist(),"inertia_com_local_kg_m2":tensor.tolist(),
            "part_names":[p["name"] for p in items]}
