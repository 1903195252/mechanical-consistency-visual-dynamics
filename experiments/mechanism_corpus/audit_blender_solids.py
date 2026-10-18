"""Independently integrate the Blender nominal solids, including moving hardware."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.mechanism_corpus.catalog import write_json
from experiments.mechanism_corpus.solids import shift_tensor


def nominal_mesh_digest(objects):
    digest=hashlib.sha256()
    for obj in sorted(objects,key=lambda obj:obj['part_id']):
        digest.update(obj['part_id'].encode())
        digest.update(np.asarray(obj.matrix_local,dtype=np.float64).tobytes())
        digest.update(np.asarray([v.co for v in obj.data.vertices],dtype=np.float32).tobytes())
        topology=[value for p in obj.data.polygons for value in [len(p.vertices),*p.vertices]]
        digest.update(np.asarray(topology,dtype=np.int32).tobytes())
    return digest.hexdigest()


def integrate(vertices,polygons,density):
    faces=[];edges={}
    for polygon in polygons:
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
        faces.extend([[polygon[0],polygon[i],polygon[i+1]] for i in range(1,len(polygon)-1)])
    vertices=np.asarray(vertices,dtype=np.float64)
    origin=(vertices.min(axis=0)+vertices.max(axis=0))/2
    tri=(vertices-origin)[np.asarray(faces)]
    volume=np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2]))/6
    total=volume.sum()
    if total<=0:raise ValueError("Solid has nonpositive oriented volume")
    center=np.einsum("i,ij->j",volume,tri.sum(axis=1)/4)/total
    xyz=tri-center
    shifted_volume=np.einsum("ij,ij->i",xyz[:,0],np.cross(xyz[:,1],xyz[:,2]))/6
    sums=xyz.sum(axis=1)
    second=(np.einsum("tki,tkj->tij",xyz,xyz)+np.einsum("ti,tj->tij",sums,sums))/20
    second=np.einsum("t,tij->ij",shifted_volume,second)*density
    inertia=np.trace(second)*np.eye(3)-second
    return total*density,center+origin,inertia,all(value==2 for value in edges.values())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--assembly-dir",type=Path,required=True)
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    folder=args.assembly_dir.resolve()
    assembly=json.loads((folder/"assembly.json").read_text())
    bpy.ops.wm.open_mainfile(filepath=str(folder/"assembly.blend"))
    lookup={obj["part_id"]:obj for obj in bpy.context.scene.objects if obj.type=="MESH" and "part_id" in obj}
    rows=[]
    for part in assembly["parts"]:
        obj=lookup[part["name"]]
        vertices=[np.asarray(obj.matrix_local@vertex.co) for vertex in obj.data.vertices]
        faces=[list(poly.vertices) for poly in obj.data.polygons]
        mass,center,inertia,closed=integrate(vertices,faces,part["density_kg_m3"])
        targetI=np.asarray(part["inertia_com_body_kg_m2"])
        row={"part":part["name"],"body":part["body"],"manifold_edge_pairs":closed,
             "mass_relative_error":float(abs(mass-part["mass_kg"])/part["mass_kg"]),
             "COM_error_m":float(np.linalg.norm(center-part["com_body_m"])),
             "inertia_relative_frobenius_error":float(np.linalg.norm(inertia-targetI)/np.linalg.norm(targetI))}
        row["passed"]=closed and row["mass_relative_error"]<.002 and row["COM_error_m"]<2e-6 and row["inertia_relative_frobenius_error"]<.003
        rows.append(row)
    result={"passed":all(row["passed"] for row in rows),"parts_checked":len(rows),"rows":rows,
            "scope":"Nominal pre-finishing mesh, local rigid-body frame. Visual micro-bevels are recorded surface finish.",
            "assembly_physical_digest":assembly["physical_digest"],
            "blend_sha256":hashlib.sha256((folder/"assembly.blend").read_bytes()).hexdigest(),
            "nominal_mesh_digest":nominal_mesh_digest(lookup.values())}
    write_json(folder/"blender_solid_validation.json",result)
    print(json.dumps({"passed":result["passed"],"parts":len(rows),"failures":[row for row in rows if not row["passed"]]}),flush=True)
    if not result["passed"]:raise RuntimeError("Blender nominal solid properties differ from physical part inventory")


if __name__=="__main__":main()
