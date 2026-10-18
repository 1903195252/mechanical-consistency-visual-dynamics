"""Check nominal mesh interpenetration between distinct physical rigid bodies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import bpy
import numpy as np
from mathutils import Quaternion
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.mechanism_corpus.catalog import write_json
from experiments.mechanism_corpus.audit_blender_solids import nominal_mesh_digest
from experiments.mechanism_corpus.solid_winding import winding_number
from experiments.mechanism_corpus.contact_rules import evaluate as evaluate_contact_rule


def inside(point,solid):
    value=abs(winding_number(point,solid['vertices'],solid['triangles']))
    if not np.isfinite(value) or .45<value<.55:raise RuntimeError('Ambiguous solid-boundary containment')
    return value>.5


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly-dir',type=Path,required=True)
    parser.add_argument('--conditions',nargs='+',default=['low_speed','rated','limit_or_load'])
    parser.add_argument('--frames',type=int,nargs='+',default=None)
    parser.add_argument('--output-report',type=Path,default=None)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]);folder=args.assembly_dir.resolve()
    assembly=json.loads((folder/'assembly.json').read_text());bpy.ops.wm.open_mainfile(filepath=str(folder/'assembly.blend'))
    meshes=[];inset=1e-6
    for obj in bpy.context.scene.objects:
        if obj.type!='MESH' or 'part_id' not in obj:continue
        local=np.asarray(obj.matrix_local);vertices=np.array([v.co for v in obj.data.vertices])@local[:3,:3].T+local[:3,3]
        normals=np.array([v.normal for v in obj.data.vertices])@local[:3,:3].T
        normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-15)
        vertices-=inset*normals
        meshes.append({'name':obj['part_id'],'body':obj['rigid_body_owner'],'vertices':vertices,
                       'faces':[list(p.vertices) for p in obj.data.polygons],
                       'triangles':np.asarray([[p.vertices[0],p.vertices[k],p.vertices[k+1]]
                                               for p in obj.data.polygons for k in range(1,len(p.vertices)-1)]),
                       'interior_probe':vertices[0]-normals[0]*2e-6})
    findings=[];samples=[];start=time.monotonic();candidate_count=0;declared_checks=[];accepted_contact_crossings=[]
    contact_rules=assembly.get('permitted_contacts',[]);part_map={p['name']:p for p in assembly['parts']}
    owners=np.asarray([mesh['body'] for mesh in meshes])
    cross_body=np.triu(owners[:,None]!=owners[None,:],k=1)
    for condition in args.conditions:
        with np.load(folder/'conditions'/condition/'render_poses.npz',allow_pickle=False) as source:
            names=list(source['body_names']);positions=source['body_position_m'];quaternions=source['body_quaternion_wxyz']
            pose_times=source['time']
        reference={}
        if contact_rules:
            with np.load(folder/'conditions'/condition/'physics.npz',allow_pickle=False) as source:
                reference={'time':source['time'],**{r['normal_force_field']:source[r['normal_force_field']] for r in contact_rules}}
        for frame in (args.frames if args.frames is not None else range(len(positions))):
            if not 0<=frame<len(positions):raise ValueError('Frame outside recorded poses')
            transforms={name:(np.asarray(Quaternion(quaternions[frame,index]).to_matrix(),dtype=np.float64),positions[frame,index])
                        for index,name in enumerate(names)}
            transforms['ground']=(np.eye(3),np.zeros(3));allowed={};frame_failures=0
            for rule in contact_rules:
                index=int(np.argmin(abs(reference['time']-pose_times[frame])))
                if abs(reference['time'][index]-pose_times[frame])>1e-8:raise ValueError('Contact force/pose timestamps differ')
                checked=evaluate_contact_rule(rule,part_map,transforms,reference[rule['normal_force_field']][index])
                checked.update(condition=condition,frame=frame);declared_checks.append(checked)
                key=frozenset(checked['parts']);allowed[key]=checked['passed']
                if not checked['passed']:
                    frame_failures+=1;findings.append({'condition':condition,'frame':frame,'parts':checked['parts'],
                                                       'kind':'declared_contact_invalid','contact_check':checked})
            prepared=[]
            for mesh in meshes:
                R,p=transforms[mesh['body']]
                points=mesh['vertices']@R.T+p
                prepared.append({'vertices':points,'lower':points.min(axis=0),'upper':points.max(axis=0),
                                 'probe':mesh['interior_probe']@R.T+p,'tree':None,'triangles':mesh['triangles']})
            lower=np.asarray([item['lower'] for item in prepared]);upper=np.asarray([item['upper'] for item in prepared])
            overlap_bounds=np.all(upper[:,None,:]>=lower[None,:,:],axis=2)
            candidates=np.argwhere(cross_body&overlap_bounds&overlap_bounds.T)
            for i,j in candidates:
                first,second=meshes[i],meshes[j];a,b=prepared[i],prepared[j];candidate_count+=1
                for index in [i,j]:
                    item=prepared[index]
                    if item['tree'] is None:item['tree']=BVHTree.FromPolygons(item['vertices'].tolist(),meshes[index]['faces'])
                overlap=a['tree'].overlap(b['tree']);kind=None
                if overlap:kind='surface_crossing'
                elif np.all(a['lower']>=b['lower']) and np.all(a['upper']<=b['upper']) and inside(a['probe'],b):kind='contained_solid'
                elif np.all(b['lower']>=a['lower']) and np.all(b['upper']<=a['upper']) and inside(b['probe'],a):kind='contained_solid'
                if kind:
                    if allowed.get(frozenset([first['name'],second['name']]),False):
                        accepted_contact_crossings.append({'condition':condition,'frame':frame,'parts':[first['name'],second['name']]})
                        continue
                    frame_failures+=1
                    findings.append({'condition':condition,'frame':frame,'parts':[first['name'],second['name']],
                                     'bodies':[first['body'],second['body']],'kind':kind,'triangle_pairs':len(overlap)})
            samples.append({'condition':condition,'frame':frame,'failures':frame_failures})
            print(json.dumps(samples[-1]),flush=True)
    result={'passed':not findings,'assembly_physical_digest':assembly['physical_digest'],'frames_checked':samples,
            'findings':findings,'candidate_pairs':candidate_count,'seconds':time.monotonic()-start,
            'declared_contact_checks':declared_checks,'accepted_bounded_contact_crossings':accepted_contact_crossings,
            'surface_inset_m':inset,'containment_method':'float64_signed_solid_angle_winding',
            'scope':'Nominal pre-finishing solids of distinct rigid bodies. Vertex-normal inset permits touching interfaces within 1 micrometre; same-body bolted/merged interfaces are excluded. AABB/BVH surface crossings and float64 signed-solid-angle full containment are tested at listed frames.'}
    result['nominal_mesh_digest']=nominal_mesh_digest([obj for obj in bpy.context.scene.objects if obj.type=='MESH' and 'part_id' in obj])
    result['render_pose_sha256']={condition:hashlib.sha256((folder/'conditions'/condition/'render_poses.npz').read_bytes()).hexdigest() for condition in args.conditions}
    if contact_rules:
        result['contact_rules_sha256']=hashlib.sha256(json.dumps(contact_rules,sort_keys=True).encode()).hexdigest()
        result['contact_reference_sha256']={condition:hashlib.sha256((folder/'conditions'/condition/'physics.npz').read_bytes()).hexdigest() for condition in args.conditions}
    write_json(args.output_report or folder/'blender_clearance_validation.json',result)
    print(json.dumps({'passed':result['passed'],'findings':len(findings),'seconds':result['seconds']}),flush=True)
    if findings:raise RuntimeError('Physical solid interpenetration detected')


if __name__=='__main__':main()
