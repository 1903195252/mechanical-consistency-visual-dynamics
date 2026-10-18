"""Verify exported artifacts and capture reproducible dataset provenance."""

from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    project=Path(__file__).resolve().parents[2]
    root=project/'output'/'fourbar'
    source=Path(__file__).resolve().parent
    manifest={'created_at':datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'purpose':'Synthetic planar single-loop 4R rigid-body data prototype',
              'cases':[],'source_sha256':{}}
    for filename in ['config.json','appearance.json','geometry.py','mechanics.py','simulate.py','export_physics.py',
                     'blender_scene.py','visual_parts.py','scene_manifest.py','render_batch.py',
                     'test_mechanics.py','run.sh','finalize.py','README.md']:
        manifest['source_sha256'][filename]=hashlib.sha256((source/filename).read_bytes()).hexdigest()
    nominal_actions=None
    for case in ['nominal','heavy_payload','high_friction']:
        folder=root/case
        config=json.loads((folder/'physical_metadata.json').read_text())
        report=json.loads((folder/'validation.json').read_text())
        with np.load(folder/'ground_truth.npz') as data:
            applied=np.zeros_like(data['q']);applied[:,0]=data['motor_torque']
            residual=np.einsum('nij,nj->ni',data['mass_matrix_kg_m2'],data['qddot'])+data['coriolis_Nm']+data['gravity_Nm']-applied-data['generalized_friction_Nm']-data['external_generalized_force_Nm']-data['generalized_constraint_force_Nm']
            report['exported_full_equation_residual_max_Nm']=float(np.max(np.linalg.norm(residual,axis=1)))
            report['checks']['exported_equation_terms_consistent']=report['exported_full_equation_residual_max_Nm']<1e-8
            assert all(np.isfinite(data[name]).all() for name in data.files)
        with np.load(folder/'actions.npz') as data:
            actions={name:data[name].copy() for name in data.files}
            if nominal_actions is None:nominal_actions=actions
            for name in actions:assert np.array_equal(actions[name],nominal_actions[name]),(case,name)
        report['checks']['matched_exogenous_actions']=True
        report['passed']=all(report['checks'].values())
        assert report['passed'],case
        assert (folder/'fourbar.blend').exists()
        appearance=json.loads((folder/'appearance_manifest.json').read_text())
        assert appearance['checks']['animation_matches_recorded_states']
        assert appearance['appearance_file_sha256']==manifest['source_sha256']['appearance.json']
        for filename,digest in appearance['builder_source_sha256'].items():
            assert digest==manifest['source_sha256'][filename],(case,filename)
        for filename,digest in appearance['physics_input_sha256'].items():
            assert hashlib.sha256((folder/filename).read_bytes()).hexdigest()==digest,(case,filename)
        (folder/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
        entry={'case':case,'passed':report['passed'],'payload_mass_kg':config['payload_mass_kg'],
               'friction_multiplier':3 if case=='high_friction' else 1,
               'validation':str(Path(case)/'validation.json'),
               'loop_position_max_m':report['closure_position_max_m'],
               'global_reference_angle_error_max_rad':report['independent_replayed_torque_angle_max_error_rad'],
               'file_sha256':{}}
        for filename in ['model.xml','physical_metadata.json','ground_truth.npz','actions.npz','video_trajectory.npz','fourbar.blend',
                         'blender_camera.json','blender_preview.png','joint_detail.png','mujoco_rgb.mp4',
                         'appearance_manifest.json','assembly_beauty.png','payload_detail.png']:
            entry['file_sha256'][filename]=hashlib.sha256((folder/filename).read_bytes()).hexdigest()
        if case=='nominal':
            entry['file_sha256']['blender_rgb.mp4']=hashlib.sha256((folder/'blender_rgb.mp4').read_bytes()).hexdigest()
        manifest['cases'].append(entry)
        print(case,'passed: geometry, dynamics, power balance, reference rollout and exported fields')
    (root/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
