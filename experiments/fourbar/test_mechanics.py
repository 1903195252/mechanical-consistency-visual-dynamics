"""Checks that distinguish geometric validity from dynamical validity."""

import json
from pathlib import Path
import unittest

import numpy as np
from geometry import link_mesh
from mechanics import (closure,closure_jacobian,inverse_kinematics,
                       mass_bias_energy,friction,properties,drive_torque)


class MechanicsTests(unittest.TestCase):
    def setUp(self):
        self.config=json.loads((Path(__file__).parent/'config.json').read_text())

    def test_full_crank_revolution_is_reachable_without_rank_loss(self):
        for angle in np.linspace(0,2*np.pi,361):
            q=inverse_kinematics(angle,self.config)
            self.assertLess(np.linalg.norm(closure(q,self.config)),1e-12)
            self.assertGreater(np.linalg.svd(closure_jacobian(q,self.config),compute_uv=False)[-1],0.03)

    def test_inertias_are_physically_admissible(self):
        masses,inertias=properties(self.config)
        self.assertTrue((masses>0).all())
        self.assertTrue((inertias>0).all())
        for inertia in inertias:
            self.assertLessEqual(2*max(inertia),sum(inertia)+1e-12)

    def test_render_solid_matches_analytic_mass_com_and_inertia(self):
        # Integrate signed tetrahedra of the actual shared render mesh.
        # This detects omitted holes, wrong end moments and flipped surface normals.
        masses,inertias=properties(self.config)
        for i,length in enumerate(self.config['lengths_m']):
            vertices,faces=link_mesh(length,self.config['bar_height_m'],
                                     self.config['bar_depth_m'],self.config['joint_bore_radius_m'])
            triangles=np.asarray(vertices)[faces]
            volumes=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))/6
            com=np.einsum('i,ij->j',volumes,triangles.sum(axis=1)/4)/volumes.sum()
            np.testing.assert_allclose(com,[length/2,0,0],atol=1e-12)
            triangles=triangles-com
            volumes=np.einsum('ij,ij->i',triangles[:,0],np.cross(triangles[:,1],triangles[:,2]))/6
            squares=(np.sum(triangles**2,axis=1)+np.sum(triangles,axis=1)**2)/20
            moments=np.einsum('i,ij->j',volumes,squares)*self.config['density_kg_m3']
            np.testing.assert_allclose(volumes.sum()*self.config['density_kg_m3'],masses[i],rtol=2e-4)
            np.testing.assert_allclose(moments.sum()-moments,inertias[i],rtol=2e-4)
            edges={}
            for face in faces:
                for a,b in zip(face,face[1:]+face[:1]):
                    edge=tuple(sorted((a,b)));edges[edge]=edges.get(edge,0)+1
            self.assertTrue(all(count==2 for count in edges.values()))

    def test_friction_never_generates_power(self):
        rng=np.random.default_rng(0)
        for velocity in rng.normal(size=(100,3))*10:
            force,_,_=friction(velocity,self.config)
            self.assertLessEqual(force@velocity,1e-12)

    def test_geometrically_valid_stationary_motion_can_fail_dynamics(self):
        q=inverse_kinematics(.65,self.config)
        self.assertLess(np.linalg.norm(closure(q,self.config)),1e-12)
        mass,coriolis,gravity,_,_=mass_bias_energy(q,np.zeros(3),self.config)
        jacobian=closure_jacobian(q,self.config)
        null=np.cross(jacobian[0],jacobian[1]);null/=np.linalg.norm(null)
        # Even an arbitrary closure reaction cannot cancel force along this tangent.
        residual=gravity-np.array([2.,0.,0.])
        self.assertGreater(abs(null@residual),.1)

    def test_dataset_torque_does_not_encode_future_velocity(self):
        for t in np.linspace(0,4,100):
            self.assertEqual(drive_torque(t,-100,self.config),drive_torque(t,100,self.config))


if __name__=='__main__':unittest.main()
