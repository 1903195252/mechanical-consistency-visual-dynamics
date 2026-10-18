"""The instrument must preserve real motion and never impose future closure."""
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from .calibrate_spatial_reset import align
from .optical_measurement_refinement import fuse


class InstrumentOptimizationTests(unittest.TestCase):
    def test_reset_calibration_keeps_future_relative_motion(self):
        q=np.zeros((4,10));q[:,:3]=np.array([1.,2.,3.])+np.arange(4)[:,None]*[.01,-.02,.03]
        rotations=Rotation.from_rotvec(np.column_stack((np.arange(4)*.01,np.zeros(4),np.full(4,.1))))
        q[:,3:7]=rotations.as_quat()[:,[3,0,1,2]]
        q[:,7:]=np.arange(12).reshape(4,3)/100
        original=q.copy()
        known=np.r_[[.1,.2,.3],[1.,0.,0.,0.],[.5,.6,.7]]
        corrected=align(q,q[0],known)
        np.testing.assert_allclose(corrected[0,np.r_[0:3,7:10]],known[np.r_[0:3,7:10]],atol=1e-12)
        # q and -q encode the same orientation; no sign convention is imposed
        # by scipy's rotation composition or required by the SI measurement.
        np.testing.assert_allclose(Rotation.from_quat(corrected[0,[4,5,6,3]]).as_matrix(),
                                   Rotation.from_quat(known[[4,5,6,3]]).as_matrix(),atol=1e-12)
        np.testing.assert_allclose(np.diff(corrected[:,:3],axis=0),np.diff(q[:,:3],axis=0),atol=1e-12)
        np.testing.assert_allclose(np.diff(corrected[:,7:],axis=0),np.diff(q[:,7:],axis=0),atol=1e-12)
        before=Rotation.from_quat(q[:,[4,5,6,3]])
        after=Rotation.from_quat(corrected[:,[4,5,6,3]])
        np.testing.assert_allclose((before[0].inv()*before).as_matrix(),(after[0].inv()*after).as_matrix(),atol=1e-12)
        np.testing.assert_array_equal(q,original)
        self.assertGreater(np.linalg.norm(corrected[-1]-known),.01)

    def test_optical_fusion_preserves_rapid_correctly_measured_motion(self):
        t=np.arange(90)/30
        motion=np.stack((np.sin(2*np.pi*6*t)*.01,np.cos(2*np.pi*3*t)*.02),axis=-1)[:,None,:]
        increment=np.diff(motion,axis=0)
        np.testing.assert_allclose(fuse(motion,increment,np.ones((89,1)),64),motion,atol=1e-12)
        np.testing.assert_allclose(fuse(motion,np.ones_like(increment),np.zeros((89,1)),64),motion,atol=1e-12)


if __name__=="__main__":unittest.main()
