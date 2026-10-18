"""Double-precision containment for closed oriented nominal solid meshes."""
import numpy as np


def winding_number(point,vertices,triangles):
    rays=np.asarray(vertices,dtype=np.float64)[np.asarray(triangles)]-np.asarray(point,dtype=np.float64)
    a,b,c=rays[:,0],rays[:,1],rays[:,2]
    la,lb,lc=(np.linalg.norm(x,axis=1) for x in [a,b,c])
    numerator=np.einsum('ij,ij->i',a,np.cross(b,c))
    denominator=la*lb*lc+np.einsum('ij,ij->i',a,b)*lc+np.einsum('ij,ij->i',b,c)*la+np.einsum('ij,ij->i',c,a)*lb
    return float(np.sum(2*np.arctan2(numerator,denominator))/(4*np.pi))
