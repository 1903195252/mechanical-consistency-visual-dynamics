"""Independent rigid-tree kinematics, mass matrix and Newton--Euler bias.

MuJoCo is used only to import a model's static calibration. Evaluation uses
NumPy and the explicit joint graph, never MuJoCo Jacobians or inverse dynamics.
Free/ball angular velocities follow the child-local convention used by MuJoCo.
"""
import numpy as np


def skew(x):
    a,b,c=x
    return np.array([[0.,-c,b],[c,0.,-a],[-b,a,0.]])


def quaternion_matrix(q):
    q=np.asarray(q,dtype=float);q=q/np.linalg.norm(q);w,x,y,z=q
    return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],
                     [2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],
                     [2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])


def axis_rotation(axis,angle):
    axis=np.asarray(axis,dtype=float);axis=axis/np.linalg.norm(axis);K=skew(axis)
    return np.eye(3)+np.sin(angle)*K+(1-np.cos(angle))*(K@K)


class SpatialTree:
    def __init__(self,specification):
        self.specification=specification;self.nv=specification['nv'];self.nq=specification['nq']
        self.armature=np.asarray(specification['armature']);self.gravity=np.asarray(specification['gravity'])
        self.bodies=specification['bodies']

    @classmethod
    def from_mujoco(cls,model):
        import mujoco
        bodies=[];kinds={0:'free',1:'ball',2:'slide',3:'hinge'}
        for i in range(1,model.nbody):
            Ri=quaternion_matrix(model.body_iquat[i]);I=Ri@np.diag(model.body_inertia[i])@Ri.T;joints=[]
            first=int(model.body_jntadr[i]);count=int(model.body_jntnum[i])
            for j in range(first,first+count):
                kind=kinds[int(model.jnt_type[j])];qa=int(model.jnt_qposadr[j]);da=int(model.jnt_dofadr[j]);nq={'free':7,'ball':4,'slide':1,'hinge':1}[kind]
                joints.append({'kind':kind,'qpos_address':qa,'dof_address':da,'reference':model.qpos0[qa:qa+nq].tolist(),
                               'position':model.jnt_pos[j].tolist(),'axis':model.jnt_axis[j].tolist(),
                               'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j)})
            bodies.append({'id':i,'name':mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i) or str(i),
                           'parent':int(model.body_parentid[i]),'position':model.body_pos[i].tolist(),
                           'rotation':quaternion_matrix(model.body_quat[i]).tolist(),'mass':float(model.body_mass[i]),
                           'com':model.body_ipos[i].tolist(),'inertia':I.tolist(),'joints':joints})
        return cls({'nv':model.nv,'nq':model.nq,'gravity':model.opt.gravity.tolist(),'armature':model.dof_armature.tolist(),'bodies':bodies})

    def forward(self,q,v):
        q=np.asarray(q);v=np.asarray(v);n=self.nv
        if q.shape!=(self.nq,) or v.shape!=(n,):raise ValueError('State dimensions do not match rigid tree')
        states={0:{'position':np.zeros(3),'rotation':np.eye(3),'J_origin':np.zeros((3,n)),
                   'J_rotation':np.zeros((3,n)),'velocity':np.zeros(3),'omega':np.zeros(3),
                   'origin_acceleration_bias':np.zeros(3),'angular_acceleration_bias':np.zeros(3)}}
        for body in self.bodies:
            parent=states[body['parent']];offset=parent['rotation']@body['position']
            p=parent['position']+offset;R=parent['rotation']@body['rotation']
            Jw=parent['J_rotation'].copy();Jp=parent['J_origin']-skew(offset)@Jw
            w=parent['omega'].copy();alpha=parent['angular_acceleration_bias'].copy()
            velocity=parent['velocity']+np.cross(w,offset)
            acceleration=parent['origin_acceleration_bias']+np.cross(alpha,offset)+np.cross(w,np.cross(w,offset))
            for joint in body['joints']:
                kind=joint['kind'];qa=joint['qpos_address'];da=joint['dof_address']
                if kind=='free':
                    if body['parent']!=0 or len(body['joints'])!=1:raise ValueError('Free joint must be a root body')
                    p=q[qa:qa+3].copy();R=quaternion_matrix(q[qa+3:qa+7]);Jp=np.zeros((3,n));Jw=np.zeros((3,n))
                    Jp[:,da:da+3]=np.eye(3);Jw[:,da+3:da+6]=R
                    velocity=v[da:da+3].copy();w=R@v[da+3:da+6];acceleration=np.zeros(3);alpha=np.zeros(3)
                    continue
                if kind=='slide':
                    axis=R@joint['axis'];r=axis*(q[qa]-joint['reference'][0])
                    p=p+r;Jp=Jp-skew(r)@Jw;Jp[:,da]+=axis
                    velocity=velocity+np.cross(w,r)+axis*v[da]
                    acceleration=acceleration+np.cross(alpha,r)+np.cross(w,np.cross(w,r))+2*np.cross(w,axis*v[da])
                    continue
                local_anchor=np.asarray(joint['position']);r=R@local_anchor
                anchor=p+r;Janchor=Jp-skew(r)@Jw
                vanchor=velocity+np.cross(w,r)
                aanchor=acceleration+np.cross(alpha,r)+np.cross(w,np.cross(w,r))
                if kind=='hinge':
                    axis=R@joint['axis'];Rnew=R@axis_rotation(joint['axis'],q[qa]-joint['reference'][0])
                    added_w=axis*v[da];Jwnew=Jw.copy();Jwnew[:,da]+=axis
                elif kind=='ball':
                    reference=quaternion_matrix(joint['reference']);Rnew=R@reference.T@quaternion_matrix(q[qa:qa+4])
                    added_w=Rnew@v[da:da+3];Jwnew=Jw.copy();Jwnew[:,da:da+3]+=Rnew
                else:raise ValueError('Unknown joint kind')
                anew=alpha+np.cross(w,added_w);wnew=w+added_w;back=-Rnew@local_anchor
                p=anchor+back;Jp=Janchor-skew(back)@Jwnew
                velocity=vanchor+np.cross(wnew,back)
                acceleration=aanchor+np.cross(anew,back)+np.cross(wnew,np.cross(wnew,back))
                R,Jw,w,alpha=Rnew,Jwnew,wnew,anew
            rc=R@body['com'];Jcom=Jp-skew(rc)@Jw
            acom=acceleration+np.cross(alpha,rc)+np.cross(w,np.cross(w,rc))
            states[body['id']]={'position':p,'rotation':R,'J_origin':Jp,'J_rotation':Jw,'J_com':Jcom,
                               'com':p+rc,'velocity':velocity,'omega':w,'origin_acceleration_bias':acceleration,
                               'com_acceleration_bias':acom,'angular_acceleration_bias':alpha,'name':body['name']}
        return states

    def use_body_calibration(self,calibration):
        """Use independently aggregated physical part moments for dynamics."""
        for body in self.bodies:
            b=calibration[body['name']]
            body['mass']=float(b['mass_kg']);body['com']=list(b['com_local_m'])
            body['inertia']=np.asarray(b['inertia_com_local_kg_m2']).tolist()
        self.specification['inertial_source']='Independent physical part inventory'
        return self

    def dynamics(self,q,v,states=None):
        states=self.forward(q,v) if states is None else states;M=np.diag(self.armature.copy());bias=np.zeros(self.nv);V=0.
        for body in self.bodies:
            k=states[body['id']];mass=body['mass'];I=k['rotation']@body['inertia']@k['rotation'].T
            Jv,Jw=k['J_com'],k['J_rotation']
            M+=mass*Jv.T@Jv+Jw.T@I@Jw
            bias+=Jv.T@(mass*(k['com_acceleration_bias']-self.gravity))
            bias+=Jw.T@(I@k['angular_acceleration_bias']+np.cross(k['omega'],I@k['omega']))
            V-=mass*self.gravity@k['com']
        return M,bias,float(.5*np.asarray(v)@M@v),float(V)

    @staticmethod
    def point(state,local):
        r=state['rotation']@np.asarray(local)
        return state['position']+r,state['J_origin']-skew(r)@state['J_rotation']
