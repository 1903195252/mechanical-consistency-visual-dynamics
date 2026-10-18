"""Official 2D KeyCLD with declared geometry and a calibrated external controller."""
import jax
import jax.numpy as jnp
from keycld.models import KeyCLD


class ControlledKeyCLD(KeyCLD):
    def __init__(self, geometry):
        self.geometry = geometry
        def constraints(x):
            B,C,P = x.reshape(3,2)
            A,D = geometry[:2],geometry[2:4]
            lengths = geometry[4:7]
            third = jnp.where(geometry[8] > .5, C[1]-A[1]-geometry[7], jnp.linalg.norm(C-D)-lengths[2])
            return jnp.r_[jnp.linalg.norm(B-A)-lengths[0], jnp.linalg.norm(C-B)-lengths[1], third, P-(B+C)/2]
        super().__init__(3,3,32,True,constraints,image_size=64)

    def physical_action(self, x, v, command):
        r = x[:2] - self.geometry[:2]
        speed = (r[0]*v[1]-r[1]*v[0])/jnp.maximum(jnp.dot(r,r),1e-12)
        torque = jnp.clip(command[1]*(command[0]-speed),-command[2],command[2])
        return jnp.r_[torque,command[3:5]]

    def euler_lagrange(self, params, x, x_t, action):
        return super().euler_lagrange(params,x,x_t,self.physical_action(x,x_t,action))


def rk4_four(f, initial, times, action):
    def interval(y, pair):
        t,dt = pair
        def step(i,state):
            h=dt/4; now=t+i*h
            k1=f(state,now,action); k2=f(state+h*k1/2,now+h/2,action)
            k3=f(state+h*k2/2,now+h/2,action); k4=f(state+h*k3,now+h,action)
            return state+h*(k1+2*k2+2*k3+k4)/6
        result=jax.lax.fori_loop(0,4,step,y)
        return result,result
    _,rest=jax.lax.scan(interval,initial,(times[:-1],jnp.diff(times)))
    return jnp.concatenate([initial[None],rest])
