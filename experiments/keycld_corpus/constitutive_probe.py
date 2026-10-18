"""Privileged-state SI input-gain and friction probes on completed state controls."""
import json
import os
from pathlib import Path
import pickle

os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
import jax
import jax.numpy as jnp
import numpy as np
from keycld.models import KeyCLD
from .model import ControlledKeyCLD
from .train import write,sha

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/keycld_corpus'


def main():
    # Input arrays are prepared by the independent SI evaluator, not the model.
    with np.load(OUT/'constitutive_probe_inputs.npz') as z:
        arrays={k:z[k] for k in z.files}
    manifest=json.loads((OUT/'constitutive_probe_manifest.json').read_text())
    @jax.jit
    def batch(parameters,x,unit_velocity,g,external):
        def at_state(position,velocity,geometry,force):
            model=ControlledKeyCLD(geometry); r=position[:2]-geometry[:2];square=jnp.dot(r,r)
            def acceleration(torque,speed):
                v=velocity*speed
                # Bypass only the known servo to apply a specified physical motor
                # torque; the complete learned KeyCLD force/mass modules remain.
                a=KeyCLD.euler_lagrange(model,parameters,position,v,jnp.r_[torque,force])
                omega=(r[0]*v[1]-r[1]*v[0])/square
                return (r[0]*a[1]-r[1]*a[0])/square-2*jnp.dot(r,v[:2])*omega/square
            gain=(acceleration(.25,0.)-acceleration(-.25,0.))/.5
            odd=(acceleration(0.,1.)-acceleration(0.,-1.))/2
            return gain,odd
        return jax.vmap(at_state)(x,unit_velocity,g,external)
    for arm in ('state_euler','state_rk4'):
        for seed in (42,43,44):
            folder=OUT/'runs'/('%s_seed%d'%(arm,seed));path=folder/'constitutive_probe.json'
            if path.exists() or not (folder/'training_report.json').exists():
                continue
            report=json.loads((folder/'training_report.json').read_text())
            assert report['state_supervision'] and report['checkpoint_sha256']==sha(folder/'params.p')
            with (folder/'params.p').open('rb') as f:
                params=pickle.load(f)
            gain,odd=jax.device_get(batch(params,arrays['x'],arrays['unit_velocity'],arrays['geometry'],arrays['external']))
            valid=np.isfinite(gain)&(gain>1e-8);estimated=np.full(len(gain),np.nan);estimated[valid]=1/gain[valid]
            relative=np.abs(estimated-arrays['true_effective_inertia'])/arrays['true_effective_inertia']
            friction_error=(odd-arrays['true_friction_odd_acceleration'])*arrays['true_effective_inertia']
            rows=[]
            for i,item in enumerate(manifest['records']):
                rows.append(dict(item,positive_gain=bool(valid[i]),
                     estimated_effective_inertia_kg_m2=float(estimated[i]) if valid[i] else None,
                     true_effective_inertia_kg_m2=float(arrays['true_effective_inertia'][i]),
                     inertia_relative_error=float(relative[i]) if valid[i] else None,
                     learned_friction_odd_acceleration_rad_s2=float(odd[i]) if np.isfinite(odd[i]) else None,
                     true_friction_odd_acceleration_rad_s2=float(arrays['true_friction_odd_acceleration'][i]),
                     friction_equivalent_torque_error_Nm=float(friction_error[i]) if np.isfinite(friction_error[i]) else None))
            result={'post_hoc_diagnostic':True,'state_supervised_component_only':True,'arm':arm,'seed':seed,
                    'checkpoint_sha256':report['checkpoint_sha256'],'probe_input_sha256':sha(OUT/'constitutive_probe_inputs.npz'),
                    'probe_source_sha256':sha(Path(__file__)),'sampled_states':len(rows),'test_assemblies':36,
                    'positive_action_gain_states':int(valid.sum()),
                    'effective_inertia_relative_error_median':float(np.nanmedian(relative)) if valid.any() else None,
                    'effective_inertia_relative_error_mean':float(np.nanmean(relative)) if valid.any() else None,
                    'friction_torque_error_rms_Nm':float(np.sqrt(np.mean(friction_error**2))) if np.isfinite(friction_error).all() else None,
                    'maximum_learned_odd_acceleration':float(np.max(abs(odd))) if np.isfinite(odd).all() else None,
                    'devices':[str(x) for x in jax.devices()],
                    'interpretation':'SI input-gain behavior, not native point masses in kg. True states are a declared intervention; the video encoder is not scored. Public servo damping is separated from unknown joint friction.',
                    'rows':rows}
            write(path,result);print(arm,seed,result['effective_inertia_relative_error_median'],flush=True)


if __name__=='__main__':
    main()
