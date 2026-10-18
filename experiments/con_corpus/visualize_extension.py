"""Scientific diagnostic visuals from saved arrays; fixed first test case, seed42."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype":42,"ps.fonttype":42})
from PIL import Image,ImageDraw,ImageFont
from scipy.integrate import cumulative_trapezoid
from experiments.keycld_corpus.physics_score import MODULES,reference
ROOT=Path(__file__).resolve().parents[2];EXT=ROOT/'output/con_corpus/dynamics_extension';DEST=EXT/'figures'

def main():
    DEST.mkdir(exist_ok=True);read=lambda p:json.loads(p.read_text())
    manifest=read(ROOT/'output/keycld_corpus/dataset_manifest.json');record=next(r for r in manifest['records'] if r['split']=='test')
    images={}
    for model in ['keycld','con']:
        with np.load(EXT/'video_diagnosis'/f'{model}_seed42'/'example.npz') as z:images[model]={k:z[k] for k in z.files}
    assert np.array_equal(images['keycld']['target'],images['con']['target'])
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    animation=[]
    for j in range(0,87,2):
        canvas=Image.new('RGB',(746,496),'white');draw=ImageDraw.Draw(canvas)
        draw.text((18,10),f'{record["case_id"]} | seed42 | frame {33+j}',font=font,fill='black')
        for col,title in enumerate(['Ground truth','Framewise reconstruction','Prefix forecast']):draw.text((125+205*col,43),title,font=small,fill='black')
        for row,model in enumerate(['keycld','con']):
            draw.text((12,146+212*row),model.upper(),font=font,fill='black')
            d=images[model];frames=[d['target'][j+3],d['reconstruction'][j+3],d['forecast'][j]]
            for col,frame in enumerate(frames):
                tile=Image.fromarray(frame).resize((192,192),Image.Resampling.NEAREST);canvas.paste(tile,(125+205*col,65+212*row))
        draw.text((18,477),'Reconstruction observes each true frame. Forecast observes only frames 30-32.',font=small,fill='black')
        animation.append(canvas)
    animation[-1].save(DEST/'video_comparison.png')
    animation[0].save(DEST/'video_comparison.gif',save_all=True,append_images=animation[1:],duration=67,loop=0)
    params,truth,t,points=reference(record);c=params['config'];module=MODULES[c['category']];target=params['command_speed_rad_s']
    qt=truth['q'][np.arange(248,953),0];anchor=int(np.argmin(abs(t-2/30)));future=t>=.1-1e-7
    names=[('keycld_video_euler_seed42','KeyCLD video','#aa3377'),('con_video_con_iae_seed42','CON video','#ee7733'),
           ('keycld_state_rk4_seed42','KeyCLD state RK4','#0077bb'),('con_integrator_state_rk4_con_iae_seed42','CON state RK4','#009988')]
    fig,axes=plt.subplots(3,1,figsize=(8,5.4),sharex=True,constrained_layout=True);curve_data={}
    for folder,label,color in names:
        path=EXT/'physics_readout'/folder/(record['case_id']+'.npz')
        with np.load(path) as z:q=z['q'];v=z['qdot'];a=z['qddot']
        force=[];energy=[];power=[]
        for qi,vi,ai in zip(q,v,a):
            J=module.closure_jacobian(qi,c);N=np.r_[1,np.linalg.solve(J[:,1:],-J[:,0])]
            M,C,G,T,V=module.mass_bias_energy(qi,vi,c,params['bodies']);Qf=module.friction(vi,c);Qext=module.external(qi,c)
            actuator=c['actuator'];tau=np.clip(actuator['speed_gain_Nm_s_rad']*(target-vi[0]),-actuator['torque_limit_Nm'],actuator['torque_limit_Nm'])
            force.append(N@(M@ai+C+G-Qf-Qext)-tau);energy.append(T+V);power.append(tau*vi[0]+Qf@vi+Qext@vi)
        force=np.asarray(force);energy=np.asarray(energy);power=np.asarray(power)
        defect=energy[anchor:]-energy[anchor]-cumulative_trapezoid(power[anchor:],t[anchor:],initial=0)
        row=read(path.with_suffix('.json'));m=row['conditioned']['metrics']
        assert np.isclose(np.sqrt(np.mean(force[future]**2)),m['force_balance_Nm'],rtol=1e-6,atol=1e-9)
        assert np.isclose(np.abs(defect[t[anchor:]>=.1-1e-7]).max(),m['work_energy_J'],rtol=1e-6,atol=1e-9)
        phase=np.arctan2(np.sin(q[:,0]-qt),np.cos(q[:,0]-qt))
        axes[0].plot(t+1,phase,label=label,color=color,lw=1.5)
        axes[1].plot(t+1,np.abs(force),color=color,lw=1.5)
        axes[2].plot(t[anchor:]+1,np.abs(defect),color=color,lw=1.5)
        curve_data[folder+'_force_Nm']=force;curve_data[folder+'_energy_defect_J']=defect
    axes[0].set_ylabel('Wrapped phase error (rad)');axes[0].legend(ncol=2,fontsize=9)
    axes[1].set_ylabel('|Torque residual| (N m)');axes[1].axhline(.25,color='.3',ls='--',lw=1,label='0.25 N m');axes[1].set_yscale('symlog',linthresh=.1)
    axes[2].set_ylabel('|Work-energy defect| (J)');axes[2].axhline(.05,color='.3',ls='--',lw=1);axes[2].set_yscale('symlog',linthresh=.02);axes[2].set_xlabel('Absolute video time (s)')
    axes[1].set_ylim(bottom=0);axes[2].set_ylim(bottom=0)
    for ax in axes:
        ax.grid(alpha=.2);ax.axvline(1.1,color='.6',ls=':',lw=1);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Geometry-conditioned readout: '+record['case_id']+' (seed42)',fontsize=11)
    for suffix in ['png','pdf']:fig.savefig(DEST/('dynamics_example.'+suffix),dpi=180)
    plt.close(fig);np.savez_compressed(DEST/'dynamics_example_data.npz',time=t,energy_time=t[anchor:],**curve_data)
    (DEST/'selection.json').write_text(json.dumps(dict(case_id=record['case_id'],seed=42,selection='First test case in frozen manifest, seed42; static image uses final future frame119; animation contains the full future interval',physics_track='Additional geometry-conditioned readout',energy_and_force_reproduced=True),indent=2)+'\n')
    print('Saved video GIF and independent force / energy example curves.')
if __name__=='__main__':main()
