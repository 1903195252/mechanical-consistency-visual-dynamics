"""Figure 3: all recorded physical-readout errors and their unchanged RMSE summary."""
import hashlib
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from prepare_artifacts import ROOT, PAPER, BLUE, ORANGE, save, read

FIELDS = ('encoded_rmse_declared_m', 'encoded_rmse_validation_permutation_m', 'constant_position_rmse_m')
NAMES = ('Declared order', 'Validation mapping', 'Mean pose')
CODES = ('D', 'V', 'M')
COLORS = (BLUE, ORANGE, '#8797A4')


def measurements():
    root = ROOT / 'output/keycld_fourbar/diagnosis'
    rows = sorted(read(root/'learned_rows.json'), key=lambda row: row['index'])
    report = read(root/'report.json')
    indices = np.asarray([row['index'] for row in rows])
    assert np.array_equal(indices, np.arange(500, 550))
    values = 1000*np.asarray([[row[field] for field in FIELDS] for row in rows])
    assert values.shape == (50,3) and np.isfinite(values).all()
    means = np.array([report['summary']['visual_state'][f]['mean']*1000 for f in FIELDS])
    spreads = np.array([report['summary']['visual_state'][f]['std']*1000 for f in FIELDS])
    np.testing.assert_allclose(values.mean(axis=0), means, rtol=1e-12)
    np.testing.assert_allclose(values.std(axis=0, ddof=0), spreads, rtol=1e-12)
    evidence = {
        'source': str((root/'learned_rows.json').relative_to(ROOT)),
        'source_sha256': hashlib.sha256((root/'learned_rows.json').read_bytes()).hexdigest(),
        'summary_source': str((root/'report.json').relative_to(ROOT)),
        'summary_source_sha256': hashlib.sha256((root/'report.json').read_bytes()).hexdigest(),
        'unit': 'mm', 'test_trajectory_ids': indices.tolist(), 'fields': list(FIELDS),
        'per_trajectory_RMSE_mm': values.tolist(), 'mean_trajectory_RMSE_mm': means.tolist(),
        'standard_deviation_mm': spreads.tolist(), 'std_ddof': 0,
        'individual_bars': 150, 'summary_bars': 3, 'test_training_seed': 42,
        'metric': 'Each trajectory: coordinate RMSE over all 50 reference frames, three keypoints and two coordinates.',
        'summary': 'Arithmetic mean of the 50 trajectory RMSEs, with the recorded ddof=0 standard deviation; unchanged from the recorded report.',
        'semantics': 'Framewise visual-state diagnosis receives every reference image. These 50 trajectories are tests of one trained checkpoint.',
        'display': 'Orthographic 3D bars; zero-based common 0--160 mm vertical range; original trajectory order; no omissions.',
    }
    (PAPER/'sources/diagnostic_figure_data.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return root, indices, values, means, spreads


def style_3d(ax, aspect):
    ax.set_proj_type('ortho')
    ax.set_box_aspect(aspect)
    ax.view_init(elev=31, azim=-64)
    ax.set_zlim(0,160)
    ax.set_zticks([0,50,100,150])
    ax.tick_params(axis='both', labelsize=6.6, pad=0)
    ax.tick_params(axis='z', labelsize=6.6, pad=1)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('RMSE (mm)', fontsize=7, rotation=90, labelpad=2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.968,.978,.985,1))
        axis.line.set_color('#A9B6C0'); axis.line.set_linewidth(.5)
        axis._axinfo['grid'].update({'color':'#D9E2E8','linewidth':.45})
        axis._axinfo['tick']['linewidth'] = {True:.5, False:.5}


def build():
    root, ids, values, means, spreads = measurements()
    from prepare_visual_analysis import build as analyze_visual
    analyze_visual()
    fig = plt.figure(figsize=(7.05,2.90))
    fig.text(.031,.977,'(a) Image prediction',fontsize=8.4,va='top')
    fig.text(.290,.977,'(b) Physical readout',fontsize=8.4,va='top')
    fig.text(.031,.453,'(c) Training integrator',fontsize=8.4,va='top')

    plate = fig.add_gridspec(3,3,left=.055,right=.253,bottom=.516,top=.884,hspace=.045,wspace=.035)
    with np.load(root/'visual_example.npz') as data:
        reference, reconstructed, predicted = data['reference_rgb'], data['reconstructed_rgb'], data['predicted_rgb']
    for row, images in enumerate((reference,reconstructed,predicted)):
        for col, frame in enumerate((2,9,49)):
            ax=fig.add_subplot(plate[row,col]); ax.imshow(images[frame],interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values(): spine.set_visible(False)
            if row==0: ax.set_title('%.2f s'%((frame+1)/100),fontsize=6.8,pad=2)
            if col==0: ax.set_ylabel(('Ref.','Recon.','Pred.')[row],fontsize=6.6,labelpad=2)

    euler=read(root/'state_oracle/euler/report.json'); rk4=read(root/'state_oracle/rk4_4/report.json')
    left={row['index']:1000*row['position_rmse_m'] for row in euler['rows']}
    right={row['index']:1000*row['position_rmse_m'] for row in rk4['rows']}
    assert left.keys()==right.keys() and len(left)==50
    xs,ys=np.asarray([[left[i],right[i]] for i in sorted(left)]).T
    limit=np.ceil(max(xs.max(),ys.max())/20)*20
    ax=fig.add_axes([.070,.090,.184,.310])
    ax.plot([0,limit],[0,limit],ls='--',color='#899AA6',lw=.65)
    ax.scatter(xs,ys,s=9,color=BLUE,alpha=.72,edgecolors='white',linewidths=.2)
    ax.scatter(xs.mean(),ys.mean(),marker='D',s=23,color=ORANGE,edgecolors='#263747',linewidths=.4,zorder=4)
    ax.set(xlim=(0,limit),ylim=(0,limit));ax.set_aspect('equal')
    ax.set_xticks(np.linspace(0,limit,3));ax.set_yticks(np.linspace(0,limit,3));ax.tick_params(labelsize=6.4,pad=1)
    ax.set_xlabel('Euler RMSE (mm)',fontsize=6.8,labelpad=1)
    ax.set_ylabel('RK4 RMSE (mm)',fontsize=6.8,labelpad=1)

    handles=[Patch(facecolor=color,edgecolor='none',label=code+': '+name)
             for name,code,color in zip(NAMES,CODES,COLORS)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.286,.913),ncol=3,
               frameon=False,fontsize=6.9,handlelength=1.15,handletextpad=.4,columnspacing=.8,borderaxespad=0)
    fig.text(.487,.791,'All 50 test trajectories',ha='center',fontsize=7.4)
    fig.text(.851,.791,'Mean RMSE ± SD',ha='center',fontsize=7.4)

    cases=fig.add_axes([.281,.088,.423,.694],projection='3d')
    style_3d(cases,(4.0,1.65,2.05))
    # A single collection sorts faces across all 150 bars, preserving 3D occlusion.
    # Low bars in the front row keep all three sets of bar tops visible.
    display_columns = (2,1,0)
    cases.bar3d(np.tile(np.arange(50)-.34,3),np.repeat(np.arange(3)-.25,50),np.zeros(150),
                .68,.50,values[:,display_columns].T.ravel(),
                color=np.repeat(np.asarray(COLORS)[list(display_columns)],50),shade=False,
                edgecolor='#FFFFFF',linewidth=.10,zsort='average')
    cases.view_init(elev=36,azim=-72)
    cases.set_xlim(-1,50);cases.set_ylim(-.55,2.65)
    cases.set_xticks([0,10,20,30,40,49],['500','510','520','530','540','549'])
    cases.set_yticks([0,1,2],['M','V','D'])
    cases.set_xlabel('Test trajectory ID',fontsize=7,labelpad=3)
    cases.set_ylabel('Readout',fontsize=7,labelpad=3)

    summary=fig.add_axes([.711,.123,.273,.652],projection='3d')
    style_3d(summary,(2.65,.90,2.90))
    for i,(mean,spread,color) in enumerate(zip(means,spreads,COLORS)):
        summary.bar3d(i-.27,-.20,0,.54,.40,mean,color=color,shade=False,
                      edgecolor='#FFFFFF',linewidth=.30,zsort='average')
        summary.plot([i,i],[0,0],[mean-spread,mean+spread],color='#263747',lw=.8,zorder=7)
        for z in (mean-spread,mean+spread):
            summary.plot([i-.10,i+.10],[0,0],[z,z],color='#263747',lw=.8,zorder=7)
        summary.text(i,0,mean+spread+5,'%.2f'%mean,ha='center',va='bottom',fontsize=7,color='#263747',zorder=8)
    summary.set_zlabel('')
    summary.set_xlim(-.6,2.6);summary.set_ylim(-.5,.5)
    summary.set_xticks([0,1,2],CODES);summary.set_yticks([])
    summary.set_xlabel('Readout',fontsize=7,labelpad=2)
    fig.text(.31,.024,'All 50 trajectories · common 0–160 mm range',
             fontsize=6.8,color='#536775')
    save(fig,'controlled_diagnostics')


if __name__=='__main__': build()
