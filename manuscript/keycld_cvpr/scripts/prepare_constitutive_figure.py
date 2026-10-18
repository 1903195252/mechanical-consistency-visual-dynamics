"""Structural probes for Fig. 5 and geometric correction for Fig. 3(d)."""
import hashlib
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
from prepare_artifacts import ROOT, PAPER, BLUE, ORANGE, GREEN, read
from prepare_rendered_figures import export

DATA = ROOT/'output/keycld_corpus'
FAMILIES = ('S-A','S-B','S-J')
NAMES = ('Crank-rocker','Crank-slider','Double-crank')
COLORS = (BLUE,ORANGE,GREEN)
INSET_BOUNDS = ((.65,.13,.28,.23),(.56,.12,.34,.38))
CALLOUT_COLOR = '#8DA0AC'


def diagnostic_frame(parent, inset, renderer):
    """Frame the complete diagnostic, including its title and tick labels."""
    bounds = inset.get_tightbbox(renderer).padded(1.5 * parent.figure.dpi / 72)
    box = bounds.transformed(parent.transAxes.inverted())
    parent.add_patch(Rectangle((box.x0, box.y0), box.width, box.height,
                               transform=parent.transAxes, facecolor='white',
                               edgecolor=CALLOUT_COLOR, lw=.6, zorder=4, clip_on=False))
    return box


def connect_diagnostic(parent, source, source_transform, target):
    parent.add_artist(ConnectionPatch(xyA=source, coordsA=source_transform,
                                      xyB=target, coordsB=parent.transAxes,
                                      color=CALLOUT_COLOR, lw=.5, zorder=3,
                                      clip_on=False))


def collect():
    sources=[]; probes=[]
    for arm in ('state_euler','state_rk4'):
        for seed in (42,43,44):
            path=DATA/'runs'/('%s_seed%d'%(arm,seed))/'constitutive_probe.json'
            p=read(path);assert p['sampled_states']==648 and p['maximum_learned_odd_acceleration']==0
            valid=[r['inertia_relative_error'] for r in p['rows'] if r['positive_gain']]
            assert len(valid)==p['positive_action_gain_states']
            np.testing.assert_allclose(np.median(valid),p['effective_inertia_relative_error_median'])
            np.testing.assert_allclose(np.mean(valid),p['effective_inertia_relative_error_mean'])
            probes.append(p);sources.append({'path':str(path.relative_to(ROOT)),
                                            'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    probe_source=ROOT/'experiments/keycld_corpus/constitutive_probe.py'
    digest=hashlib.sha256(probe_source.read_bytes()).hexdigest()
    assert all(p['probe_source_sha256']==digest for p in probes)
    assert 'valid=np.isfinite(gain)&(gain>1e-8)' in probe_source.read_text()
    sources.append({'path':str(probe_source.relative_to(ROOT)),'sha256':digest})
    reference=probes[0]['rows']
    identities=[(r['case_id'],r['pose_index']) for r in reference]
    for p in probes:
        assert [(r['case_id'],r['pose_index']) for r in p['rows']]==identities
        assert all(r['learned_friction_odd_acceleration_rad_s2']==0 for r in p['rows'])
    friction=np.abs([r['true_effective_inertia_kg_m2']*r['true_friction_odd_acceleration_rad_s2'] for r in reference])
    np.testing.assert_allclose(friction,[r['friction_equivalent_torque_error_Nm'] for r in reference])
    np.testing.assert_allclose(np.sqrt(np.mean(friction**2)),probes[0]['friction_torque_error_rms_Nm'])
    path=DATA/'kinetic_capacity_audit.json'; capacity=read(path)
    sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    mass=np.asarray(capacity['shared_oracle_effective_point_masses_kg'])
    true=np.concatenate([r['target_metric_kg_m2'] for r in capacity['rows']])
    fitted=np.vstack([r['features'] for r in capacity['rows']])@mass
    categories=np.concatenate([[r['category']]*r['poses'] for r in capacity['rows']])
    assert true.shape==fitted.shape==(44280,)
    shared_error=float(np.sqrt(np.mean((fitted/true-1)**2)))
    np.testing.assert_allclose(shared_error,capacity['shared_mass_metric_relative_rms'])
    max_individual=max(r['best_positive_mass_metric_relative_rms'] for r in capacity['rows'])
    torque_path=ROOT/'output/keycld_fourbar/extended_dynamics/field_probes.json'
    torque=read(torque_path)
    correction_path=ROOT/'output/con_corpus/dynamics_extension/physical_case_metrics.csv'
    summary_path=ROOT/'output/con_corpus/dynamics_extension/summary.json'
    for path in (torque_path,correction_path,summary_path):
        sources.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    pooled={}
    for arm in ('state_euler','state_rk4'):
        rr=[r for p in probes if p['arm']==arm for r in p['rows']]
        ratios=[r['estimated_effective_inertia_kg_m2']/r['true_effective_inertia_kg_m2'] for r in rr if r['positive_gain']]
        pooled[arm]={'total':len(rr),'positive':len(ratios),'unusable_gain_percent':100*(len(rr)-len(ratios))/len(rr),
                     'positive_inertia_ratio_median':float(np.median(ratios))}
    evidence={'sources':sources,'unique_query_states':648,'checkpoints':6,
        'all_learned_velocity_odd_responses_zero':True,
        'reference_friction_equivalent_torque_RMS_Nm':float(np.sqrt(np.mean(friction**2))),
        'positive_gain_counts':[p['positive_action_gain_states'] for p in probes],
        'inertia_relative_error_medians':[p['effective_inertia_relative_error_median'] for p in probes],
        'inertia_relative_error_means':[p['effective_inertia_relative_error_mean'] for p in probes],
        'capacity_assemblies':360,'capacity_conditions':1080,'capacity_poses':44280,
        'shared_point_mass_relative_RMS':shared_error,'maximum_per_condition_relative_RMS':max_individual,
        'family_shared_relative_RMS':{family:float(np.sqrt(np.mean((fitted[categories==family]/true[categories==family]-1)**2))) for family in FAMILIES},
        'relative_RMS_definition':'sqrt(mean((I_fit/I_reference - 1)^2)), equally weighted recorded poses',
        'scope':'Figure 5 panels a,b are frozen state-component interventions. Panel c is a reference-only kinetic representation fit. Figure 3(d) compares the cost of the same geometric-readout intervention on both models.',
        'layout':{'constitutive_evidence':{'panels':3,'rows':1,'columns':3},
                  'geometric_readout_cost':{'destination':'Figure 3(d), Section 5.3','panels':1}},
        'sampling':'All query states and capacity poses retained; no score-based selection or outlier removal.',
        'actuator_panel':'Pooled usable-response inertia-ratio ECDFs with a separate failed-gain-check bar; 1944 queries per arm, no discarded failures.',
        'gain_check':'Usable iff finite and k_tau > 1e-8. False flags include near-zero and nonfinite gains; they do not establish exclusively wrong-sign responses.',
        'pooled_response':pooled,
        'inset_layout':{'axes_bounds':[list(bounds) for bounds in INSET_BOUNDS],
                        'style':'Inset frames remain inside the parent panels; only panel b uses grey association lines',
                        'association_semantics':{
                            'a':'The zero-response point band is associated with the separate 100 Hz friction diagnostic; the inset is not a coordinate zoom.',
                            'b':'The entire response panel is associated with the complementary failed-gain fraction; failures have no assigned inertia-ratio coordinates.'},
                        'render_clearance':'Full inset and label bounds avoid main traces, ideal guides, panel titles, and odd-response points'},
        'correction_panel':'Native-to-geometric-readout coordinate RMS, all finite cases; box whiskers retain min/max, diamonds are balanced means. This is the specified intervention cost, not nearest-manifold distance.'}
    return probes,reference,friction,true,fitted,categories,evidence


def correction_data():
    """Frozen distributions and balanced means for the planar intervention."""
    ext=ROOT/'output/con_corpus/dynamics_extension'
    with (ext/'physical_case_metrics.csv').open() as stream: rows=list(csv.DictReader(stream))
    summary=read(ext/'summary.json')
    groups=[]
    for model,color,offset,arms in [('keycld',BLUE,.14,('video_euler','state_euler','state_rk4')),
                                  ('con',ORANGE,-.14,('video_con_iae','state_euler_con_iae','state_rk4_con_iae'))]:
        values=[];means=[]
        for arm in arms:
            group='con_integrator' if model=='con' and arm.startswith('state_') else model
            vals=[float(r['correction_rms_m'])*1000 for r in rows if r['model']==group and r['arm']==arm and r['track']=='raw' and r['correction_rms_m']]
            values.append(vals);means.append(summary['physics'][group+'/'+arm]['correction_rms_m']['mean']*1000)
        groups.append(dict(model=model,arms=list(arms),values_mm=values,balanced_means_mm=means))
    return dict(groups=groups,statistics='Quartiles and full-range whiskers over finite cases; diamonds are seed/family/condition-balanced means.',
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                 for p in (ext/'physical_case_metrics.csv',ext/'summary.json')])


def draw_geometric_correction(ax, data=None, compact=False):
    data=correction_data() if data is None else data
    for group,color,offset in zip(data['groups'],(BLUE,ORANGE),(.14,-.14)):
        values,means=group['values_mm'],group['balanced_means_mm']
        positions=np.array([2,1,0])+offset
        bp=ax.boxplot(values,positions=positions,vert=False,widths=.22,patch_artist=True,whis=(0,100),showfliers=True,manage_ticks=False)
        for box in bp['boxes']:box.set(facecolor=color,alpha=.32,edgecolor=color,lw=.7)
        for key in ('whiskers','caps','medians'):
            for line in bp[key]:line.set(color=color,lw=.7)
        ax.scatter(means,positions,color=color,marker='D',s=13,zorder=4)
    ax.set_xscale('log');ax.set_xlim(1e-5,1e3);ax.set_xticks([1e-4,.01,1,100]);ax.set_xticklabels(['0.0001','0.01','1','100'])
    ax.set_yticks([2,1,0],['Video','State–Euler','State–RK4'],fontsize=6.8);ax.set_ylim(-.5,2.6)
    ax.set_xlabel('Correction RMS (mm)' if compact else 'Geometric-readout correction RMS (mm)',fontsize=6.5 if compact else 7,labelpad=2)
    ax.tick_params(axis='x',labelsize=6 if compact else 7,length=2,pad=2)
    ax.legend(handles=[Patch(facecolor=BLUE,alpha=.5,label='KeyCLD'),Patch(facecolor=ORANGE,alpha=.5,label='CON')],
              loc='upper left',bbox_to_anchor=(0,1.04),ncol=2,frameon=False,fontsize=6.5,handlelength=.8,columnspacing=.9,
              title='Model',title_fontsize=6)
    ax.grid(which='major',color='#E2E8ED',lw=.45);ax.set_axisbelow(True)


def geometric_correction():
    """Standalone export of the same panel used inside Figure 3."""
    fig,ax=plt.subplots(figsize=(3.45,1.74))
    draw_geometric_correction(ax)
    ax.set_title('Cost of anchoring physical geometry',loc='left',fontsize=8,pad=5)
    fig.subplots_adjust(left=.235,right=.99,top=.84,bottom=.23)
    export(fig,'geometric_readout_cost',dpi=360)


def build():
    probes,reference,friction,true,fitted,categories,evidence=collect()
    fig,axes=plt.subplots(1,3,figsize=(7.05,2.35))
    ax=axes[0]
    signed=np.array([r['true_effective_inertia_kg_m2']*r['true_friction_odd_acceleration_rad_s2'] for r in reference])*1000
    bound=max(abs(signed))*1.08
    ax.plot([-bound,bound],[-bound,bound],'--',color='#7D8991',lw=.85,label='Ideal')
    for family,name,color in zip(FAMILIES,NAMES,COLORS):
        chosen=np.array([r['assembly_id'].startswith(family) for r in reference])
        ax.scatter(signed[chosen],np.zeros(chosen.sum()),s=8,color=color,alpha=.45,lw=0,label=name,zorder=3)
    ax.set(xlim=(-bound,bound),ylim=(-bound,bound),xticks=[-90,0,90],yticks=[-90,0,90])
    ax.set_xlabel('Reference odd torque (mN m)',fontsize=7,labelpad=1)
    ax.set_ylabel('Learned odd torque (mN m)',fontsize=7,labelpad=1)
    ax.set_title('(a) Missing odd-velocity\ndissipation',loc='left',fontsize=8,pad=9)
    ax.text(.045,.94,'648 states · 6 checkpoints\nZero odd response',transform=ax.transAxes,
            ha='left',va='top',fontsize=5.7,linespacing=1.2)
    # The independent 100 Hz diagnostic occupies the empty lower-right region.
    inset=ax.inset_axes(INSET_BOUNDS[0])
    p=read(ROOT/'output/keycld_fourbar/extended_dynamics/field_probes.json')
    vals=[p['conditions'][n]['friction']['true_odd_acceleration_rms_rad_s2'] for n in ('viscous','coulomb','mixed_friction')]
    assert all(p['conditions'][n]['friction']['learned_odd_acceleration_max_abs_rad_s2']==0
               for n in ('viscous','coulomb','mixed_friction'))
    inset.barh([2,1,0],vals,height=.48,color='#C1D0D9',edgecolor='#637B8B',lw=.5,zorder=2)
    inset.scatter(vals,[2,1,0],color='#637B8B',s=10,zorder=4)
    inset.scatter([0,0,0],[2,1,0],color=BLUE,marker='x',s=11,zorder=4)
    inset.set_xscale('symlog',linthresh=1)
    inset.set(xlim=(-.25,300),ylim=(-.5,2.5),yticks=[2,1,0],yticklabels=['Visc.','Coul.','Mix.'],xticks=[0,10,200])
    inset.set_xticklabels(['0','10','200'])
    inset.tick_params(labelsize=5.4,pad=1,length=1.5)
    inset.set_title('100 Hz · rad/s²',fontsize=5.4,pad=2)
    evidence['odd_acceleration_inset']={'reference_RMS_rad_s2':vals,'learned_RMS_rad_s2':[0,0,0],
                                      'conditions':['viscous','coulomb','mixed_friction'],
                                      'encoding':'Reference horizontal bars and endpoint circles; model crosses at zero',
                                      'axis_scale':'symlog','linear_threshold_rad_s2':1}
    odd_inset = inset

    ax=axes[1]
    # The records retain the finite-and-positive gain check and usable inertia estimates.
    # Failed checks are not necessarily negative gains; do not fabricate their signs or magnitudes.
    ratios=[]
    for arm,color,label in [('state_euler','#8797A4','Euler'),('state_rk4',BLUE,'RK4')]:
        rr=[r for p in probes if p['arm']==arm for r in p['rows']]
        v=np.sort([r['estimated_effective_inertia_kg_m2']/r['true_effective_inertia_kg_m2'] for r in rr if r['positive_gain']])
        ratios.append(v)
        ax.step(v,100*np.arange(1,len(v)+1)/len(v),where='post',color=color,lw=1.3,label=label)
    ax.set_xscale('log');ax.set_xlim(.8,1e5);ax.set_ylim(0,105);ax.set_yticks([0,50,100]);ax.set_xticks([1,100,10000])
    ax.set_xticklabels(['1×','100×','10,000×']);ax.axvline(1,color='#536775',ls=':',lw=.8)
    ax.set_xlabel('Usable-response inertia ratio\n'+r'$\widehat{\mathcal{I}}_{resp}/\mathcal{I}$',fontsize=7,labelpad=1)
    ax.set_ylabel('Usable responses (%)',fontsize=7,labelpad=1)
    ax.set_title('(b) Actuator gain\nand scale',loc='left',fontsize=8,pad=9)
    ax.legend(loc='upper left',frameon=False,fontsize=6.5)
    # The upper-left inset used to cover the rising ECDFs. This region is empty.
    inset=ax.inset_axes(INSET_BOUNDS[1]);inset.set_facecolor('white')
    wrong=[evidence['pooled_response'][arm]['unusable_gain_percent'] for arm in ('state_euler','state_rk4')]
    inset.barh([1,0],wrong,color=['#8797A4',BLUE],height=.5)
    for y,v in zip([1,0],wrong):inset.text(v+1,y,'%.1f%%'%v,fontsize=6,va='center')
    inset.set_xlim(0,55);inset.set_yticks([1,0],['E','R'],fontsize=6);inset.set_xticks([])
    inset.set_title('Fails gain check',fontsize=6,pad=2)
    inset.tick_params(length=0,pad=2)
    gain_inset = inset

    ax=axes[2]
    for family,color in zip(FAMILIES,COLORS):
        chosen=categories==family
        ax.scatter(true[chosen],fitted[chosen],s=1.1,color=color,alpha=.18,edgecolors='none',rasterized=True,zorder=2)
    limits=(6e-4,.25);ax.plot(limits,limits,'--',color='#516776',lw=.7,zorder=3)
    ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits,xticks=[.001,.01,.1],yticks=[.001,.01,.1])
    ax.set_xlabel(r'Reference $\mathcal{I}$ (kg m$^2$)',fontsize=7,labelpad=1)
    ax.set_ylabel(r'Shared-fit $\mathcal{I}$ (kg m$^2$)',fontsize=7,labelpad=1)
    ax.set_title('(c) Theoretical three-point\nmass capacity',loc='left',fontsize=8,pad=9)
    capacity_fit_inset=ax.inset_axes([.23,.70,.24,.23],zorder=5)
    errors=100*np.array([evidence['maximum_per_condition_relative_RMS'],evidence['shared_point_mass_relative_RMS']])
    capacity_fit_inset.barh([1,0],errors,height=.48,color=['#8797A4','#526B7B'])
    capacity_fit_inset.scatter(errors,[1,0],s=8,color=['#8797A4','#526B7B'],zorder=3)
    capacity_fit_inset.set(xlim=(0,25),ylim=(-.6,1.6),xticks=[0,25],yticks=[1,0])
    capacity_fit_inset.set_yticklabels(['Separate','Shared'])
    capacity_fit_inset.set_title('RMS (%)',fontsize=5.5,pad=2)
    capacity_fit_inset.tick_params(labelsize=5.2,pad=1,length=1.5)

    capacity_family_inset=ax.inset_axes([.67,.12,.29,.27],zorder=5)
    family_errors=[100*evidence['family_shared_relative_RMS'][family] for family in FAMILIES]
    capacity_family_inset.bar(range(3),family_errors,color=COLORS,width=.62)
    capacity_family_inset.set(ylim=(0,35),xticks=range(3),yticks=[0,30])
    capacity_family_inset.set_xticklabels(['A','B','J'])
    capacity_family_inset.set_title('Shared RMS (%)',fontsize=5.5,pad=2)
    capacity_family_inset.tick_params(labelsize=5.2,pad=1,length=1.5)
    evidence['capacity_insets']={'fit_comparison':{'axes_bounds':[.23,.70,.24,.23],
        'separate_statistic':'Maximum relative RMS over all 1080 separate condition fits',
        'shared_statistic':'Pooled relative RMS over all 44280 poses',
        'values_percent':errors.tolist()},
        'shared_by_family':{'axes_bounds':[.67,.12,.29,.27],
        'families':list(FAMILIES),'values_percent':family_errors}}
    for ax in axes.flat:ax.grid(which='major',color='#E2E8ED',lw=.45);ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([],[],color=c,lw=1.5,label=n) for c,n in zip(COLORS,NAMES)],
               loc='upper center',bbox_to_anchor=(.49,1.015),ncol=3,frameon=False,fontsize=6.5)
    fig.subplots_adjust(left=.07,right=.99,top=.73,bottom=.25,wspace=.40)
    all_insets=(odd_inset,gain_inset,capacity_fit_inset,capacity_family_inset)
    for inset in all_insets:
        for side in ('left', 'bottom'):
            inset.spines[side].set_color(CALLOUT_COLOR)
            inset.spines[side].set_linewidth(.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    odd_frame = diagnostic_frame(axes[0], odd_inset, renderer)
    gain_frame = diagnostic_frame(axes[1], gain_inset, renderer)
    assert 0<=odd_frame.x0<odd_frame.x1<=1 and 0<=odd_frame.y0<odd_frame.y1<=1, ('Odd-response diagnostic must stay inside panel a',odd_frame.bounds)
    assert 0<=gain_frame.x0<gain_frame.x1<=1 and 0<=gain_frame.y0<gain_frame.y1<=1, 'Gain diagnostic must stay inside panel b'
    capacity_frames=[diagnostic_frame(axes[2],inset,renderer)
                     for inset in (capacity_fit_inset,capacity_family_inset)]
    # Related diagnostics retain their own units and populations. These links
    # associate views; neither inset is a magnification of parent coordinates.
    left, right = float(signed.min()-3), float(signed.max()+3)
    axes[0].add_patch(Rectangle((left,-5),right-left,10,fill=False,
                                 edgecolor=CALLOUT_COLOR,lw=.6,zorder=4))
    # This is a second experiment, so its title identifies it without zoom links.
    axes[1].add_patch(Rectangle((0,0),1,1,transform=axes[1].transAxes,
                                 fill=False,edgecolor=CALLOUT_COLOR,lw=.6,zorder=4))
    for y,target_y in ((0,gain_frame.y0),(1,gain_frame.y1)):
        connect_diagnostic(axes[1],(1,y),axes[1].transAxes,(gain_frame.x1,target_y))
    evidence['inset_layout']['parent_frames'] = {
        'a': {'coordinate_system':'data','bounds':[left,-5,right-left,10]},
        'b': {'coordinate_system':'axes','bounds':[0,0,1,1]}}
    evidence['inset_layout']['diagnostic_frame_bounds'] = [
        list(box.bounds) for box in (odd_frame,gain_frame)]
    for ax,inset in zip((axes[0],axes[1],axes[2],axes[2]),all_insets):
        bounds = inset.get_tightbbox(renderer).padded(1.5*fig.dpi/72)
        for line in ax.lines:
            path = line.get_transform().transform_path(line.get_path())
            if path.intersects_bbox(bounds,filled=False):
                fig.savefig(PAPER/'build/constitutive_evidence.png',dpi=210,bbox_inches='tight')
                raise AssertionError('Inset obscures a main trace: '+ax.get_title(loc='left')+
                                     '; frame='+str(bounds.transformed(ax.transAxes.inverted()).bounds))
        assert not ax._left_title.get_window_extent(renderer).overlaps(bounds), 'Inset obscures a panel title'
    zero_points = axes[0].transData.transform(np.column_stack((signed,np.zeros(len(signed)))))
    bounds = odd_inset.get_tightbbox(renderer).padded(1.5*fig.dpi/72)
    assert not any(bounds.contains(x,y) for x,y in zero_points), 'Inset obscures odd-response points'
    capacity_points=axes[2].transData.transform(np.column_stack((true,fitted)))
    for inset in (capacity_fit_inset,capacity_family_inset):
        bounds=inset.get_tightbbox(renderer).padded(1.5*fig.dpi/72)
        assert not np.any((capacity_points[:,0]>=bounds.x0)&(capacity_points[:,0]<=bounds.x1)&
                          (capacity_points[:,1]>=bounds.y0)&(capacity_points[:,1]<=bounds.y1)), 'Inset obscures capacity points'
    evidence['capacity_insets']['diagnostic_frame_bounds']=[list(box.bounds) for box in capacity_frames]
    export(fig,'constitutive_evidence',dpi=360)
    geometric_correction()
    (PAPER/'sources/constitutive_figure_data.json').write_text(json.dumps(evidence,indent=2)+'\n')


if __name__=='__main__':build()
