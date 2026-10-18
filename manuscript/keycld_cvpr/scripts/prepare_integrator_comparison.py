"""Plot both paired training experiments in the established KeyCLD figure style."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
import sys

import numpy as np

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
SOURCE = ROOT / 'output/con_corpus/dynamics_extension/physical_case_metrics.csv'
FAMILIES = ('S-A', 'S-B', 'S-J')
NAMES = ('Crank-rocker', 'Crank-slider', 'Double-crank')
METRICS = ('prediction_m', 'force_balance_Nm', 'work_energy_J')
SCATTER_LIMITS = ((.55, 470), (.06, 140), (.01, 1600))
ZOOM_SPECS = (
    ((65,125),(3,14),((70,90),(5,10)),(.04,.66,.36,.30)),
    ((1.3,3.6),(.22,.65),((1.5,3),(.3,.6)),(.04,.54,.36,.34)),
    ((5,80),(.2,3),((10,50),(.3,2)),(.04,.65,.36,.31)),
    ((55,100),(55,115),((60,90),(60,100)),(.04,.54,.36,.34)),
    ((3.4,5.4),(3.3,5.5),((4,5),(4,5)),(.04,.54,.36,.34)),
    ((5,16),(5,20),((5,15),(8,15)),(.04,.68,.36,.28)),
)
EXT = ROOT/'output/con_corpus/dynamics_extension'
RECOVERED = ROOT/'output/planar_resolution/cloud/recovered_20261017/geek/expanded/output/planar_resolution/runs'
TRACE_CONFIGS = (
    ('KeyCLD', 64, 'keycld_video_euler_seed42', ROOT/'output/keycld_corpus/runs/video_euler_seed42'),
    ('KeyCLD', 256, 'keycld_256_seed42', RECOVERED/'keycld_256_seed42'),
    ('KeyCLD', None, 'keycld_state_rk4_seed42', ROOT/'output/keycld_corpus/runs/state_rk4_seed42'),
    ('CON', 64, 'con_video_con_iae_seed42', ROOT/'output/con_corpus/runs/video_con_iae_seed42'),
    ('CON', 256, 'con_256_seed42', RECOVERED/'con_256_seed42'),
    ('CON', None, 'con_integrator_state_rk4_con_iae_seed42', EXT/'runs/state_rk4_con_iae_seed42'),
)
TRACE_COLORS = {
    'KeyCLD': {64: '#5A97BD', 256: '#0072B2', None: '#3B87C4'},
    'CON': {64: '#C18B4C', 256: '#C25100', None: '#E1842B'},
}


def trace_style(model, width):
    """Keep model hues while separating configurations at publication size."""
    return dict(color=TRACE_COLORS[model][width],
                linewidth={64: 1.15, 256: 1.25, None: 1.25}[width],
                dashes={64: [3.2, 2.2], 256: [5.5, 1.7, 1.0, 1.7], None: []}[width],
                marker={'KeyCLD': 'o', 'CON': 's'}[model],
                markersize=2.5, markeredgewidth=.7,
                markerfacecolor=TRACE_COLORS[model][width] if width is None else 'white',
                dash_capstyle='round', solid_capstyle='round')


def collect_traces(case_id):
    """Replay the same SI equations on all six saved, geometry-readout states."""
    from scipy.integrate import cumulative_trapezoid
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))
    from experiments.keycld_corpus.physics_score import MODULES, reference
    from experiments.con_corpus.crank_readout import reconstruct

    manifest_path=ROOT/'output/keycld_corpus/dataset_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    record=next(r for r in manifest['records'] if r['case_id']==case_id)
    params,truth,t,_=reference(record)
    c=params['config'];module=MODULES[c['category']]
    reference_phase=truth['q'][np.arange(248,953),0]
    # Simulator timestamps accumulate about 1.6 ps of FP64 roundoff; the
    # integer 240 Hz sample indices must still agree exactly.
    np.testing.assert_array_equal(np.rint(truth['time'][np.arange(248,953)]*240).astype(int),np.arange(248,953))
    np.testing.assert_allclose(truth['time'][np.arange(248,953)],t+1,atol=1e-9,rtol=0)
    anchor=int(np.argmin(abs(t-2/30)));future=t>=.1-1e-7
    previous_path=EXT/'figures/dynamics_example_data.npz'
    with np.load(previous_path) as z:
        previous={k:z[k] for k in z.files}
    np.testing.assert_array_equal(previous['time'],t)
    np.testing.assert_array_equal(previous['energy_time'],t[anchor:])
    paths=[manifest_path,previous_path,EXT/'figures/selection.json',
           ROOT/record['physics_reference'],ROOT/record['physical_parameters'],
           ROOT/'experiments/con_corpus/crank_readout.py',
           ROOT/'experiments/keycld_corpus/physics_score.py',
           ROOT/'experiments/mechanism_corpus/fourbar.py']
    traces=[]
    for model,width,arm,folder in TRACE_CONFIGS:
        state_path=(folder/'measurements' if width==256 else EXT/'physics_readout'/arm)/(case_id+'.npz')
        prediction_path=folder/'predictions/test'/(case_id+'.npz')
        report=json.loads(state_path.with_suffix('.json').read_text())
        meta=json.loads(prediction_path.with_suffix('.json').read_text())
        assert report['case_id']==case_id and report['identity']['seed']==42
        assert report['conditioned']['coverage']==1
        assert hashlib.sha256(state_path.read_bytes()).hexdigest()==report['readout_prediction_sha256']
        assert hashlib.sha256(prediction_path.read_bytes()).hexdigest()==report['native_prediction_sha256']==meta['prediction_sha256']
        if width==256:
            assert report['identity']['width']==256
        with np.load(state_path) as z:
            np.testing.assert_array_equal(z['time'],t)
            saved={k:z[k] for k in ('points','velocity','acceleration','q','qdot','qddot')}
        with np.load(prediction_path) as z:
            np.testing.assert_allclose(z['time'],t,atol=2e-7,rtol=0)
            raw=[z[k][:,report['identity']['permutation']].copy() for k in ('points','velocity','acceleration')]
        coordinates,state=reconstruct(*raw,c)
        for name,values in zip(('points','velocity','acceleration'),coordinates):
            np.testing.assert_allclose(saved[name],values,rtol=1e-11,atol=1e-12)
        for name,values in state.items():
            np.testing.assert_allclose(saved[name],values,rtol=1e-11,atol=1e-12)
        force=[];energy=[];power=[]
        for q,v,a in zip(saved['q'],saved['qdot'],saved['qddot']):
            J=module.closure_jacobian(q,c)
            tangent=np.r_[1,np.linalg.solve(J[:,1:],-J[:,0])]
            M,C,G,T,V=module.mass_bias_energy(q,v,c,params['bodies'])
            friction=module.friction(v,c);external=module.external(q,c)
            actuator=c['actuator']
            tau=np.clip(actuator['speed_gain_Nm_s_rad']*(params['command_speed_rad_s']-v[0]),
                        -actuator['torque_limit_Nm'],actuator['torque_limit_Nm'])
            force.append(tangent@(M@a+C+G-friction-external)-tau)
            energy.append(T+V);power.append(tau*v[0]+friction@v+external@v)
        force=np.asarray(force);energy=np.asarray(energy);power=np.asarray(power)
        defect=energy[anchor:]-energy[anchor]-cumulative_trapezoid(power[anchor:],t[anchor:],initial=0)
        phase=np.arctan2(np.sin(saved['q'][:,0]-reference_phase),np.cos(saved['q'][:,0]-reference_phase))
        metrics=report['conditioned']['metrics']
        torque_rms=float(np.sqrt(np.mean(force[future]**2)))
        energy_max=float(np.abs(defect[t[anchor:]>=.1-1e-7]).max())
        np.testing.assert_allclose([torque_rms,energy_max],
            [metrics['force_balance_Nm'],metrics['work_energy_J']],rtol=1e-8,atol=1e-10)
        if width!=256:
            np.testing.assert_allclose(force,previous[arm+'_force_Nm'],rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(defect,previous[arm+'_energy_defect_J'],rtol=1e-12,atol=1e-12)
        label=model+(' State–RK4' if width is None else f' {width} px')
        traces.append(dict(model=model,width=width,arm=arm,label=label,
            linestyle={64:'--',256:'-.',None:'-'}[width],
            style=trace_style(model,width),
            marker_times_s=np.arange({64:1.35,256:1.60,None:1.475}[width],4,.6).tolist(),
            phase_rad=phase.tolist(),force_Nm=force.tolist(),energy_defect_J=defect.tolist(),
            future_torque_rms_Nm=torque_rms,future_energy_max_J=energy_max,
            saved_case_metrics_verified=True,common_reconstruction_verified=True,
            original_curve_preserved=width!=256))
        paths.extend([state_path,state_path.with_suffix('.json'),prediction_path,prediction_path.with_suffix('.json')])
    protocol=dict(absolute_time_s=(t+1).tolist(),energy_time_s=(t[anchor:]+1).tolist(),
        forecast_start_s=1.1,energy_anchor_s=float(t[anchor]+1),future_samples=int(future.sum()),
        plotted_state_samples=len(t),plotted_energy_samples=len(t[anchor:]),
        phase_definition='atan2(sin(predicted crank angle minus reference), cos(same difference))',
        reconstruction='Predicted crank direction and native derivatives with public geometry; identical map for all six curves.',
        force_and_energy='Independent SI evaluator; controller reevaluated at each predicted speed; trapezoidal work from the last observed frame.')
    sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]
    return traces,protocol,sources


def collect():
    with SOURCE.open() as stream:
        rows = list(csv.DictReader(stream))
    summary_path = EXT/'summary.json'
    summary = json.loads(summary_path.read_text())
    panels = []
    for model, label, arms in (
        ('keycld', 'KeyCLD', ('state_euler', 'state_rk4')),
        ('con_integrator', 'CON', ('state_euler_con_iae', 'state_rk4_con_iae')),
    ):
        for metric in METRICS:
            track = 'raw' if metric == 'prediction_m' else 'conditioned'
            selected = [[r for r in rows if r['model'] == model and r['arm'] == arm and r['track'] == track]
                        for arm in arms]
            lookup = [{(r['seed'], r['case_id']): r for r in group} for group in selected]
            assert all(len(group) == len(index) == 324 for group, index in zip(selected, lookup))
            assert lookup[0].keys() == lookup[1].keys()
            pairs, unavailable = [], []
            scale = 1000 if metric == 'prediction_m' else 1
            for identity in sorted(lookup[0]):
                a, b = [index[identity] for index in lookup]
                if not a[metric] or not b[metric]:
                    unavailable.append({'seed': int(identity[0]), 'case_id': identity[1]})
                    continue
                x, y = float(a[metric]) * scale, float(b[metric]) * scale
                assert math.isfinite(x) and math.isfinite(y) and min(x, y) > 0
                if metric != 'prediction_m':
                    assert a['coverage'] == b['coverage'] == '1.0'
                pairs.append({'seed': int(identity[0]), 'case_id': identity[1], 'family': identity[1][:3],
                              'euler': x, 'rk4': y})
            expected = 315 if model == 'con_integrator' and track == 'conditioned' else 324
            assert len(pairs) == expected and len(pairs) + len(unavailable) == 324
            # Preserve the paper's equal seed/family/condition weighting.
            cells = {}
            for pair in pairs:
                key = (pair['seed'], pair['family'], pair['case_id'].split('__')[1])
                cells.setdefault(key, []).append(pair)
            means = [mean(mean(p[arm] for p in cell) for cell in cells.values())
                     for arm in ('euler', 'rk4')]
            stats = {'euler_mean': means[0], 'rk4_mean': means[1],
                     'change_percent': 100*(means[1]/means[0]-1)}
            if model == 'con_integrator' and track == 'conditioned':
                paired = summary['con_rk4_minus_euler'][track+'/'+metric]
                assert math.isclose(means[1]-means[0], paired['mean'], abs_tol=1e-12)
                stats.update(delta=paired['mean'], bootstrap_95pct=paired['bootstrap_95pct'])
            else:
                for arm, value in zip(arms, means):
                    expected_mean = summary['physics'][model+'/'+arm][track]['metrics'][metric]['mean']*scale
                    assert math.isclose(value, expected_mean, rel_tol=1e-12)
            panels.append({'model': label, 'metric': metric, 'track': track, 'registered': 324,
                           'unit': 'mm' if metric == 'prediction_m' else 'N m' if metric == 'force_balance_Nm' else 'J',
                           'pairs': pairs, 'unavailable_pairs': unavailable, 'statistics': stats})
            column = METRICS.index(metric)
            xlim,ylim,ticks,bounds = ZOOM_SPECS[len(panels)-1]
            lo,hi = map(math.log10, SCATTER_LIMITS[column])
            for pair in pairs:
                x,y = [(math.log10(pair[key])-lo)/(hi-lo) for key in ('euler','rk4')]
                assert not (bounds[0]<x<bounds[0]+bounds[2] and bounds[1]<y<bounds[1]+bounds[3]), \
                    'Inset would hide a parent scatter point'
            panels[-1]['zoom'] = {
                'xlim':list(xlim),'ylim':list(ylim),'axes_bounds':list(bounds),
                'xticks':list(ticks[0]),'yticks':list(ticks[1]),
                'scale':'log x and y, as in parent','parent_points_obscured':0,
                'render_clearance':'Inset axes and individual tick-label boxes checked with 1.5 pt marker margin',
                'pairs':[{'seed':p['seed'],'case_id':p['case_id']} for p in pairs
                         if xlim[0]<=p['euler']<=xlim[1] and ylim[0]<=p['rk4']<=ylim[1]]}
            panels[-1]['density'] = {
                'samples':len(pairs),'method':'Gaussian kernel sum, pooled over families',
                'coordinates':'log10 x and y','normalized_axis_bandwidth':[.045,.055],
                'relative_peak_levels':[.08,.24,.50,.80,1.001],
                'xlim':list(SCATTER_LIMITS[column]),'ylim':list(SCATTER_LIMITS[column]),
                'grid':[120,120]}
    selection = json.loads((EXT/'figures/selection.json').read_text())
    traces,trace_protocol,trace_sources=collect_traces(selection['case_id'])
    return {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            'summary_source': str(summary_path.relative_to(ROOT)),
            'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
            'annotation_aggregation': summary['aggregation'],
            'pass_region': {'scatter': 'RK4 criterion only (y <= tolerance)',
                            'thresholds': {'point_mm': 10, 'torque_Nm': .25, 'energy_J': .05},
                            'energy_scatter_axis_min_J': .01},
            'trace_case': selection['case_id'], 'trace_seed': 42,
            'trace_sources': trace_sources, 'trace_protocol':trace_protocol, 'trace_curves':traces,
            'panels': panels, 'family_colors': ['#0072B2', '#D55E00', '#009E73'], 'family_markers': ['o', 's', '^'],
            'font': 'DejaVu Sans', 'no_outlier_removal': True,
            'interpretation': 'Two within-model paired Euler/RK4 training experiments. Both models use native point error and the common geometric force/energy readout. KeyCLD uses 43 four-step windows per sequence; CON uses one 48-step rollout. These are different learning protocols, not an equal-compute architecture comparison.'}


def build():
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle, ConnectionPatch
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    from prepare_artifacts import BLUE, ORANGE, GREEN, save
    from prepare_rendered_figures import density_background

    evidence = collect()
    fig = plt.figure(figsize=(7.05, 4.75))
    height_pt = 4.75 * 72
    # Fixed physical panel sizes leave separate rows for axis labels, the
    # model legend, and trace titles without reducing any text or inset size.
    axes = np.array([[fig.add_axes([left, bottom/height_pt, .245, 66/height_pt]) for left in (.075,.417,.755)]
                     for bottom in (245,133)])
    names = ('Point', 'Torque', 'Energy')
    limits = SCATTER_LIMITS
    thresholds = (10, .25, .05)
    ticks = ((1, 10, 100), (.1, 1, 10, 100), (.01, 1, 100))
    pass_color, tolerance_color = '#2ca02c', '#3F6650'
    inset_checks = []
    for i, (ax, panel) in enumerate(zip(axes.flat, evidence['panels'])):
        column = i % 3
        density = density_background(ax,panel['pairs'],'euler','rk4',limits[column],limits[column],
                                     log_x=True,log_y=True)
        assert density == panel['density']
        ax.axhspan(limits[column][0], thresholds[column], color=pass_color, alpha=.10, lw=0, zorder=-1)
        for family, name, color, marker in zip(FAMILIES, NAMES, (BLUE, ORANGE, GREEN), ('o', 's', '^')):
            selected = [p for p in panel['pairs'] if p['family'] == family]
            x, y = [p['euler'] for p in selected], [p['rk4'] for p in selected]
            assert all(limits[column][0] < v < limits[column][1] for v in x + y)
            ax.scatter(x, y, s=8, c=color, marker=marker, alpha=.5, lw=0, label=name, zorder=3)
        ax.plot(limits[column], limits[column], '--', lw=.85, color='#7D8991', zorder=0)
        ax.axhline(thresholds[column], ls='--', color=tolerance_color, lw=.9, zorder=2)
        ax.axvline(thresholds[column], ls='--', color=tolerance_color, lw=.9, zorder=2)
        ax.set(xscale='log', yscale='log', xlim=limits[column], ylim=limits[column],
               xticks=ticks[column], yticks=ticks[column])
        ax.set_xlabel('Euler (%s)' % panel['unit'], fontsize=7, labelpad=1)
        ax.set_ylabel('RK4 (%s)' % panel['unit'], fontsize=7, labelpad=1)
        ax.set_title('(%s) %s: %s (n=%d)' % (chr(97+i),panel['model'],names[column].lower(),len(panel['pairs'])),
                     loc='left',pad=12,fontsize=7.5)
        stats = panel['statistics']
        if i < 4:
            digits = (('.0f', '.1f'), ('.2f', '.3f'), ('.1f', '.2f'), ('.0f', '.0f'))[i]
            values = [format(stats[k], precision) for k, precision in zip(('euler_mean', 'rk4_mean'), digits)]
            change = ('%+.0f%%' % stats['change_percent']).replace('-', '\N{MINUS SIGN}')
            annotation = 'Mean: %s → %s %s (%s)' % (*values, panel['unit'], change)
        else:
            lo, hi = stats['bootstrap_95pct']
            assert lo < 0 < hi
            annotation = 'Δ = %+.2f %s (95%% CI spans 0)' % (stats['delta'], panel['unit'])
        ax.text(0,1.02,annotation,transform=ax.transAxes,ha='left',va='bottom',
                fontsize=6.05,color='#243747',clip_on=False)
        ax.grid(which='major', color='#E5E9EC', lw=.45)
        if 'zoom' in panel:
            # Boxed clusters retain the parent log scales; all parent points stay visible.
            detail = panel['zoom']
            zoom_xlim,zoom_ylim = detail['xlim'],detail['ylim']
            zoom_ticks = (detail['xticks'],detail['yticks'])
            zoom_box = detail['axes_bounds']
            zoom = ax.inset_axes(zoom_box, zorder=6)
            for family,color,marker in zip(FAMILIES,(BLUE,ORANGE,GREEN),('o','s','^')):
                selected = [p for p in panel['pairs'] if p['family']==family]
                zoom.scatter([p['euler'] for p in selected],[p['rk4'] for p in selected],
                             s=5,c=color,marker=marker,alpha=.6,lw=0,zorder=3)
            zoom.plot(limits[column],limits[column],'--',color='#7D8991',lw=.6,zorder=1)
            zoom.set(xscale='log',yscale='log',xlim=zoom_xlim,ylim=zoom_ylim)
            for axis,positions in zip((zoom.xaxis,zoom.yaxis),zoom_ticks):
                axis.set_major_locator(FixedLocator(positions))
                axis.set_major_formatter(FuncFormatter(lambda value,pos: '%g'%value))
                axis.set_minor_locator(NullLocator())
            zoom.tick_params(labelsize=5.2,length=2,pad=1)
            zoom.yaxis.tick_right()
            zoom.grid(color='#E5E9EC',lw=.35);zoom.set_axisbelow(True)
            for spine in zoom.spines.values():
                spine.set_visible(True);spine.set_color('#718895');spine.set_linewidth(.6)
            ax.add_patch(Rectangle((zoom_xlim[0],zoom_ylim[0]),np.ptp(zoom_xlim),np.ptp(zoom_ylim),
                                   fill=False,edgecolor='#718895',lw=.7,zorder=4))
            for y,corner in zip(zoom_ylim,[(1,0),(1,1)]):
                ax.add_artist(ConnectionPatch(xyA=(zoom_xlim[0],y),coordsA=ax.transData,
                                              xyB=corner,coordsB=zoom.transAxes,
                                              color='#718895',lw=.55,zorder=2))
            inset_checks.append((ax,zoom,panel))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Patch(facecolor=pass_color, alpha=.10, edgecolor='none'))
    labels.append('Within tolerance')
    handles.append(Patch(facecolor='#DCE9F2',edgecolor='#B7CBD9',lw=.4))
    labels.append('Point density')
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False, fontsize=6.5,
               bbox_to_anchor=(.51, 1.015), markerscale=1.4,handlelength=1.2,columnspacing=1.1)
    trace_axes = [fig.add_axes([left,23/height_pt,.245,56/height_pt]) for left in (.075,.417,.755)]
    trace_protocol=evidence['trace_protocol']
    for curve in evidence['trace_curves']:
        for ax,x,y in zip(trace_axes,
                (trace_protocol['absolute_time_s'],trace_protocol['absolute_time_s'],trace_protocol['energy_time_s']),
                (curve['phase_rad'],np.abs(curve['force_Nm']),np.abs(curve['energy_defect_J']))):
            # Sparse glyphs identify the model without hiding the complete trace.
            markevery=np.searchsorted(x,curve['marker_times_s']).tolist()
            ax.plot(x,y,ls=curve['linestyle'],markevery=markevery,
                    label=curve['label'],**curve['style'])
    trace_axes[0].set_ylim(-3.3,3.3);trace_axes[0].set_yticks([-3,0,3]);trace_axes[0].axhline(0,color='#929CA3',lw=.5)
    for ax,threshold,lin in zip(trace_axes[1:],(.25,.05),(.1,.02)):
        ax.set_yscale('symlog',linthresh=lin);ax.set_ylim(bottom=0)
        ax.axhspan(0,threshold,color=pass_color,alpha=.10,lw=0,zorder=-1)
        ax.axhline(threshold,color=tolerance_color,ls='--',lw=.9,zorder=2)
    for ax,title,ylabel in zip(trace_axes, ('(g) Fixed-case phase','(h) Fixed-case torque','(i) Fixed-case energy'),
                              ('Phase error (rad)',r'$|r_\tau|$ (N m)',r'$|r_E|$ (J)')):
        ax.set_title(title,fontsize=7.7,loc='left',pad=5);ax.set_ylabel(ylabel,fontsize=7,labelpad=1)
        ax.set_xlim(1,4);ax.set_xticks([1,2,3,4]);ax.set_xlabel('Video time (s)',fontsize=7,labelpad=1)
        ax.axvline(1.1,color='#929CA3',ls=':',lw=.6);ax.grid(color='#E5E9EC',lw=.45)
    fig.legend(handles=[Line2D([],[],ls=c['linestyle'],label=c['label'],**c['style'])
                        for c in evidence['trace_curves']],
               loc='center',bbox_to_anchor=(.51,99/height_pt),ncol=6,frameon=False,fontsize=6.6,
               handlelength=3.6,columnspacing=.9,handletextpad=.45)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    legend_bounds=fig.legends[1].get_window_extent(renderer)
    assert 0<=legend_bounds.x0 and legend_bounds.x1<=fig.bbox.width, 'Six-curve legend exceeds the figure width.'
    bands = [[fig.legends[0]], [ax._left_title for ax in axes[0]],
             [ax.xaxis.label for ax in axes[0]], [ax._left_title for ax in axes[1]],
             [ax.xaxis.label for ax in axes[1]], [fig.legends[1]],
             [ax._left_title for ax in trace_axes]]
    for upper,lower in zip(bands,bands[1:]):
        gap = min(a.get_window_extent(renderer).y0 for a in upper) - max(a.get_window_extent(renderer).y1 for a in lower)
        assert gap >= 2*fig.dpi/72, 'Figure 4 text bands need at least two points of separation.'
    for ax,zoom,panel in inset_checks:
        # Include tick labels and a marker-radius margin, not just the white axes box.
        labels = zoom.get_xticklabels()+zoom.get_yticklabels()
        bounds = [zoom.get_window_extent(renderer)]+[
            label.get_window_extent(renderer) for label in labels if label.get_visible()]
        bounds = [box.padded(1.5*fig.dpi/72) for box in bounds]
        points = ax.transData.transform([(p['euler'],p['rk4']) for p in panel['pairs']])
        assert not any(box.contains(x,y) for box in bounds for x,y in points), \
            'Inset labels would hide a parent point: %s %s' % (panel['model'],panel['metric'])
    save(fig, 'corpus_paired_residuals')
    (PAPER/'sources/integrator_comparison_data.json').write_text(json.dumps(evidence, indent=2)+'\n')
    return evidence


if __name__ == '__main__':
    build()
