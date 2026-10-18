"""Matched planar resolution, image readout, and geometric correction evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from prepare_artifacts import BLUE, ORANGE, GREEN, save
from prepare_resolution_artifacts import collect as resolution_data, REC
from prepare_constitutive_figure import correction_data

PAPER = Path(__file__).resolve().parents[1]
ROOT = PAPER.parents[1]
EXT = ROOT/'output/con_corpus/dynamics_extension'
FRAMES = (33, 119)


def collect():
    resolution = resolution_data()
    selection = json.loads((EXT/'figures/selection.json').read_text())
    sources = [EXT/'figures/selection.json', ROOT/'output/planar_resolution/256/dataset_manifest.json']
    for model, arm in [('keycld','video_euler'),('con','video_con_iae')]:
        sources += [EXT/'video_diagnosis'/f'{model}_seed42/example.npz',
                    ROOT/f'output/{model}_corpus/runs/{arm}_seed42/predictions/test/{selection["case_id"]}.npz',
                    REC/f'geek/figure_inputs/{model}_256_seed42.npz', REC/f'geek/figure_inputs/{model}_256_seed42.json']
    return dict(case_id=selection['case_id'], seed=42, frames=list(FRAMES),
        selection='First frozen test case, seed 42; first and last future frame at both resolutions, independent of scores.',
        overlay='Native B,C,P after the declared frozen readout; dashed reference joints; identical world crop at both resolutions.',
        resolution_runs=[dict(model=r['model'],width=r['width'],visual=r['visual'],native_score=r['native']['score'],
                             correction_mm=r['correction_mm']) for r in resolution['runs']],
        paired_reader='256px; same reader and phase derivative on correct/generated RGB; 108 unique references, 324 forecasts/model.',
        geometric_correction=correction_data(),
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],
        resolution_sources=resolution['sources'])


def style(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(length=2,pad=2,labelsize=6)
    ax.grid(axis='x',color='#E0E5E9',lw=.4); ax.set_axisbelow(True)


def check_text_separation(fig, axes, labels):
    """Reject collisions in the two compact quantitative panels at export size."""
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    artists=list(labels)
    for ax in axes:
        artists.extend([ax.title,ax.xaxis.label,*ax.texts,*ax.get_xticklabels(),*ax.get_yticklabels()])
    artists=[a for a in artists if a.get_visible() and a.get_text()]
    for i,a in enumerate(artists):
        aa=a.get_window_extent(renderer)
        for b in artists[i+1:]:
            bb=b.get_window_extent(renderer)
            overlap_x=min(aa.x1,bb.x1)-max(aa.x0,bb.x0)
            overlap_y=min(aa.y1,bb.y1)-max(aa.y0,bb.y0)
            assert min(overlap_x,overlap_y)<.25, (a.get_text(),b.get_text(),overlap_x,overlap_y)


def build():
    data=resolution_data(); evidence=collect(); case=evidence['case_id']
    manifest=json.loads((ROOT/'output/planar_resolution/256/dataset_manifest.json').read_text())
    record=next(r for r in manifest['records'] if r['case_id']==case)
    cfg=json.loads((ROOT/record['physical_parameters']).read_text())['config']
    anchors=np.array([[0,cfg['pivot_height_m']],[cfg['ground_length_m'],cfg['pivot_height_m']]])
    target256=np.load(ROOT/'output/planar_resolution/256'/record['images'],mmap_mode='r')
    images={}; joints={}
    truth=None
    for model,arm in [('keycld','video_euler'),('con','video_con_iae')]:
        with np.load(EXT/'video_diagnosis'/f'{model}_seed42/example.npz') as z:
            images[(model,64)]=z['forecast'][np.array(FRAMES)-33]
            target64=z['target'][np.array(FRAMES)-30]
            truth=z['truth_points'][np.array(FRAMES)-30]
        for width in (64,256):
            if width==64:
                path=ROOT/f'output/{model}_corpus/runs/{arm}_seed42/predictions/test/{case}.npz'
                permutation=json.loads((ROOT/f'output/{model}_corpus/runs/{arm}_seed42/scores.json').read_text()).get('reader_permutation_from_validation',[0,1,2])
            else:
                path=REC/f'geek/figure_inputs/{model}_256_seed42.npz'
                meta=json.loads(path.with_suffix('.json').read_text())
                assert hashlib.sha256(path.read_bytes()).hexdigest()==meta['prediction_sha256']
                scores=json.loads((REC/f'comparison/source_reports/{model}_256_seed42/scores.json').read_text())
                permutation=scores['permutation']
            with np.load(path) as z:
                indices=[int(np.argmin(abs(z['time']-(frame/30-1)))) for frame in FRAMES]
                np.testing.assert_allclose(z['time'][indices],np.array(FRAMES)/30-1,atol=1e-6)
                joints[(model,width)]=z['points'][indices][:,permutation]
                if width==256: images[(model,width)]=z['rgb_prediction'][np.array(FRAMES)-33]
    points=np.vstack([truth.reshape(-1,2),anchors]+[v.reshape(-1,2) for v in joints.values()])
    low,high=points.min(0)-.06,points.max(0)+.06; center=(low+high)/2; half=max(high-low)/2
    fig=plt.figure(figsize=(7.05,2.70))
    fig.text(.006,.99,'(a) Matched frames and native joints',fontsize=7.5,weight='bold',va='top')
    gs=fig.add_gridspec(3,4,left=.061,right=.474,bottom=.18,top=.824,hspace=.05,wspace=.055)
    for col in range(4):
        width=(64,256)[col//2];frame=FRAMES[col%2]
        for row,label in enumerate(('Reference','KeyCLD','CON')):
            ax=fig.add_subplot(gs[row,col]);mi=('keycld','con')[row-1]
            rgb=(target64[col%2] if width==64 else target256[frame-30]) if row==0 else images[(mi,width)][col%2]
            half_pixel=.65/(width-1)
            ax.imshow(rgb,extent=(-.45-half_pixel,.85+half_pixel,-.3-half_pixel,1.+half_pixel),interpolation='nearest',origin='upper')
            def skeleton(pts,color,ls):
                chain=np.vstack([anchors[0],pts[0],pts[1],anchors[1]])
                ax.plot(chain[:,0],chain[:,1],ls,color=color,lw=.7,alpha=.9)
                ax.scatter(pts[:,0],pts[:,1],s=5,color=color,edgecolor='white',linewidth=.2,zorder=5)
            skeleton(truth[col%2],GREEN,'--')
            if row:skeleton(joints[(mi,width)][col%2],(BLUE,ORANGE)[row-1],'-')
            ax.set(xlim=(center[0]-half,center[0]+half),ylim=(center[1]-half,center[1]+half),xticks=[],yticks=[])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0 and col%2==0:
                # Same metric scale at both resolutions, calibrated to the world crop.
                bar_width=.2/(2*half)
                ax.plot([.08,.08+bar_width],[.92,.92],transform=ax.transAxes,color='white',lw=1.1)
                ax.text(.08+bar_width/2,.88,'0.2 m',transform=ax.transAxes,ha='center',va='top',fontsize=5.5,color='white')
            if col==0: ax.set_ylabel(label,fontsize=6.7,labelpad=3)
            if row==0: ax.set_title(f'{frame/30:.2f} s',fontsize=6.3,pad=2)
    fig.text(.163,.907,'64 × 64',ha='center',fontsize=7.2,weight='bold')
    fig.text(.374,.907,'256 × 256',ha='center',fontsize=7.2,weight='bold')
    fig.add_artist(Line2D([.268,.268],[.17,.93],transform=fig.transFigure,lw=.45,color='#BDC7CD'))
    handles=[Line2D([],[],color=GREEN,ls='--',lw=.8,label='Reference'),Line2D([],[],color=BLUE,lw=.8,label='KeyCLD (K)'),Line2D([],[],color=ORANGE,lw=.8,label='CON (C)')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.02,.055),ncol=3,frameon=False,fontsize=6.1,handlelength=1.1,columnspacing=.7,handletextpad=.35)

    compact_axes=[]; compact_labels=[]
    compact_labels.append(fig.text(.501,.99,'(b) Image MSE / persistence',fontsize=7.1,weight='bold',va='top'))
    ax=fig.add_axes([.572,.59,.168,.33]); compact_axes.append(ax)
    for i,r in enumerate(data['runs']):
        color=BLUE if r['model']=='KeyCLD' else ORANGE; y=3-i
        for value,offset,hatch in [(r['visual']['rgb_hold_ratio'],.23,None),
                                   (r['visual']['motion_hold_ratio'],-.23,'////')]:
            ax.barh(y+offset,value,height=.29,facecolor=color if hatch is None else 'white',
                    edgecolor=color,hatch=hatch,lw=.7,zorder=3)
            ax.text(value+.035,y+offset,f'{value:.2f}',va='center',fontsize=6.2,color='#20272D')
    ax.axvline(1,color='#4B555D',ls='--',lw=.8,zorder=2)
    ax.set(xlim=(0,1.33),ylim=(-.55,3.65),xticks=[0,.5,1],yticks=[3,2,1,0],
           yticklabels=['KeyCLD 64','KeyCLD 256','CON 64','CON 256'])
    style(ax)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=3,labelsize=6.1)
    legend=fig.legend(handles=[Patch(facecolor='#505D68',edgecolor='#505D68',label='RGB'),
                              Patch(facecolor='white',edgecolor='#505D68',hatch='////',label='motion mask')],
        loc='center',bbox_to_anchor=(.625,.514),ncol=2,frameon=False,fontsize=6.1,
        handlelength=1.2,handleheight=.75,handletextpad=.45,columnspacing=.9)
    compact_labels.extend(legend.get_texts())

    compact_labels.append(fig.text(.501,.465,'(c) Matched RGB · 256 px',fontsize=7.2,weight='bold',va='top'))
    for left,metric,title,tol,limits in [(.527,'dynamics','Torque (N m)',.25,(.01,25)),(.652,'energy','Energy (J)',.05,(.005,60))]:
        ax=fig.add_axes([left,.13,.096,.23]); compact_axes.append(ax); vals=[]
        for model,source in [('KeyCLD','reference'),('KeyCLD','generated'),('CON','generated')]:
            vals.append(data['reader'][f'{model}/{source}/phase_sg7d3']['means'][metric])
        ax.axhspan(limits[0],tol,color=GREEN,alpha=.08,zorder=0)
        ax.bar(range(3),vals,color=['#637782',BLUE,ORANGE],width=.60,zorder=3)
        ax.axhline(tol,color='#51785A',ls='--',lw=.8,zorder=4)
        for x,v in enumerate(vals):
            label=f'{v:.1f}' if v>=10 else (f'{v:.2f}' if v>=1 else f'{v:.3f}')
            # Keep labels above the target line as well as the bar; a dashed
            # tolerance must never strike through the reference annotation.
            ax.text(x,max(v*1.23,tol*1.3),label,ha='center',va='bottom',fontsize=5.9)
        ax.set(yscale='log',ylim=limits,xlim=(-.55,2.55),xticks=[0,1,2],xticklabels=['Ref\nRGB','K','C'])
        ax.set_title(title,fontsize=6.2,pad=4);style(ax);ax.grid(False)
        ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.set_yticks([tol,1,10]);ax.set_yticklabels([f'{tol:g}','1','10'],fontsize=5.7)
        ax.get_yticklabels()[0].set_color('#51785A')

    fig.text(.77,.99,'(d) Geometric anchoring',fontsize=7.1,weight='bold',va='top')
    def boxes(ax, values, means, positions, colors):
        for vv,mean,pos,color in zip(values,means,positions,colors):
            bp=ax.boxplot([vv],positions=[pos],vert=False,widths=.32,patch_artist=True,whis=(0,100),showfliers=False,manage_ticks=False)
            bp['boxes'][0].set(facecolor=color,edgecolor=color,alpha=.25,lw=.7)
            for key in ('whiskers','caps','medians'):
                for line in bp[key]:line.set(color=color,lw=.7)
            ax.scatter(mean,pos,s=11,marker='D',color=color,zorder=4)
            label=f'{mean:.1f}' if mean>=.001 else '<1 µm'
            ax.text(mean,pos+.22,label,ha='center',fontsize=5.7,color=color)
    ax=fig.add_axes([.823,.595,.166,.29])
    boxes(ax,[r['corrections_mm'] for r in data['runs']],[r['correction_mm'] for r in data['runs']],
          [3,2,1,0],[BLUE,BLUE,ORANGE,ORANGE])
    ax.set(xlim=(0,240),ylim=(-.5,3.65),yticks=[3,2,1,0],yticklabels=['K 64','K 256','C 64','C 256'],xticks=[0,100,200])
    style(ax);ax.set_xlabel('Video correction (mm)',fontsize=6.1,labelpad=2)
    fig.text(.77,.48,'Shared exact-state controls',fontsize=6.5,weight='bold',va='top')
    ax=fig.add_axes([.823,.15,.166,.255])
    values=[];means=[]
    for group in evidence['geometric_correction']['groups']:
        values+=group['values_mm'][1:];means+=group['balanced_means_mm'][1:]
    boxes(ax,values,means,[3,2,1,0],[BLUE,BLUE,ORANGE,ORANGE])
    ax.set(xscale='log',xlim=(1e-4,550),ylim=(-.5,3.7),yticks=[3,2,1,0],yticklabels=['K Euler','K RK4','C Euler','C RK4'],xticks=[1e-4,.01,1,100])
    ax.set_xticklabels([r'$10^{-4}$',r'$10^{-2}$','1',r'$10^2$'])
    style(ax);ax.set_xlabel('State correction (mm, log)',fontsize=6.1,labelpad=2)
    for x in (.486,.755):fig.add_artist(Line2D([x,x],[.02,.96],transform=fig.transFigure,lw=.45,color='#D9E0E5'))
    check_text_separation(fig,compact_axes,compact_labels)
    save(fig,'shared_video_audit')
    (PAPER/'sources/shared_video_figure_data.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return {'shared_video_audit':'Matched 64/256 frames, normalized image errors, paired RGB mechanics, and geometric correction distributions.'}


if __name__=='__main__':
    additions=build();path=PAPER/'sources/evidence_map.json';mapping=json.loads(path.read_text());mapping.update(additions)
    path.write_text(json.dumps(mapping,indent=2)+'\n')
