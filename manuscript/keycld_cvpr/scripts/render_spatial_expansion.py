"""Compact scientific comparison of all three-seed LingBot forecasts."""
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, NullLocator
from prepare_artifacts import ROOT, PAPER
from prepare_rendered_figures import export
from prepare_spatial_expansion import collect, aggregate, FAMILIES, STEPS, RUNS, RECOVERED

COLORS=('#176B78','#D56E4C')
NAMES=('Stewart','Delta')


def figure():
    from prepare_spatial_artifacts import skeleton, draw_lingbot_overlay, style_axis
    from experiments.model_expansion.evaluate_spatial import Audit
    from PIL import Image
    from prepare_vjepa_expansion import collect as collect_ac
    ac_data=collect_ac()
    d=collect();groups={k:aggregate(v) for k,v in d['groups'].items()}
    fig=plt.figure(figsize=(7.05,2.75))
    for x,y,t in [(.003,.995,'(a) LingBot video and CAD readout'),(.465,.995,'(b) Adaptation: 50 / 0 updates'),
                  (.003,.320,'(c) Point error vs. hold: AC / 0 / 50'),(.465,.660,'(d) Generated vs. reference mechanics')]:
        fig.text(x,y,t,fontsize=7.3,weight='bold',va='top')
    examples=[]
    for j,label in enumerate(('Reference','0 updates','50 updates')):
        fig.text(.087+j*.132,.925,label,ha='center',fontsize=6.3)
    for i,f in enumerate(FAMILIES):
        row=min((r for r in d['tests'].values() if r['category']==f),key=lambda r:r['video_id'])
        vid=row['video_id'];frames=[]
        files=[RECOVERED/'pro/reference/rgb'/f'{vid}.npz']+[
            RECOVERED/f'pro/runs/seed42/step{s:04d}/predictions/{vid}_rgb.npz' for s in STEPS]
        for path in files:
            with np.load(path) as data: frames.append(data['rgb'][119].copy())
        audit=Audit(row)
        folder=RUNS/f'seed42/step0050/readout/predictions/{f}'
        raw_path=folder/f'{vid}.npz';proj_path=folder/f'{vid}_projected.npz'
        with np.load(raw_path) as a:raw=a['qpos'][119].astype(np.float64)
        with np.load(proj_path) as a:proj=a['qpos'][119]
        np.testing.assert_allclose(proj,audit.project(raw),atol=1e-9)
        camera_path=(ROOT/row['video']).with_suffix('.json')
        camera=json.loads(camera_path.read_text())['camera_frames'][119]
        native,_=skeleton(audit,raw);recon,_=skeleton(audit,proj)
        overlay=dict(camera=camera,native={k:v.tolist() for k,v in native.items()},reconstructed={k:v.tolist() for k,v in recon.items()})
        for j,rgb in enumerate(frames):
            asset=PAPER/f'figures/assets/lingbot_{f}_{("reference","zero_shot","adapted")[j]}.png'
            Image.fromarray(rgb).save(asset)
            ax=fig.add_axes([.0365+j*.132,.650-i*.268,.103,.103*7.05/2.75])
            if j==2:draw_lingbot_overlay(ax,rgb,overlay)
            else:ax.imshow(rgb);ax.set(xticks=[],yticks=[])
            for sp in ax.spines.values():sp.set_visible(True);sp.set_linewidth(.35);sp.set_edgecolor('#CBD2D6')
            if j==0:ax.set_ylabel(NAMES[i],fontsize=6.3,color=COLORS[i],labelpad=3)
        examples.append(dict(video_id=vid,seed=42,frame=119,**overlay,
            sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[raw_path,proj_path,camera_path]}))
    fig.legend(handles=[Line2D([],[],color='#D55E00',lw=.8,label='Native'),Line2D([],[],color='#009E73',lw=.8,ls='--',label='CAD / IK')],
        loc='center',bbox_to_anchor=(.215,.350),ncol=2,frameon=False,fontsize=6,handlelength=1.3,columnspacing=1.0)
    reconstruction=[]
    # Full distributions and separate magnifications preserve every outlier.
    from scipy.stats import gaussian_kde
    rng=np.random.default_rng(20261017)
    adapt=[]; zooms=[]; density=[]
    full=fig.add_axes([.525,.730,.170,.175])
    zoom=fig.add_axes([.795,.730,.155,.175])
    for ax in (full,zoom):
        ax.axvline(1,color='#5E6D75',lw=.65,ls='--',zorder=1)
        ax.set(ylim=(-.5,3.5),yticks=[3,2,1,0]);style_axis(ax);ax.grid(visible=False)
        ax.tick_params(labelsize=5.7,pad=1.5)
    full.set(xscale='log',xlim=(.012,8),xticks=[.1,1,5]);full.set_yticklabels(['S RGB','D RGB','S motion','D motion'])
    full.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));full.xaxis.set_minor_locator(NullLocator())
    zoom.set(xlim=(.7,1.3),xticks=[.8,1,1.2],yticklabels=[]);zoom.tick_params(axis='y',length=0)
    full.axvspan(.7,1.3,facecolor='#DCE5EB',alpha=.38,zorder=0)
    full.set_title('Full range (log)',fontsize=5.8,pad=3)
    zoom.set_title('Magnified: 0.7–1.3×',fontsize=5.8,pad=3)
    fig.text(.730,.925,'Mean Δ',fontsize=5.7,ha='center')
    fig.text(.960,.925,'Mean Δ',fontsize=5.7,ha='center')
    for j,(f,k) in enumerate((('H-A','rgb_mse'),('H-B','rgb_mse'),('H-A','motion_mask_mse'),('H-B','motion_mask_mse'))):
        before=[r['visual'][k] for e in d['groups'][f'{f}_0'] for r in e['rows']]
        after=[r['visual'][k] for e in d['groups'][f'{f}_50'] for r in e['rows']]
        ratios=np.array(after)/before;y=3-j;color=COLORS[FAMILIES.index(f)];yy=y+rng.uniform(-.23,.23,len(ratios))
        mask=(ratios>=.7)&(ratios<=1.3)
        full.scatter(ratios,yy,s=1.8,color=color,alpha=.30,linewidths=0,rasterized=True)
        zoom.scatter(ratios[mask],yy[mask],s=1.8,color=color,alpha=.24,linewidths=0,rasterized=True)
        q=np.quantile(ratios,[.25,.5,.75])
        grid=np.linspace(.7,1.3,180);kde=gaussian_kde(ratios,bw_method=.18)(grid);height=.30*kde/kde.max()
        zoom.fill_between(grid,y-height,y+height,color=color,alpha=.12,lw=.4,zorder=0)
        for ax in (full,zoom):
            ax.hlines(y,q[0],q[2],color=color,lw=1.7,zorder=3);ax.plot(q[1],y,'|',color='#26343D',ms=4,mew=.7,zorder=4)
        change=100*(np.mean(after)/np.mean(before)-1)
        full.text(1.04,y,f'{change:+.1f}%',transform=full.get_yaxis_transform(),fontsize=5.9,color=color,va='center')
        zoom.text(1.10,y,f'{change:+.1f}%',transform=zoom.get_yaxis_transform(),fontsize=5.9,color=color,va='center')
        adapt.append(dict(family=f,metric=k,case_ratios=ratios.tolist(),aggregate_percent_change=change))
        zooms.append(dict(panel='b',family=f,metric=k,bounds=[.7,1.3],visible_cases=int(mask.sum()),total_cases=len(ratios)))
    # Both views combine the same mean bars with every visible case.
    full=fig.add_axes([.068,.085,.165,.150]);zoom=fig.add_axes([.280,.085,.175,.150]);point=[]
    point_groups={}
    for f in FAMILIES:
        ac=[r for e in ac_data['groups'][f+'_calibrated'] for r in e['rows']]
        point_groups[f]=[('AC',ac)]+[(s,[r for e in d['groups'][f'{f}_{s}'] for r in e['rows']
                                      if r.get('native',{}).get('finite')]) for s in STEPS]
    means=[np.mean([r['native']['point_rmse_mm']-r['readout_persistence']['point_rmse_mm'] for r in rows])
           for candidates in point_groups.values() for _,rows in candidates]
    # Retain the central magnification while keeping every group mean visible.
    point_limits=(min(-2.5,np.floor((min(means)-1)/2.5)*2.5),
                  max(5,np.ceil((max(means)+1)/2.5)*2.5))
    for ax in (full,zoom):
        ax.axhline(0,color='#677780',lw=.65,ls='--',zorder=1)
        ax.set(xlim=(-.6,6.6),xticks=[0,1,2,4,5,6],xticklabels=['AC','0','50','AC','0','50'])
        style_axis(ax);ax.tick_params(labelsize=5.6,pad=1.5);ax.grid(axis='y',alpha=.18,lw=.4)
    full.set(ylim=(-10,27),yticks=[0,20]);zoom.set(ylim=point_limits,yticks=[point_limits[0],0,point_limits[1]])
    full.axhspan(*point_limits,color='#DCE5EB',alpha=.22,zorder=0)
    full.set_ylabel('ΔRMSE (mm)',fontsize=5.9,labelpad=1.0)
    fig.text(.151,.249,'Full range · means labeled',ha='center',fontsize=5.8)
    fig.text(.368,.249,f'Zoom: {point_limits[0]:g} to {point_limits[1]:g} mm',ha='center',fontsize=5.8)
    for spine in zoom.spines.values():spine.set_visible(True);spine.set_color('#8A9BA6');spine.set_linewidth(.5)
    for i,f in enumerate(FAMILIES):
        candidates=point_groups[f]
        for j,(s,rows) in enumerate(candidates):
            x=j+4*i;delta=np.array([r['native']['point_rmse_mm']-r['readout_persistence']['point_rmse_mm'] for r in rows]);color=COLORS[i]
            # Individual cases share the centre and width of their mean bar.
            xx=x+rng.uniform(-.23,.23,len(delta));mask=(delta>=point_limits[0])&(delta<=point_limits[1])
            for ax,visible in ((full,np.ones(len(delta),dtype=bool)),(zoom,mask)):
                ax.bar(x,delta.mean(),width=.60,facecolor=color,edgecolor=color,alpha=.25,lw=.7,zorder=2)
                ax.scatter(xx[visible],delta[visible],s=1.8,color=color,
                           alpha=.23,linewidths=0,rasterized=True,zorder=3)
                ax.hlines(delta.mean(),x-.30,x+.30,color=color,lw=.95,zorder=4)
            point.append(dict(family=f,step=s,values_mm=delta.tolist(),mean_mm=float(delta.mean()),
                              full_range_cases=len(delta),zoom_cases=int(mask.sum()),
                              display='Individual cases overlaid directly on their mean bar in both full and magnified views'))
            zooms.append(dict(panel='c',family=f,step=s,bounds=list(point_limits),visible_cases=int(mask.sum()),total_cases=len(delta)))
            for ax in (full,zoom):
                ax.annotate(f'{delta.mean():.1f}',(x,delta.mean()),xytext=(0,2 if delta.mean()>=0 else -2),
                            textcoords='offset points',ha='center',va='bottom' if delta.mean()>=0 else 'top',
                            fontsize=6.2,color='#24333D',zorder=5,
                            bbox=dict(facecolor='white',edgecolor='none',alpha=.90,pad=.2))

    values=[v for r in point for v in r['values_mm']]
    full.set_ylim(min(-10, np.floor(min(values)/5)*5-2), max(27, np.ceil(max(values)/5)*5+2))
    lo,hi=full.get_ylim()
    ticks=MaxNLocator(nbins=2,min_n_ticks=2).tick_values(lo,hi)
    full.set_yticks(ticks[(ticks>=lo)&(ticks<=hi)])

    # Full paired-error distributions and a complementary diagnostic: the
    # distribution of generated/reference residual ratios for the matched cases.
    mechanics=[]; core_views=[]; mechanics_diagnostics=[]
    for j,(metric,tol,title) in enumerate((('effort','effort_tolerance','Actuator residual'),('energy_J','energy_tolerance_J','Energy residual'))):
        ax=fig.add_axes([.515+j*.275,.135,.205,.380])
        core=ax.inset_axes([.63,.10,.33,.32],zorder=7)
        allvals=[p[key+'_'+metric]/p[tol] for f in FAMILIES for p in groups[f'{f}_50']['paired']['pairs'] for key in ('reference','model')]
        limits=[10**np.floor(np.log10(min(allvals))),10**np.ceil(np.log10(max(allvals)))];limits[0]=min(limits[0],.1)
        ax.set(xscale='log',yscale='log',xlim=limits,ylim=limits)
        ax.set_facecolor('#F3F6F8')
        # The tolerance bands and the above-identity wedge encode distinct
        # comparisons: engineering qualification and excess over the reference.
        ax.fill_between(limits,limits,[limits[1]]*2,color='#E7DBCB',alpha=.24,zorder=-2)
        ax.axhspan(limits[0],1,color='#DCEBDC',alpha=.55,zorder=-1)
        ax.axvline(1,color='#65836E',ls=':',lw=.65,zorder=0)
        ax.axhline(1,color='#65836E',ls=':',lw=.65,zorder=0)
        ax.plot(limits,limits,color='#81909A',lw=.7,zorder=0)
        ax.text(.42,.26,'1:1',rotation=37,fontsize=5.7,color='#647580',rotation_mode='anchor')
        core.set(xscale='log',xlim=(1,100) if j==0 else (.1,1000),ylim=(-.45,1.9),
                 xticks=[1,10,100] if j==0 else [.1,10,1000],yticks=[1,0],yticklabels=['S','D'])
        core.set_facecolor('#F3F6F8')
        core.axhspan(.55,1.45,color=COLORS[0],alpha=.07,zorder=0)
        core.axhspan(-.45,.45,color=COLORS[1],alpha=.07,zorder=0)
        core.axvline(1,color='#81909A',lw=.6,ls=':',zorder=0)
        for i,f in enumerate(FAMILIES):
            g=groups[f'{f}_50'];pairs=g['paired']['pairs']
            x=np.array([p['reference_'+metric]/p[tol] for p in pairs]);y=np.array([p['model_'+metric]/p[tol] for p in pairs])
            kde=gaussian_kde(np.log10(np.vstack([x,y])),bw_method=.30)
            grid=np.linspace(np.log10(limits[0]),np.log10(limits[1]),120);gx,gy=np.meshgrid(grid,grid)
            z=kde(np.vstack([gx.ravel(),gy.ravel()])).reshape(gx.shape);z/=z.max()
            ax.contourf(10**gx,10**gy,z,levels=[.15,.45,.75,1.01],colors=[COLORS[i]]*3,alpha=.075,zorder=1)
            ax.contour(10**gx,10**gy,z,levels=[.45,.75],colors=[COLORS[i]],linewidths=.55,alpha=.75,zorder=2)
            ax.scatter(x,y,s=3.5,marker='o' if i==0 else '^',color=COLORS[i],alpha=.34,linewidths=0,rasterized=True,zorder=3)
            ax.scatter(x.mean(),y.mean(),s=18,marker='D',facecolors='white',edgecolors=COLORS[i],linewidths=.85,zorder=5)
            ratios=y/x;quartiles=np.quantile(ratios,[.25,.5,.75]);mean=float(ratios.mean())
            core.bxp([dict(med=quartiles[1],q1=quartiles[0],q3=quartiles[2],
                           whislo=ratios.min(),whishi=ratios.max(),fliers=[])],
                     positions=[1-i],widths=.28,vert=False,manage_ticks=False,patch_artist=True,
                     boxprops=dict(facecolor=COLORS[i]+'30',edgecolor=COLORS[i],lw=.65),
                     whiskerprops=dict(color=COLORS[i],lw=.65),capprops=dict(color=COLORS[i],lw=.65),
                     medianprops=dict(color=COLORS[i],lw=.8),zorder=2)
            core.scatter(mean,1-i,marker='D',s=9,facecolor='white',edgecolor=COLORS[i],lw=.65,zorder=4)
            core.text(mean,1.27-i,f'{mean:.1f}×',ha='center',va='bottom',fontsize=5.7,color=COLORS[i])
            mechanics_diagnostics.append(dict(family=f,metric=metric,case_ratios=ratios.tolist(),
                paired_cases=len(pairs),mean=mean,quartiles=quartiles.tolist(),range=[float(ratios.min()),float(ratios.max())],
                comparison='Generated residual / matched reference residual',position='lower right'))
            si=[g['paired']['means'][k+'_'+metric] for k in ('reference','model')]
            mechanics.append(dict(family=f,metric=metric,reference_ratios=x.tolist(),generated_ratios=y.tolist(),limits=limits,paired_cases=len(pairs),si_means=si))
            density.append(dict(panel='d',family=f,metric=metric,coordinate_space='log10 error/tolerance on both axes',kernel='Gaussian',bandwidth_factor=.30,levels_of_peak=[.15,.45,.75],cases=len(pairs)))
        ax.set(xticks=[.1,1,10,100],yticks=[1,10,100]);ax.set_title(title,loc='left',fontsize=6.6,pad=3)
        core.set_title('Gen./ref. (×)',loc='right',fontsize=5.7,pad=2)
        for view in (ax,core):
            for axis in ((view.xaxis,view.yaxis) if view is ax else (view.xaxis,)):
                axis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'));axis.set_minor_locator(NullLocator())
            style_axis(view);view.grid(visible=False);view.tick_params(labelsize=5.6 if view is ax else 5.1,pad=1.2)
        ax.set_xlabel('Reference error / tolerance',fontsize=6,labelpad=1.5)
        if j==0:ax.set_ylabel('Generated error / tolerance',fontsize=6,labelpad=1.6)
        for spine in core.spines.values():spine.set_visible(True);spine.set_color('#8A9BA6');spine.set_linewidth(.6)
        core_views.append((ax,core,metric))
    fig.legend(handles=[Line2D([],[],marker='o',color=COLORS[0],lw=0,ms=3,label='Stewart: 450 pairs'),
        Line2D([],[],marker='^',color=COLORS[1],lw=0,ms=3,label='Delta: 424 pairs'),
        Line2D([],[],marker='D',color='#53616A',mfc='white',lw=0,ms=3,label='Mean')],loc='center',bbox_to_anchor=(.739,.590),ncol=3,frameon=False,fontsize=5.9,handlelength=.7,columnspacing=.8)
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    for ax,core,metric in core_views:
        box=core.get_tightbbox(renderer)
        for record in mechanics:
            if record['metric']!=metric:continue
            points=ax.transData.transform(np.column_stack((record['reference_ratios'],record['generated_ratios'])))
            hidden=[i for i,(x,y) in enumerate(points) if box.contains(x,y)]
            assert not hidden, ('Mechanical inset obscures an original case',metric,record['family'],
                                [(record['reference_ratios'][i],record['generated_ratios'][i]) for i in hidden],
                                box.transformed(ax.transAxes.inverted()).bounds)
    labels=[t for t in fig.findobj(match=Text) if t.get_visible() and t.get_text().strip()]
    boxes=[Text.get_window_extent(t,renderer).padded(.1*fig.dpi/72) for t in labels]
    overlaps=[(labels[i].get_text(),labels[j].get_text()) for i in range(len(labels)) for j in range(i+1,len(labels)) if boxes[i].overlaps(boxes[j])]
    if overlaps:fig.savefig(PAPER/'build/spatial-layout-review.png',dpi=210,bbox_inches='tight')
    assert not overlaps, ('Overlapping figure text',overlaps)
    export(fig,'spatial_vjepa_audit')
    evidence=dict(schema='lingbot-0-50-three-seeds',examples=examples,adaptation=adapt,point_comparison=point,
        matched_reference_mechanics=mechanics,mechanics_diagnostics=mechanics_diagnostics,
        reconstruction=reconstruction,zooms=zooms,density_contours=density,figure_size_inches=[7.05,2.75],
        text_layout=dict(labels_checked=len(labels),overlaps=overlaps,insets_clear_of_original_points=True),sources=dict(d['sources'], **ac_data['sources']),selection='Seed 42, first registered test case per family, final frame 119; not selected by error.',
        statistics='All seed/case observations shown. MSE changes compare equal-seed means; quartile bars summarize pooled case ratios. Point-error cases directly overlay their labeled mean bars in both full and magnified views. Mechanical diamonds are pooled means; green bands mark generated error within tolerance, and pale warm wedges mark generated error above the paired reference. Lower-right inset boxes show quartiles and the full range of generated/reference residual ratios, with labeled arithmetic-mean diamonds (mean of paired ratios, not ratio of means); tinted rows identify the two families.')
    (PAPER/'sources/spatial_expansion_figure.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


if __name__=='__main__':figure()
