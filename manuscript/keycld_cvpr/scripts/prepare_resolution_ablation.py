"""Seed-level resolution effects and fixed-video measurement-path ablations."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from prepare_resolution_artifacts import PAPER, balanced
from prepare_artifacts import BLUE, ORANGE, GREEN, save


def build():
    data=json.loads((PAPER/'sources/resolution_comparison.json').read_text())
    fig,axes=plt.subplots(2,3,figsize=(7.05,3.75))
    for col,(name,title) in enumerate([('rgb','(a) RGB / persistence'),('score','(b) Native continuous score'),('prediction','(c) Native point RMSE (mm)')]):
        ax=axes[0,col]
        for model,color in [('KeyCLD',BLUE),('CON',ORANGE)]:
            rr=[r for r in data['runs'] if r['model']==model]
            for i,seed in enumerate((42,43,44)):
                if name=='rgb':values=[r['visual']['seed_rgb'][i]/r['visual']['persistence_rgb_mse'] for r in rr]
                elif name=='score':values=[r['native']['seed_scores'][i] for r in rr]
                else:values=[balanced([x for x in r['native']['rows'] if x['seed']==seed],'prediction')*1000 for r in rr]
                ax.plot([0,1],values,color=color,alpha=.75,lw=.9,marker=('o','s','^')[i],ms=3.5)
        ax.set(xticks=[0,1],xticklabels=['64 px','256 px'],xlim=(-.18,1.18))
        if name=='rgb':ax.axhline(1,color='#596A74',ls='--',lw=.7)
        if name=='score':ax.set_ylim(0,37)
        if name=='prediction':ax.axhspan(0,10,color=GREEN,alpha=.08);ax.axhline(10,color=GREEN,ls='--',lw=.7);ax.set_ylim(0,310)
        ax.set_title(title,fontsize=7.7,loc='left',pad=6)
    methods=['raw_cartesian_spline5','cartesian_spline5','phase_sg7d3']
    for col,(metric,title,tolerance) in enumerate([('score','(d) Fixed-video continuous score',None),('dynamics','(e) Projected torque (N m)',.25),('energy','(f) Projected energy (J)',.05)]):
        ax=axes[1,col]; selected=methods if col==0 else methods[1:]
        for model,source,color,label in [('KeyCLD','reference','#637D6F','Correct RGB'),('KeyCLD','generated',BLUE,'KeyCLD RGB'),('CON','generated',ORANGE,'CON RGB')]:
            blocks=[data['reader'][f'{model}/{source}/{m}'] for m in selected]
            values=[b['score'] if metric=='score' else b['means'][metric] for b in blocks]
            ax.plot(range(len(selected)),values,color=color,lw=1.2,marker='D' if source=='reference' else 'o',ms=4,label=label)
        labels=['Cartesian','Proj. Cart.','Proj. phase'] if col==0 else ['Cartesian','Phase']
        ax.set(xticks=range(len(selected)),xticklabels=labels,xlim=(-.16,len(selected)-.84))
        if tolerance:
            ax.set_yscale('log');ax.set_ylim((.06,15) if col==1 else (.025,25))
            ax.axhspan(ax.get_ylim()[0],tolerance,color=GREEN,alpha=.09)
            ax.axhline(tolerance,color=GREEN,lw=.7,ls='--')
        else:ax.set_ylim(0,102)
        ax.set_title(title,fontsize=7.7,loc='left',pad=6)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=6.8,length=2.5,pad=3)
        ax.grid(axis='y',color='#DDE4E8',lw=.45);ax.set_axisbelow(True)
    handles=[Line2D([],[],color=BLUE,lw=1,label='KeyCLD'),Line2D([],[],color=ORANGE,lw=1,label='CON')]
    handles += [Line2D([],[],color='#667680',marker=marker,lw=0,ms=3.5,label=f'Seed {seed}') for marker,seed in zip(['o','s','^'],[42,43,44])]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.01),ncol=5,frameon=False,fontsize=7,columnspacing=1.2,handlelength=1.4)
    axes[1,0].legend(loc='lower right',fontsize=6,frameon=False)
    fig.subplots_adjust(left=.065,right=.985,bottom=.09,top=.865,hspace=.51,wspace=.32)
    save(fig,'resolution_ablation')


if __name__=='__main__':build()
