"""Standalone measured calibration figure; no manuscript files are changed."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .visual_calibration import OUT


def run():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                         "axes.spines.right":False,"pdf.fonttype":42,"svg.fonttype":"none"})
    fig,axes=plt.subplots(2,2,figsize=(10,7.4),layout="constrained")
    axes=axes.ravel()
    old,new="#7d8792","#167c80"
    data=json.loads((OUT/"optimization/robust_planar_reader/stress.json").read_text())["rows"]
    qualities=["clean","mild_blur","blur","blur_low_contrast"]
    labels=["Clean","Mild blur","Blur","Blur + low\ncontrast"]
    x=np.arange(4)
    for offset,reader,color,label in [(-.19,"baseline",old,"Clean-trained"),(.19,"robust",new,"Augmented")]:
        y=[1000*np.mean([r["point_rmse_m"] for r in data if r["reader"]==reader and r["quality"]==q
                        and r["calibration_role"]=="verification"]) for q in qualities]
        axes[0].bar(x+offset,y,.36,color=color,label=label,zorder=3)
        for xi,yi in zip(x+offset,y):axes[0].text(xi,yi*1.1,f"{yi:.1f}",ha="center",fontsize=8)
    axes[0].set(xticks=x,xticklabels=labels,yscale="log",ylim=(.9,110),ylabel="Reference-video point RMSE (mm)",
                title="(a) Image-quality robustness")
    axes[0].legend(loc="upper left",frameon=False,fontsize=8)
    n=json.loads((OUT/"optimization/coordinate_derivatives/verification.json").read_text())
    keys=["constraint_velocity","constraint_acceleration","dynamics","energy"]
    scales=np.array([.01,.1,.25,.05])
    for offset,method,color,label in [(-.19,"cartesian_sg7d3",old,"Cubic SG"),(.19,n["selected"],new,"Quintic spline")]:
        y=np.array([n["summary"][method]["maxima"][k] for k in keys])/scales
        axes[1].bar(x+offset,y,.36,color=color,label=label,zorder=3)
    axes[1].axhline(1,color="#475466",linestyle="--",linewidth=.9,label="Engineering threshold")
    axes[1].set(xticks=x,xticklabels=["Velocity","Acceleration","Torque","Energy"],yscale="log",ylim=(.007,40),
                ylabel="Max reference error / engineering threshold",title="(b) Exact-state numerical control")
    axes[1].legend(loc="upper left",frameon=False,fontsize=8)
    axes[1].text(.98,.92,"All-six pass: 40/54 → 54/54",transform=axes[1].transAxes,ha="right",fontsize=9)
    summaries=json.loads((OUT/"summary.json").read_text())["summaries"]
    x=np.arange(2)
    for offset,reader,color,label in [(-.19,"shared_spatial_256",old,"Absolute-coordinate head"),
                                     (.19,"geometry_spatial_256",new,"CAD-coordinate head")]:
        y=[next(r["verification_point_rmse_mean_mm"] for r in summaries if r["reader"]==reader and r["category"]==f)
           for f in ("H-A","H-B")]
        axes[2].bar(x+offset,y,.36,color=color,label=label,zorder=3)
        for xi,yi in zip(x+offset,y):axes[2].text(xi,yi+.25,f"{yi:.2f}",ha="center",fontsize=8)
    axes[2].set(xticks=x,xticklabels=["Stewart","Delta"],ylim=(0,16),ylabel="Reference-video point RMSE (mm)",
                title="(c) Spatial coordinate heads, 256 px")
    axes[2].legend(loc="upper right",frameon=False,fontsize=8)
    phase=OUT/"optimization/constrained_visual_phase/verification.json"
    if phase.exists():
        rows=json.loads(phase.read_text())["summary"]
        x=np.arange(2)
        for offset,method,color,label in [(-.19,"cartesian_spline5",old,"Coordinate spline"),
                                         (.19,"phase_sg7d3",new,"Cubic phase fit")]:
            r=next(r for r in rows if r["role"]=="verification" and r["quality"]=="clean" and r["method"]==method)
            y=np.array([r["mean"]["force_balance_Nm"],r["mean"]["work_energy_J"]])/[.25,.05]
            axes[3].bar(x+offset,y,.36,color=color,label=label,zorder=3)
            for xi,yi in zip(x+offset,y):axes[3].text(xi,yi*1.08,f"{yi:.1f}×",ha="center",fontsize=8)
        axes[3].axhline(1,color="#475466",linestyle="--",linewidth=.9)
        axes[3].set(xticks=x,xticklabels=["Torque","Energy"],yscale="log",ylim=(.6,100),
                    ylabel="Mean reference error / engineering threshold",
                    title="(d) Video differentiation after projection")
        axes[3].legend(loc="upper left",frameon=False,fontsize=8)
        axes[3].set_xlabel("27 held-out videos; engineering limits unchanged",fontsize=8)
    else:
        axes[3].set_visible(False)
    for ax in axes:ax.grid(axis="y",alpha=.18,zorder=0)
    for suffix in ("png","pdf","svg"):
        fig.savefig(OUT/("calibration_overview."+suffix),dpi=180)
    plt.close(fig)


if __name__=="__main__":run()
