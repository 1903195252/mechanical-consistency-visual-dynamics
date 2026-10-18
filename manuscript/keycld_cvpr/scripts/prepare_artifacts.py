"""Build manuscript figures/tables from completed experiments; no model execution."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "output/fourbar/.cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter
import numpy as np

sys.path.insert(0, str(ROOT))
from experiments.closed_chain_eval.planar import physics

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
})
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
LABELS = ["Nominal", "Viscous", "Coulomb", "Mixed", "Payload", "Density", "Size", "Slider"]
NAMES = ["nominal", "viscous", "coulomb", "mixed_friction", "heavy_payload",
         "dense_bars", "geometry_scaled", "crank_slider"]
SCORED = ROOT / "output/keycld_fourbar/scored_evaluation"
EXTENDED = ROOT / "output/keycld_fourbar/extended_dynamics"


def read(path):
    return json.loads(path.read_text())


def save(fig, name):
    fig.savefig(PAPER / "figures" / (name + ".pdf"), bbox_inches="tight", pad_inches=.025)
    fig.savefig(PAPER / "figures" / (name + ".svg"), bbox_inches="tight", pad_inches=.025)
    fig.savefig(PAPER / "build" / (name + ".png"), dpi=180, bbox_inches="tight", pad_inches=.025)
    plt.close(fig)


def mechanisms():
    fig = plt.figure(figsize=(7.05, 2.35))
    grid = fig.add_gridspec(2, 3, height_ratios=[1, .43], width_ratios=[1, 1, 1.03])
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    for ax, name, title in zip(axes[:2], ["nominal", "crank_slider"], ["(a) Four-bar (4R)", "(b) Crank-slider (3R+1P)"]):
        c = read(SCORED / "truth" / name / "run_3000.json")
        with np.load(SCORED / "truth" / name / "run_3000.npz") as data:
            q = data["q"][47]
        B, C, P = physics.point_states(q, np.zeros(3), np.zeros(3), c)[0]
        A = np.array([0., c["pivot_height_m"]])
        D = np.array([c["ground_length_m"], c["pivot_height_m"]])
        ax.plot([A[0], .64 if name == "crank_slider" else D[0]], [A[1], A[1]], color="#9BA4AD", lw=3, zorder=0)
        segments = [(A, B, BLUE), (B, C, ORANGE)]
        if name == "nominal":
            segments += [(D, C, GREEN)]
        else:
            ax.add_patch(Rectangle((C[0]-.025, C[1]-.017), .05, .034, facecolor=GREEN, zorder=3))
        for start, end, color in segments:
            ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=6, solid_capstyle="round", zorder=2)
        for key, point, offset in [("A", A, (-.025,-.045)), ("B", B, (-.018,.03)), ("C", C, (.013,.029))]:
            ax.scatter(*point, s=23, facecolor="white", edgecolor="#26323D", zorder=4, lw=.8)
            ax.text(*(point+offset), key, fontsize=9)
        if name == "nominal":
            ax.scatter(*D, s=23, facecolor="white", edgecolor="#26323D", zorder=4, lw=.8)
            ax.text(D[0]-.005, D[1]-.045, "D", fontsize=9)
        ax.scatter(*P, marker="x", s=22, color="#26323D", zorder=4)
        ax.text(P[0]-.025, P[1]+.03, "P", fontsize=8)
        ax.annotate(r"$\tau_A$", xy=A+[.005,.008], xytext=A+[-.10,.055],
                    arrowprops={"arrowstyle": "->", "color": "#333"}, fontsize=10)
        ax.set_xlim(-.14, .68)
        ax.set_ylim(.17, .71)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(title, loc="left", pad=3)
    with np.load(SCORED / "observations/nominal/run_3000.npz") as data:
        a = data["x"][2]
    with np.load(SCORED / "observations/crank_slider/run_3000.npz") as data:
        b = data["x"][2]
    axes[2].imshow(np.concatenate([a, b], axis=1), interpolation="nearest")
    axes[2].axis("off")
    axes[2].set_title("(c) Actual input frames", loc="left", pad=3)
    axes[2].text(.5, -.10, r"$64\times64$ RGB, calibrated view", ha="center", transform=axes[2].transAxes, fontsize=8)
    flow = fig.add_subplot(grid[1, :])
    flow.set_xlim(0, 1); flow.set_ylim(0, 1); flow.axis("off")
    stages = [(0.005, .19, "3 RGB frames\n+ known action", "#EDF4FA"),
              (.245, .16, "Frozen\nKeyCLD", "#EDF4FA"),
              (.455, .22, "Raw point state\n+ semantic check", "#F4F4F4"),
              (.745, .245, "Independent SI audit\n+ coverage + residuals", "#EAF5F0")]
    for x, width, text, color in stages:
        flow.add_patch(Rectangle((x, .32), width, .64, facecolor=color, edgecolor="#71808B", lw=.7))
        flow.text(x+width/2, .64, text, ha="center", va="center", fontsize=8)
    for a, b in zip(stages[:-1], stages[1:]):
        flow.add_patch(FancyArrowPatch((a[0]+a[1]+.003,.64), (b[0]-.005,.64),
                                     arrowstyle="-|>", mutation_scale=8, lw=.8, color="#485460"))
    flow.text(.867, .05, "Mass, inertia, friction: evaluator only", ha="center", fontsize=6.8, color="#42584B")
    fig.subplots_adjust(left=.015, right=.995, top=.91, bottom=.06, wspace=.12, hspace=.31)
    save(fig, "mechanisms")


def controlled_diagnostics():
    from prepare_diagnostic_figure import build
    build()


def score_profile(report):
    keys = ["geometry","constraint_velocity","constraint_acceleration","dynamics","energy","prediction"]
    values = []
    for model in ["video","state_RK4"]:
        row=[]
        for key in keys:
            item=report["scorecards"][model]["component_scores"][key]
            row.append(item["score_100"] if item["measured_trajectories"] else np.nan)
        values.append(row)
    values=np.asarray(values)
    cmap=LinearSegmentedColormap.from_list("audit_pass",["#F8FAFC","#A9D4E8","#006899"])
    cmap.set_bad("#E2E5E8")
    fig,ax=plt.subplots(figsize=(3.45,1.65))
    ax.imshow(np.ma.masked_invalid(values),vmin=0,vmax=100,cmap=cmap,aspect="auto",interpolation="nearest")
    for i in range(2):
        for j in range(6):
            value=values[i,j]
            if np.isnan(value):
                ax.add_patch(Rectangle((j-.5,i-.5),1,1,fill=False,hatch="///",edgecolor="#B0B6BD",lw=0))
                text="N/A"; color="#343A40"
            else:
                text="%g" % value; color="white" if value>=70 else "#26323D"
            ax.text(j,i,text,ha="center",va="center",fontsize=8.2,color=color,fontweight="bold")
    ax.set_xticks(range(6),["Geom.","Vel.","Acc.","Force","Energy","Pred."],fontsize=7)
    ax.set_yticks(range(2),["Video","State"],fontsize=8)
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True)
    ax.grid(which="minor",color="white",linewidth=1.2)
    ax.tick_params(which="both",length=0)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_title("Per-criterion pass rate (%)",loc="left",fontsize=9,pad=7)
    ax.text(0,-.65,"N/A: no physical readout",fontsize=7,ha="left",transform=ax.transAxes,color="#52616B")
    ax.text(1,-.65,"Joint pass: 0/160 each",fontsize=7,ha="right",transform=ax.transAxes,color="#52616B")
    fig.subplots_adjust(left=.16,right=.99,top=.81,bottom=.40)
    save(fig,"score_profile")


def residuals(report):
    rows = report["scorecards"]["state_RK4"]["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.45))
    jitter = np.linspace(-.22, .22, 20)
    for ax, metric, ylabel, limit, title in [
        (axes[0], "dynamics", r"Motor residual RMS (N m)", .1, "(a) Independent force balance"),
        (axes[1], "energy", r"Maximum work-energy residual (J)", .02, "(b) Independent energy balance"),
    ]:
        for index, name in enumerate(NAMES):
            values = np.array([row["metrics"][metric]["value"] for row in rows if row["case_key"][1] == name])
            assert len(values) == 20 and np.isfinite(values).all() and np.all(values > 0)
            color = GREEN if name == "crank_slider" else BLUE
            ax.scatter(index+jitter, values, s=12, color=color, alpha=.68, edgecolors="none", zorder=3)
            q1, med, q3 = np.quantile(values, [.25,.5,.75])
            ax.plot([index, index], [q1,q3], color="#222", lw=2.5, zorder=4)
            ax.plot([index-.19,index+.19], [med,med], color="#222", lw=1.2, zorder=4)
        ax.axhline(limit, color=ORANGE, ls="--", lw=1.1, zorder=1)
        ax.set_yscale("log")
        ax.set_xticks(range(8), LABELS, rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left", pad=7)
        ax.grid(axis="y", which="major", color="#DDE2E6", lw=.5)
    axes[0].set_ylim(.014, 30)
    axes[1].set_ylim(.009, 900)
    fig.text(.53, .99, "20 paired trajectories / condition    Black: median and IQR    Dashed: diagnostic tolerance",
             ha="center", va="top", fontsize=7)
    fig.subplots_adjust(left=.078, right=.994, top=.83, bottom=.25, wspace=.29)
    save(fig, "residuals")


def qualitative():
    with np.load(ROOT / "output/keycld_fourbar/trained/test_example_0.npz") as data:
        ref, pred = data["reference_rgb"], data["prediction_rgb"]
    indices = [0,3,9,29,49]
    fig, axes = plt.subplots(2, 5, figsize=(7.0, 2.55))
    for j, index in enumerate(indices):
        for i, values in enumerate([ref, pred]):
            axes[i,j].imshow(values[index], interpolation="nearest")
            axes[i,j].set_xticks([]); axes[i,j].set_yticks([])
            for spine in axes[i,j].spines.values(): spine.set_visible(False)
        axes[0,j].set_title("%.2f s" % ((index+1)/100), fontsize=9, pad=3)
    axes[0,0].set_ylabel("Reference", fontsize=9)
    axes[1,0].set_ylabel("Prediction", fontsize=9)
    fig.subplots_adjust(left=.055, right=.997, top=.90, bottom=.025, hspace=.02, wspace=.015)
    save(fig, "qualitative")


def probes_plot(probes):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    names = ["viscous", "coulomb", "mixed_friction"]
    true = [probes["conditions"][name]["friction"]["true_odd_acceleration_rms_rad_s2"] for name in names]
    model = [probes["conditions"][name]["friction"]["learned_odd_acceleration_max_abs_rad_s2"] for name in names]
    assert all(value == 0 for value in model)
    axes[0].bar(np.arange(3)-.17, true, .33, color=BLUE, label="Physical reference (RMS)")
    axes[0].plot(np.arange(3)+.17, model, "x", color=ORANGE, markersize=7, label="Model: identically zero",clip_on=False)
    for i, value in enumerate(true):
        axes[0].text(i-.17, value+5, "%.2f" % value, ha="center", fontsize=7)
        axes[0].text(i+.17, 4, "0", ha="center", fontsize=7, color=ORANGE)
    axes[0].set_xticks(range(3), ["Viscous","Coulomb","Mixed"])
    axes[0].set_ylabel(r"Odd acceleration (rad s$^{-2}$)")
    axes[0].set_ylim(0, 290)
    axes[0].legend(frameon=False, fontsize=6.7, loc="upper center")
    axes[0].set_title("(a) Matched velocity reversal", loc="left")
    names = ["nominal","heavy_payload","dense_bars","geometry_scaled","crank_slider"]
    estimates = [probes["conditions"][name]["inertia"] for name in names]
    medians = [v["equivalent_inertia_relative_error"]["median"]*100 for v in estimates]
    means = [v["equivalent_inertia_relative_error"]["mean"]*100 for v in estimates]
    axes[1].vlines(range(5),medians,means,color="#A7B4BE",lw=1)
    axes[1].plot(range(5),medians,"o",color=BLUE,label="Median",markersize=4.5)
    axes[1].plot(range(5),means,"^",color=ORANGE,label="Mean",markersize=5)
    axes[1].set_xticks(range(5), [label+"\n%d/73"%stat["positive_gain_poses"]
                                 for label,stat in zip(["Nominal","Payload","Density","Size","Slider"],estimates)])
    axes[1].set_ylabel("Relative inertia error (%)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(20,1100)
    axes[1].set_yticks([25,50,100,250,500,1000])
    axes[1].yaxis.set_major_formatter(ScalarFormatter())
    axes[1].grid(axis="y",which="major",color="#E3E7EB",lw=.5)
    axes[1].legend(frameon=False,fontsize=7,ncol=2,loc="upper left")
    axes[1].set_title("(b) Inertia response and coverage", loc="left")
    axes[1].set_xlabel("Positive-gain samples shown below each condition",fontsize=7,labelpad=4)
    fig.subplots_adjust(left=.08,right=.994,top=.86,bottom=.28,wspace=.32)
    save(fig, "probes")


def table_files(report, probes):
    rows = []
    for name, label in zip(NAMES, LABELS):
        c = report["scorecards"]["state_RK4"]["conditions"][name]
        f, e, p = [c["metrics"][metric] for metric in ["dynamics","energy","prediction"]]
        rows.append("%s & $%.3f\\!\\pm\\!%.3f$ & %.3f & %.1f & %d/20 \\\\" %
                    (label, f["mean"], f["std"], e["mean"], p["mean"]*1000, f["passed"]))
    (PAPER/"tables/transfer_rows.tex").write_text("\n".join(rows)+"\n")
    effects=[]
    for name,label in zip(NAMES,LABELS):
        c=report["scorecards"]["state_RK4"]["conditions"][name]
        effects.append("%s & %.3f & %.3f " % (label,c["metrics"]["dynamics"]["mean"],c["metrics"]["energy"]["mean"]) + chr(92)*2)
    (PAPER/"tables/parameter_effect_rows.tex").write_text("\n".join(effects)+"\n")
    labels = ["Position constraint","Velocity constraint","Acceleration constraint","Force balance","Work--energy balance","Conditional prediction"]
    keys = ["geometry","constraint_velocity","constraint_acceleration","dynamics","energy","prediction"]
    rows = []
    for key, label in zip(keys,labels):
        v = report["scorecards"]["video"]["component_scores"][key]
        s = report["scorecards"]["state_RK4"]["component_scores"][key]
        video = "--" if v["measured_trajectories"] == 0 else "%.1f" % v["score_100"]
        rows.append("%s & %s & %.1f & %d/160 \\\\" % (label, video, s["score_100"], s["passed_trajectories"]))
    (PAPER/"tables/score_rows.tex").write_text("\n".join(rows)+"\n")
    rows=[]
    for name,label in zip(["nominal","heavy_payload","dense_bars","geometry_scaled","crank_slider"],["Nominal","Payload","Density","Size","Slider"]):
        item=probes["conditions"][name]["inertia"]
        error=item["equivalent_inertia_relative_error"]
        rows.append("%s & %d/73 & %.2f & %.2f \\\\" % (label,item["positive_gain_poses"],error["median"]*100,error["mean"]*100))
    (PAPER/"tables/inertia_rows.tex").write_text("\n".join(rows)+"\n")
    controls=read(EXTENDED/"identifiability_controls.json")
    rows=[]
    for name,label in zip(NAMES,LABELS):
        c=controls[name]
        mass=max(abs(float(v)) for v in c["mass_relative_errors"])
        rows.append("%s & %d/12 & %.1f & $%.2f\\!\\times\\!10^{-6}$ & $%.2f\\!\\times\\!10^{-8}$ \\\\" %
                    (label,c["regressor_rank"],c["normalized_condition_number"],mass*1e6,c["test_torque_rms_Nm"]*1e8))
    (PAPER/"tables/identification_rows.tex").write_text("\n".join(rows)+"\n")


def main():
    report=read(SCORED/"scores.json")
    probes=read(EXTENDED/"field_probes.json")
    mechanisms()
    controlled_diagnostics()
    residuals(report)
    score_profile(report)
    qualitative()
    probes_plot(probes)
    table_files(report,probes)
    sources={
        "torque_transfer_tables":"output/keycld_fourbar/scored_evaluation/scores.json",
        "score_summary_posthoc":"output/keycld_fourbar/scored_evaluation/score_summary.json",
        "calibration":"output/keycld_fourbar/scored_evaluation/calibration.json",
        "visual_and_sampling_controls":"output/keycld_fourbar/diagnosis/report.json",
        "visual_analysis":"sources/visual_analysis.json: motion-region concentration, paired local predictions, physical semantics and verified loss construction",
        "training_integrator_comparison":"output/keycld_fourbar/diagnosis/state_oracle/paired_statistics.json",
        "structural_probes":"output/keycld_fourbar/extended_dynamics/field_probes.json",
        "classical_parameter_control":"output/keycld_fourbar/extended_dynamics/identifiability_controls.json",
        "qualitative_selection":"First saved original held-out example, seed 500; fixed frames 0,3,9,29,49",
        "rendered_figure_sources":"sources/rendered_figure_sources.json: 343 recorded isometric frames, nine detailed examples, and actual model inputs",
        "diagnostic_figure":"sources/diagnostic_figure_data.json: reference/reconstruction/forecast images, all 50 test trajectories and three readout conditions (150 3D bars), unchanged mean RMSE and SD; paired state-integrator control retained",
        "score_profile":"Existing family-balanced component pass rates; unavailable physical measurements are hatched N/A, never zeros",
        "inertia_visualization":"Both median and mean on the positive-gain subset; all outliers retained and coverage counts printed",
    }
    from prepare_corpus_artifacts import main as corpus_artifacts
    from prepare_con_artifacts import main as con_artifacts
    from prepare_dynamics_extension import main as dynamics_extension
    from prepare_shared_video_figure import build as shared_video_figure
    from render_audit_diagram import render as audit_diagram
    sources.update(corpus_artifacts())
    sources.update(con_artifacts())
    sources.update(dynamics_extension())
    sources.update(shared_video_figure())
    audit_diagram()
    (PAPER/"sources/evidence_map.json").write_text(json.dumps(sources,indent=2)+"\n")
    print("Built the unified manuscript figures, tables, and editable audit diagram exports.")


if __name__=="__main__":
    main()
