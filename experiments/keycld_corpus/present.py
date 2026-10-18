"""Show actual completion, calibrated scores and assembly-cluster confidence intervals."""
import json
from pathlib import Path
import time
import numpy as np
from experiments.mechanism_corpus.catalog import write_json
from .prepare import OUT
from .physics_score import METRICS
from experiments.closed_chain_eval.score_reports import planar_report


def intervals(reports):
    # Preserve all three conditions and pair the same assemblies across seeds.
    categories=('S-A','S-B','S-J')
    identities={c:sorted({r['assembly_id'] for r in reports[0]['rows'] if r['category']==c}) for c in categories}
    conditions=('low_speed','rated','limit_or_load')
    data=np.zeros((len(reports),3,12,3,7),dtype=float)
    for si,report in enumerate(reports):
        lookup={(r['assembly_id'],r['condition']):r for r in report['rows']}
        for ci,c in enumerate(categories):
            for ai,identity in enumerate(identities[c]):
                for ui,condition in enumerate(conditions):
                    row=lookup[(identity,condition)]
                    data[si,ci,ai,ui]=[row['continuous_components_100'][k]/100 for k in METRICS]+[float(row['joint_pass'])]
    rng=np.random.default_rng(20261012);draws=[]
    for _ in range(2000):
        seeds=rng.integers(0,len(reports),len(reports));family=[]
        for ci in range(3):
            assemblies=rng.integers(0,12,12)
            family.append(data[seeds,ci][:,assemblies].mean(axis=(0,1,2)))
        values=np.mean(family,axis=0)*100
        draws.append([values[:6].mean(),values[-1]])
    bounds=np.percentile(draws,[2.5,97.5],axis=0)
    return {'composite_95pct':bounds[:,0].tolist(),'joint_95pct':bounds[:,1].tolist(),
            'replicates':2000,'independent_training_seeds':len(reports),
            'training_seed_uncertainty_measured':len(reports)>1,
            'unit':'assembly clusters within family, paired conditions and seeds',
            'joint_bootstrap_all_zero':bool(np.all(data[:,:,:,:,-1]==0)),
            'zero_event_note':'An all-zero empirical bootstrap interval is not a zero population upper bound.'}


def main():
    protocol=json.loads((OUT/'protocol.json').read_text())
    manifest=json.loads((OUT/'dataset_manifest.json').read_text())
    calibration=json.loads((OUT/'calibration.json').read_text())
    rows=[];summaries={};total_updates=0
    for arm in ('video_euler','state_euler','state_rk4'):
        completed=[]
        for seed in (42,43,44):
            folder=OUT/'runs'/('%s_seed%d'%(arm,seed))
            state=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
            tr=json.loads((folder/'training_report.json').read_text()) if (folder/'training_report.json').exists() else {}
            sc=json.loads((folder/'scores.json').read_text()) if (folder/'scores.json').exists() else None
            if sc:
                sc = planar_report(sc, protocol['evaluation']['limits_SI'], sc['calibration_sha256'])
            updates=tr.get('optimizer_updates',state.get('updates',0));total_updates+=updates
            row={'arm':arm,'seed':seed,'epoch':tr.get('epochs',state.get('epoch',0)),'updates':updates,
                 'training_complete':bool(tr),'status':state.get('stage','pending'),
                 'complete_physical_trajectories':sc['complete_physical_trajectories'] if sc else None,
                 'score':sc['composite_score_100'] if sc else None,'joint':sc['joint_pass_rate_100'] if sc else None}
            rows.append(row)
            if sc:
                completed.append(sc)
        if completed:
            summaries[arm]={'completed_seeds':len(completed),'composite_score_100':float(np.mean([s['composite_score_100'] for s in completed])),
                            'joint_pass_rate_100':float(np.mean([s['joint_pass_rate_100'] for s in completed])),
                            'intervals':intervals(completed),'complete_registered_arm':len(completed)==3}
    value={'updated_unix_s':time.time(),'stage':'complete' if all(r['score'] is not None for r in rows) else 'experiments_in_progress',
           'training_updates_completed':total_updates,'training_updates_target':32520*9,
           'runs':rows,'summaries':summaries,'counts':manifest['counts'],'calibration_passed':calibration['passed'],
           'video_generation_resumed':False}
    write_json(OUT/'status.json',value)
    lines=['# KeyCLD 多装配体平面闭链实验','',
           '使用 S-A 四杆、S-B 曲柄滑块、S-J 双曲柄。原版二维表示的三维能力未在本轮评估。',
           '', '数据已冻结：271 个训练、18 个验证、18 个独立校准、36 个测试装配体；对应 813 / 54 / 54 / 108 条三工况序列。',
           '', '**当前表格只记录实际完成情况；“待评估”不是 0 分。** 每个臂 40 轮、32,520 次更新，三个种子；状态臂是特权信息组件诊断。',
           '', '| 实验臂 | 种子 | 已完成更新 | 训练状态 | 连续 Scomp | 联合通过率 | 完整物理轨迹 |', '|---|---:|---:|---|---:|---:|---:|']
    for r in rows:
        show=lambda x:'待评估' if x is None else '%.2f'%x
        coverage='待评估' if r['complete_physical_trajectories'] is None else '%d / 108'%r['complete_physical_trajectories']
        lines.append('| %s | %d | %d / 32520 | %s | %s | %s | %s |'%(r['arm'],r['seed'],r['updates'],
                     '40 轮完成' if r['training_complete'] else r['status'],show(r['score']),show(r['joint']),coverage))
    lines+=['','## 独立评分器校准','',
            '真实轨迹 54/54 通过，静止预测 0/54，0.8 倍时间缩放 0/54。最大真值力矩残差 2.06×10⁻⁵ N·m、功—能误差 2.23×10⁻⁴ J。它们是测量校准结果，不是模型得分。',
            '', '评分用独立 SI 刚体方程，消去约束反力，不修正模型几何。未来控制力矩由预测状态上的公共伺服律计算，不输入未来实测力矩。',
            '', '物理关键点语义或几何未通过时，力平衡和能量项不可完整测量；这与已经测得某个错误力矩是不同结论。状态对照用于进一步检验动力学组件。',
            '', '只对实际完成的种子汇总；完整三种子比较和训练随机性结论需全部完成。时间预算内无法完成的预测保留失败原因和原分母。全零通过率的经验 bootstrap 区间会退化为 [0,0]，它不表示总体真实通过率的上界为零。']
    if summaries:
        lines+=['','## 已完成结果的区间','']
        for arm,s in summaries.items():
            lines.append('- %s：已完成 %d/3 种子；综合分 %.2f，95%% 区间 %s；联合通过率 %.2f，95%% 区间 %s。'%(arm,s['completed_seeds'],s['composite_score_100'],s['intervals']['composite_95pct'],s['joint_pass_rate_100'],s['intervals']['joint_95pct']))
    lines+=['','协议与差异见 `experiments/keycld_corpus/README.md`。历史二分综合分已归档；本报告统一采用连续容差比 Scomp；本轮也不单独证明摩擦值或 SI 质量的可辨识性。','']
    (OUT/'results.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
