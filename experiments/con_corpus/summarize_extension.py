"""Verify supplemental evidence and generate a reviewable Chinese results report."""
from collections import defaultdict,Counter
import csv
import hashlib
import json
from pathlib import Path
import math
import numpy as np
ROOT=Path(__file__).resolve().parents[2];EXT=ROOT/'output/con_corpus/dynamics_extension';DATA=ROOT/'output/keycld_corpus'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def balanced(rows,access):
    groups=defaultdict(list)
    for r in rows:
        v=access(r)
        if v is not None and np.isfinite(v):groups[(r['seed'],r['category'],r['condition'])].append(float(v))
    return dict(valid=sum(map(len,groups.values())),total=len(rows),mean=float(np.mean([np.mean(v) for v in groups.values()])) if groups else None)
def paired(rows,a,b):
    valid=[dict(r,difference=a(r)-b(r)) for r in rows if a(r) is not None and b(r) is not None]
    result=balanced(valid,lambda r:r['difference']);result['registered']=len(rows)
    families=sorted({r['category'] for r in rows});conditions=sorted({r['condition'] for r in rows});seeds=[42,43,44]
    assemblies={f:sorted({r['assembly_id'] for r in rows if r['category']==f}) for f in families}
    lookup={(r['seed'],r['assembly_id'],r['condition']):r['difference'] for r in valid}
    rng=np.random.default_rng(20261012);values=[]
    for _ in range(2000):
        picked=rng.choice(seeds,3,replace=True);draws={f:rng.choice(assemblies[f],len(assemblies[f]),replace=True) for f in families}
        cells=[]
        for s in picked:
            for f in families:
                for c in conditions:
                    cell=[lookup[(s,a,c)] for a in draws[f] if (s,a,c) in lookup]
                    if cell:cells.append(np.mean(cell))
        values.append(np.mean(cells))
    result['bootstrap_95pct']=np.percentile(values,[2.5,97.5]).tolist();return result

def main():
    protocol=read(EXT/'protocol.json');manifest=read(DATA/'dataset_manifest.json')
    expected={r['case_id'] for r in manifest['records'] if r['split']=='test'};assert len(expected)==108
    assert sha(DATA/'dataset_manifest.json')==protocol['dataset_manifest_sha256']
    calibration=read(EXT/'readout_calibration.json');assert calibration['passed']
    assert calibration['readout_source_sha256']==sha(ROOT/'experiments/con_corpus/crank_readout.py')
    paired_training=[];prediction_checks=0;file_checks=0
    for seed in [42,43,44]:
        reports=[]
        for kind in ['euler','rk4']:
            p=EXT/'runs'/f'state_{kind}_con_iae_seed{seed}';r=read(p/'training_report.json');inf=read(p/'inference_report.json')
            assert r['epochs']==50 and r['updates']==r['expected_updates']==2550 and r['samples_seen']==40650
            assert r['checkpoint_sha256']==sha(p/'params.p')==inf['checkpoint_sha256']
            assert r['identity']['train_source']==sha(ROOT/'experiments/con_corpus/train_integrator.py')
            assert r['identity']['protocol']==sha(EXT/'protocol.json')
            assert r['initial_checkpoint_sha256']==sha(p/'initial_params.p');reports.append(r)
            for prediction in (p/'predictions').glob('*/*.npz'):
                meta=read(prediction.with_suffix('.json'));assert sha(prediction)==meta['prediction_sha256'];prediction_checks+=1
        assert reports[0]['initial_checkpoint_sha256']==reports[1]['initial_checkpoint_sha256']
        assert reports[0]['epoch_data_order_sha256']==reports[1]['epoch_data_order_sha256'] and len(reports[0]['epoch_data_order_sha256'])==50
        paired_training.append(dict(seed=seed,initial_sha256=reports[0]['initial_checkpoint_sha256'],updates_each=2550,sequence_exposures_each=40650,identical_epoch_order=True))
    assert prediction_checks==972
    video={};solver={}
    for model in ['keycld','con']:
        all_rows=[];solver_rows=[]
        for seed in [42,43,44]:
            r=read(EXT/'video_diagnosis'/f'{model}_seed{seed}'/'report.json');assert {x['case_id'] for x in r['rows']}==expected
            assert r['identity']['source_sha256']==sha(ROOT/'experiments/con_corpus/video_diagnosis.py')
            all_rows.extend(dict(x,seed=seed) for x in r['rows'])
            r=read(EXT/'solver_diagnosis'/f'{model}_seed{seed}'/'report.json');assert {x['case_id'] for x in r['rows']}==expected
            assert r['identity']['source_sha256']==sha(ROOT/'experiments/con_corpus/solver_diagnosis.py')
            for row in r['rows']:
                path=EXT/'solver_diagnosis'/f'{model}_seed{seed}'/(row['case_id']+'.npz');assert sha(path)==row['trajectory_sha256'];file_checks+=1
            solver_rows.extend(dict(x,seed=seed) for x in r['rows'])
        video[model]=dict(rows=all_rows,image={k:balanced(all_rows,lambda r:r['image_mse'][k]) for k in all_rows[0]['image_mse']},
                          motion={k:balanced(all_rows,lambda r:r['motion_mask_mse'][k]) for k in all_rows[0]['motion_mask_mse']},
                          local={k:balanced(all_rows,lambda r:r['local_prediction'][k]['step4_readout_self_consistency_rmse_m']) for k in ['euler','rk4','latent_persistence']},
                          encoded_point=balanced(all_rows,lambda r:r['encoded_point_rmse_m']),
                          mean_point=balanced(all_rows,lambda r:r['train_mean_point_rmse_m']),
                          forecast_minus_persistence=paired(all_rows,lambda r:r['image_mse']['prefix_forecast'],lambda r:r['image_mse']['persistence']))
        solver[model]={method:{metric:balanced(solver_rows,lambda r:r['methods'][method][metric]) for metric in solver_rows[0]['methods'][method]} for method in solver_rows[0]['methods']}
    key_rows={(r['seed'],r['case_id']):r for r in video['keycld']['rows']}
    comparison=[]
    for r in video['con']['rows']:
        k=key_rows[(r['seed'],r['case_id'])]
        comparison.append(dict(r,keycld_forecast=k['image_mse']['prefix_forecast']))
    common_video={model:balanced(comparison,lambda r:r['image_mse']['prefix_forecast'] if model=='con' and r['keycld_forecast'] is not None else r['keycld_forecast'] if model=='keycld' else None) for model in ['keycld','con']}
    video_difference=paired(comparison,lambda r:r['image_mse']['prefix_forecast'],lambda r:r['keycld_forecast'])
    physics=defaultdict(list);csv_rows=[]
    for p in sorted((EXT/'physics_readout').glob('*/report.json')):
        r=read(p);identity=r['identity'];assert {x['case_id'] for x in r['rows']}==expected
        assert identity['source_sha256']==sha(ROOT/'experiments/con_corpus/score_readout.py')
        for row in r['rows']:
            if 'readout_prediction_sha256' in row:
                assert sha(p.parent/(row['case_id']+'.npz'))==row['readout_prediction_sha256'];file_checks+=1
            physics[(identity['model'],identity['arm'])].append(dict(row,seed=identity['seed']))
            for track in ['raw','conditioned']:
                csv_rows.append(dict(model=identity['model'],arm=identity['arm'],seed=identity['seed'],case_id=row['case_id'],track=track,
                                     coverage=row[track]['coverage'],failure_reason=row[track]['failure_reason'],correction_rms_m=row['correction_rms_m'],**row[track]['metrics']))
    assert len(physics)==7
    reproduction=[]
    for (model,arm),rows in physics.items():
        if model=='con_integrator':continue
        root=ROOT/('output/keycld_corpus' if model=='keycld' else 'output/con_corpus')
        for seed in [42,43,44]:
            old={r['case_id']:r for r in read(root/'runs'/f'{arm}_seed{seed}'/'scores.json')['rows']}
            largest=0.
            for row in (r for r in rows if r['seed']==seed):
                previous=old[row['case_id']]
                assert row['raw']['coverage']==previous['coverage']
                for metric,value in row['raw']['metrics'].items():
                    expected_value=previous['metrics'][metric]
                    if value is None:assert expected_value is None
                    else:
                        assert np.isclose(value,expected_value,rtol=1e-8,atol=1e-10),(model,arm,seed,row['case_id'],metric,value,expected_value)
                        largest=max(largest,abs(value-expected_value))
            reproduction.append(dict(model=model,arm=arm,seed=seed,max_absolute_metric_difference=largest))
    summaries={}
    for (model,arm),rows in physics.items():
        assert len(rows)==324
        entry={}
        for track in ['raw','conditioned']:
            entry[track]=dict(complete=sum(r[track]['coverage']==1 for r in rows),total=324,
                              metrics={m:balanced(rows,lambda r:r[track]['metrics'][m]) for m in rows[0][track]['metrics']},
                              component_scores_100={m:100*balanced(rows,lambda r:float(r[track].get('passed',{}).get(m,False)))['mean'] for m in rows[0][track]['metrics']},
                              joint_passes=sum(r[track]['joint_pass'] for r in rows),failures=dict(Counter(r[track]['failure_reason'] for r in rows if r[track]['failure_reason'])))
        entry['correction_rms_m']=balanced(rows,lambda r:r['correction_rms_m']);summaries[f'{model}/{arm}']=entry
    euler={(r['seed'],r['case_id']):r for r in physics[('con_integrator','state_euler_con_iae')]};pair_rows=[]
    for r in physics[('con_integrator','state_rk4_con_iae')]:pair_rows.append(dict(r,euler=euler[(r['seed'],r['case_id'])]))
    integrator_effect={}
    for track,metric in [('raw','prediction_m'),('conditioned','force_balance_Nm'),('conditioned','work_energy_J')]:
        integrator_effect[f'{track}/{metric}']=paired(pair_rows,lambda r:r[track]['metrics'][metric],lambda r:r['euler'][track]['metrics'][metric])
    summary=dict(stage='verified_complete',protocol_sha256=sha(EXT/'protocol.json'),paired_training=paired_training,
                 prediction_files_verified=prediction_checks,additional_trajectory_files_verified=file_checks,native_score_reproduction=reproduction,
                 video={k:{a:b for a,b in v.items() if a!='rows'} for k,v in video.items()},
                 video_common_finite=common_video,con_minus_keycld_video_mse=video_difference,physics=summaries,
                 con_rk4_minus_euler=integrator_effect,solver=solver,
                 aggregation='Equal seed/family/condition cells; finite-only error means with all registered denominators and failures reported; paired bootstrap 2000 seed/assembly draws, seed20261012.',
                 manuscript_status='Experiment evidence complete; manuscript reconciliation follows verification.')
    (EXT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    with (EXT/'physical_case_metrics.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)
    lines=['# KEYCLD / CON：补充诊断与动力学证据','',
      '本报告在原结果已知后补做，保留原生结果，未用测试集重新调参。论文中的两模型对齐必须分别说明视频诊断、积分器控制和物理读出。','',
      '## 已完成与核验','',
      '- CON 状态 Euler / RK4：3 对种子，6 个新模型，每个 50 轮、2,550 次更新、40,650 次序列使用；逐对核验相同初始化和 50 轮样本顺序。',
      '- 972 个新原生预测文件（648 测试、324 验证）完成哈希核验。',
      '- 两模型各 324 条视频诊断：逐帧重建、三帧初始化未来预测、保持与训练均值对照、运动像素误差，以及每条 84 个四步重置窗口。',
      '- 两模型 × 3 种子 × 108 测试 × Euler / RK4 各 1、4、16 子步，共 3,888 条固定权重积分轨迹。',
      '- 21 个种子级模型臂 × 108 测试，共 2,268 组原生 / 几何读出动力学检验。','',
      '## 视频重建与预测','',
      '| 模型 | 逐帧重建 MSE | 三帧初始化未来 MSE | 有限未来预测 | 图像保持 MSE | 四步局部 Euler / 保持误差 (mm) |',
      '|---|---:|---:|---:|---:|---:|']
    for model in ['keycld','con']:
        v=video[model];a=v['image'];l=v['local']
        lines.append(f"| {model.upper()} | {a['framewise_reconstruction']['mean']:.8f} | {a['prefix_forecast']['mean']:.8f} | {a['prefix_forecast']['valid']}/324 | {a['persistence']['mean']:.8f} | {l['euler']['mean']*1000:.3f} / {l['latent_persistence']['mean']*1000:.3f} |")
    lines+=['','逐帧重建和重置窗口读取后续真实图像，仅用于定位问题；三帧初始化未来预测使用已冻结原生预测。局部毫米误差衡量各自读出的自洽性，CON 的读出为训练标签拟合的仿射映射，不能据此把两种潜在空间当作同一物理表示。',
            f"同一批 {common_video['keycld']['valid']}/324 个共同有限案例上，KEYCLD / CON 的未来 MSE 为 {common_video['keycld']['mean']:.8f} / {common_video['con']['mean']:.8f}。其余 13 个 KEYCLD 原生积分超时保留为失败；有限样本均值不能代替完整覆盖率。",'',
            '## 力矩与功—能：原生轨迹和附加读出','',
            '附加读出只用预测曲柄端点 B 及其速度、加速度和公开几何，解码曲柄相位并沿已知闭链重建位置与解析导数；质量、惯量、摩擦及未来真值只进入独立评估器。它是一个显式几何干预，不能替换原生几何结果。',
            '54 条独立校准真值全部通过；静止和 0.8 倍时间缩放均为 0/54。真值读出修正 RMS 最大 2.481 μm，力矩残差最大 0.000217 N·m，功—能缺陷最大 0.000121 J。','',
            '| 模型 / 臂 | 原生完整覆盖 | 读出完整覆盖 | 读出修正 RMS (mm) | 力矩残差 RMS (N·m) | 功—能缺陷 (J) | 读出后点误差 (mm) |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for name,v in summaries.items():
        m=v['conditioned']['metrics'];lines.append(f"| {name} | {v['raw']['complete']}/324 | {v['conditioned']['complete']}/324 | {v['correction_rms_m']['mean']*1000:.4f} | {m['force_balance_Nm']['mean']:.4f} | {m['work_energy_J']['mean']:.4f} | {m['prediction_m']['mean']*1000:.3f} |")
    lines+=['','上表力矩、能量和修正误差均值只在可解释轨迹上计算，覆盖率明确列出；曲柄方向退化与原生超时不插值补齐。力矩项是独立方程的驱动坐标等效残差，能量项是功—能缺陷，不是模型直接输出的 SI 参数。','',
            '## CON 成对训练积分器效应','',
            '| 指标（RK4 − Euler） | 配对可用案例 | 均值差 | 95% 种子 / 装配体 bootstrap 区间 |',
            '|---|---:|---:|---|']
    for k,v in integrator_effect.items():lines.append(f"| {k} | {v['valid']}/324 | {v['mean']:.6g} | [{v['bootstrap_95pct'][0]:.6g}, {v['bootstrap_95pct'][1]:.6g}] |")
    lines+=['','三个种子提供训练随机性信息；不能把 KEYCLD 的 RK4 改善结论直接移植到 CON。两者原生训练时域、损失和预算仍有区别。','',
            '## 固定权重推理积分器','',
            '| 模型 | 求解器 / 子步 | 有限状态 | 未来 RGB MSE | 对原生点轨迹的差异 (mm) |',
            '|---|---|---:|---:|---:|']
    for model,methods in solver.items():
        for method,v in methods.items():
            finite=int(round(v['finite']['mean']*324));x=v['native_difference_rmse_m']['mean'];rgb=v['rgb_mse']['mean']
            lines.append(f"| {model.upper()} | {method} | {finite}/324 | {rgb:.8f} | {x*1000:.4g} ({v['native_difference_rmse_m']['valid']}/324) |")
    lines+=['','有限状态不保证数值稳定：粗步长 Euler / RK4 中存在极大的有限坐标离群值，表中保留，不裁剪。原生轨迹差异列括号给出共同有限样本数。固定权重求解器对照没有重训，也不替换原生排名。部分原生超时案例仅重新计算原三帧初始化后运行有界固定步数预测，逐案例记录 initial_source。','',
            '## 证据文件','',
            '- `summary.json`：统计量、配对区间与文件核验数量。',
            '- `physical_case_metrics.csv`：全部 4,536 行原生 / 读出测量和失败原因。',
            '- `runs/`：6 个新模型的权重、训练顺序、日志和原生预测。',
            '- `video_diagnosis/`：6 份完整视频诊断及对应示例。',
            '- `solver_diagnosis/`：6 份固定权重积分对照，保留全部求解器状态轨迹。',
            '- `physics_readout/`：21 份动力学报告及附加读出轨迹。',
            '- `protocol.json`、`solver_protocol.json`、`readout_calibration.json`：追加协议和独立校准。','']
    (EXT/'results.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in summary.items() if k not in ['video','physics','solver']},indent=2))
if __name__=='__main__':main()
