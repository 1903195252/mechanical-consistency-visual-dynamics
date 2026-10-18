"""Present actual paired diagnostic results, without upgrading causal claims."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/keycld_fourbar/diagnosis'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--report-only',action='store_true',help='Refresh prose and tables without regenerating unchanged media')
    args=parser.parse_args()
    report=json.loads((OUT/'report.json').read_text())
    rows=json.loads((OUT/'learned_rows.json').read_text())
    oracle=json.loads((OUT/'oracle_rows.json').read_text())
    s=report['summary'];v=s['visual_state']
    lines=['# KeyCLD 四杆受控诊断', '',
           '视频基线固定第 40 轮权重、50 条同分布独立测试、输入和动作，原始结果保留。固定权重对照不更新参数；另有两组真实状态监督的组件训练，见后文。', '',
           '## 视觉与推理积分对照', '',
           '| 条件 | VPT 索引均值 ± 标准差 ↑ | 未来图像 MSE ↓ | 运动像素 MSE ↓ | 有限输出 |',
           '|---|---:|---:|---:|---:|']
    names={'odeint':'原版 odeint', 'framewise_oracle':'逐帧重建（读取真实未来帧）',
           'persistence':'最后观测图像保持', 'training_mean':'训练集平均图像',
           **{f'{k}_{n}':f'{k.upper()} 每间隔 {n} 子步' for k in ['rk4','euler'] for n in [1,4,16]}}
    for name in ['odeint','persistence','framewise_oracle','training_mean','rk4_1','rk4_4','rk4_16','euler_1','euler_4','euler_16']:
        a=s['vpt'][name];b=s['image_mse'][name];c=s['foreground_mse'][name]
        if not a['n']:
            lines.append(f'| {names[name]} | — | — | — | 0/50 |')
        else:
            lines.append(f"| {names[name]} | {a['mean']:.2f} ± {a['std']:.2f} | {b['mean']:.7g} | {c['mean']:.7g} | {a['n']}/50 |")
    lines+=['', '逐帧重建每个时刻都使用该时刻的真实图像，仅用于移除未来预测环节；它不属于可部署的未来预测成绩。运动像素掩码由每条测试视频的时间方差定义，只供评分使用。', '',
            '## 关键点物理语义', '',
            f"- 声明顺序 B/C/中点的编码位置 RMSE：{v['encoded_rmse_declared_m']['mean']*1000:.2f} mm。",
            f"- 验证集选定全局排列 {report['semantic_mapping']['validation_selected_permutation']} 后：{v['encoded_rmse_validation_permutation_m']['mean']*1000:.2f} mm；不拟合仿射映射，也不修改预测。",
            f"- 训练集平均物理点位置对照 RMSE：{v['constant_position_rmse_m']['mean']*1000:.2f} mm。",
            f"- 编码器的五项位置约束 RMSE：{v['encoded_constraint_rmse_m']['mean']*1000:.2f} mm；每序列最大杆长误差的均值 {v['encoded_distance_max_error_m']['mean']*1000:.2f} mm。", '',
            '本节位置 RMSE 按坐标分量计算，再对序列取均值。点位置误差同时受无监督关键点语义、像素分辨率和少量视差影响，不能直接归因于某一个因素。', '',
            f"原版 odeint 的位置约束漂移 RMSE 均值为 {s['solver_constraint_drift_m']['odeint']['mean']*1e6:.3f} μm（相对初始化时的约束值）。该量测量约束值是否变化，不代表约束值一开始就是零。",
            '模型代码通过约束的雅可比与 Hessian 约束加速度；初始化仅投影速度，没有把编码位置投影到零约束流形。原版损失也没有直接加入杆长位置残差。因此稳定保持非零约束值与违反初始几何可以同时发生。', '',
            '## 局部动力学诊断', '',
            '| 局部控制 | 1 步潜在点 RMSE (mm) | 4 步潜在点 RMSE (mm) |',
            '|---|---:|---:|']
    for name,value in s['local_latent_prediction'].items():
        lines.append(f"| {name} | {value['one_step_rmse_m']['mean']*1000:.3f} | {value['four_step_rmse_m']['mean']*1000:.3f} |")
    lines+=['', '每条序列有 44 个固定局部窗口，重置为真实图像编码和中心差分速度，只比较窗口内预测与后续编码。它读取未来图像，属于诊断。Euler 的第一步位置仅由初速度决定，因此 1 步误差不能单独检验所学加速度。', '',
            '## 独立物理方程与采样', '',
            '| 真实状态起步的物理对照 | 铰链位置 RMSE 均值 (mm) ↓ | 有限输出 |',
            '|---|---:|---:|']
    for name,value in s['oracle_position_rmse_m'].items():
        error=f"{value['mean']*1000:.3f}" if value['mean'] is not None else '—'
        lines.append(f"| {name} | {error} | {value['n']}/50 |")
    lines+=['', '物理对照的未绕回曲轴相位误差保存在 `oracle_rows.json`，与周期性铰链位置误差一起检查。数值发散计为失败，缺失值不以零填补；有限输出均值须结合表中成功数解读。']
    lines+=['', f"100 Hz 三帧中心差分的初始曲轴速度绝对误差均值：{s['initial_omega_abs_error_rad_s']['mean']:.4f} rad/s。",
            f"各序列最大相邻帧曲轴转角的均值：{v['max_sample_crank_displacement_rad']['mean']:.3f} rad；超过 π 的相邻采样区间总数：{s['sampling_alias_intervals_above_pi_total']}。未超过 π 也不能排除像素层面的模糊或有限差分误差。", '',
            '物理对照使用独立 NumPy 受约束动力学、真实质量/惯量/外力及初态，不是神经模型成绩。DOP853 理想约束与 MuJoCo 有限柔度约束之间仍可能有微小误差。', '',
            f"将真实物理点坐标和速度直接输入已训练动力学网络时，加速度相对 L2 误差的序列均值为 {v['learned_acceleration_true_coordinate_relative_error']['mean']:.4f}。若编码语义不匹配，这属于坐标域外探测，不能单独判定闭链动力学学习失败。", '',
            '## 配对统计', '',
            '| 差值（前者减后者） | 未来 MSE 均值差 [95% 区间] | VPT 均值差 [95% 区间] |',
            '|---|---:|---:|']
    for key in report['paired']['image_mse']:
        def format_difference(metric):
            value=report['paired'][metric][key]
            if not value.get('n'):return '—'
            lo,hi=value['bootstrap_95_percent_interval']
            return f"{value['mean_a_minus_b']:.6g} [{lo:.6g}, {hi:.6g}]"
        lines.append(f'| {key} | {format_difference("image_mse")} | {format_difference("vpt")} |')
    lines+=['', '配对 bootstrap 2000 次，种子 20261010；区间反映这 50 条轨迹间的差异，不涵盖训练随机种子变化。VPT 沿用原作者的 0..49 索引语义。', '',
            '## 结论边界', '',
            '推理积分器对照不检验训练损失中的 Euler 偏差；独立物理方程的 Euler 对照用于量化该时间尺度的数值难度。',
            '闭链特有的学习困难仍需同图像难度、运动速度、自由度和训练预算的拓扑对照。若关键点未建立物理语义，则须先隔离表征坐标与动力学，再判定动力学学习原因。',
            '现有测试集已经用于诊断，后续调参后的确认性评价应另设未见测试种子。', '',
            f"复算原版未来图像 MSE 的最大绝对差：{report['baseline_future_mse_max_abs_difference']:.3g}。最终复算执行耗时 {report['elapsed_seconds']:.1f} s，复用了此前完成的 50 条物理对照；该时长不计缓存生成成本。原训练只有 seed=42。", '',
            '完整数值见 `report.json`、`learned_rows.json`、`oracle_rows.json`；执行代码和预先固定的变量见 `diagnose.py` 与 `diagnosis_protocol.json`。']
    sampling_path=OUT/'sampling_audit.json'
    if sampling_path.exists():
        audit=json.loads(sampling_path.read_text())
        lines+=['', '## 初始采样与整个训练序列的采样分别检查', '',
                '使用真实物理关键点计算 100 Hz 中心差分，再使用给定速度约束进行欧氏投影；计算中没有图像识别误差。', '',
                '| 数据 | 序列数 | 原始差分速度相对误差均值 | 投影后相对误差均值 | 投影后最差序列 |',
                '|---|---:|---:|---:|---:|']
        for group,values in audit['groups'].items():
            q=values['summary']
            lines.append(f"| {group} | {values['run_count']} | {q['raw_velocity_relative_error']['mean']*100:.2f}% | {q['projected_velocity_relative_error']['mean']*100:.2f}% | {q['projected_velocity_relative_error']['max']*100:.2f}% |")
        lines+=['', '前三帧的初速度误差较小，不代表整段训练数据上的差分速度准确。训练后半段的高速运动存在显著差分偏差；它是采样层面的已测误差，训练效果的因果贡献还需要固定积分器、仅替换速度估计的配对训练。完整逐序列值见 `sampling_audit.json`。']
    force_path=OUT/'force_probe.json'
    if force_path.exists():
        force=json.loads(force_path.read_text())['summary']
        lines+=['', '## 给定约束曲率与学习驱动力', '',
                '| 真实物理坐标上的探测 | 总点加速度相对误差 ↓ | 置零速度后的曲轴加速度相对误差 ↓ | 真实初态下学习方程回放的位置 RMSE (mm) ↓ |',
                '|---|---:|---:|---:|']
        for name,values in force.items():
            position=values.get('oracle_initialization_position_rmse_m',{}).get('mean')
            position='—' if position is None else f'{position*1000:.2f}'
            lines.append(f"| {name} | {values['full_point_acceleration_relative_error']['mean']:.4f} | {values['zero_velocity_crank_acceleration_relative_error']['mean']:.4f} | {position} |")
        lines+=['', '零驱动力对照只将势能与动作矩阵网络置零，保留训练得到的质量和所有已知约束；约束曲率依然会产生加速度。置零速度去掉速度二次项后，正确的加速度必须由驱动力与重力决定。相对误差 0 为完美，1 对应零加速度对照。',
                '当前训练权重在这些真实坐标探测上没有显示优于零驱动力的优势。由于真实坐标不属于编码器实际使用的坐标语义，不能单独用这项探测证明模型在自身潜在域完全未学习动力学。完整结果见 `force_probe.json`。']
    state_root=OUT/'state_oracle'
    completed={arm:json.loads((state_root/arm/'report.json').read_text()) for arm in ['euler','rk4_4'] if (state_root/arm/'report.json').exists()}
    if completed:
        lines+=['', '## 绕过视觉的完整预算状态训练', '',
                '两组从同一原始随机权重开始，提供真实关键点与精确速度，保留原动力学网络和 4 步动力学损失。每组 450 条训练、50 条验证、40 轮和 18,000 次 Adam 更新，仅训练积分器不同。它们是使用状态标签的组件诊断，不是论文视频方法成绩。', '',
                '| 训练积分器 | 测试坐标分量 RMSE (mm) ↓ | 静态驱动力加速度相对误差 ↓ | 有限测试 | 耗时 (s) |',
                '|---|---:|---:|---:|---:|']
        for arm,value in completed.items():
            a=value['summary']['position_rmse_m']['mean'];b=value['summary']['zero_velocity_crank_acceleration_relative_error']['mean']
            lines.append(f"| {arm} | {a*1000:.3f} | {b:.4f} | {value['finite_test_runs']}/50 | {value['elapsed_seconds']:.1f} |")
        lines+=['', '测试时均用原版 odeint 和真实中间观测时刻的初态，最终第 40 轮权重固定；不按测试集或验证集挑选权重。两组更新数相同，RK4 算力成本更高。当前只包含一个配对训练随机种子，不能推断所有机构或所有随机初始化。']
        if len(completed)==2:
            from collections import defaultdict
            differences=defaultdict(list)
            for a,b in zip(completed['rk4_4']['rows'],completed['euler']['rows']):
                assert a['index']==b['index']
                for key in ['position_rmse_m','zero_velocity_crank_acceleration_relative_error']:
                    if a[key] is not None and b[key] is not None:differences[key].append(a[key]-b[key])
            if (state_root/'physics_scores.json').exists():
                physics=json.loads((state_root/'physics_scores.json').read_text())['conditions']
                key='independent_tangent_force_residual_rmse_Nm'
                for a,b in zip(physics['rk4_4']['rows'],physics['euler']['rows']):
                    assert a['index']==b['index']
                    differences[key].append(a['physics'][key]-b['physics'][key])
            rng=np.random.RandomState(20261010);paired={}
            for key,values in differences.items():
                values=np.asarray(values);boot=rng.choice(values,(2000,len(values))).mean(axis=1)
                paired[key]={'n':len(values),'rk4_minus_euler_mean':float(values.mean()),
                             'bootstrap_95_percent_interval':np.percentile(boot,[2.5,97.5]).tolist()}
                unit=1000 if key=='position_rmse_m' else 1
                lo,hi=paired[key]['bootstrap_95_percent_interval']
                suffix=' mm。' if unit==1000 else ' N·m。' if key.endswith('_Nm') else '。'
                lines.append(f"\n配对 {key} 差值（RK4−Euler）：{values.mean()*unit:.5g}，95% 区间 [{lo*unit:.5g}, {hi*unit:.5g}]"+suffix)
            (state_root/'paired_statistics.json').write_text(json.dumps(paired,indent=2)+'\n')
        physics_path=state_root/'physics_scores.json'
        if physics_path.exists():
            physics=json.loads(physics_path.read_text())['conditions']
            lines+=['', '### 预测轨迹的独立力矩方程检验', '',
                    '| 训练积分器 | 原始杆长 RMSE (mm) ↓ | 独立切向力矩残差 RMS (N·m) ↓ | 1 cm诊断容差内的未来帧 |',
                    '|---|---:|---:|---:|']
            for arm,value in physics.items():
                q=value['summary']
                lines.append(f"| {arm} | {q['native_link_length_rmse_m']['mean']*1000:.5g} | {q['independent_tangent_force_residual_rmse_Nm']['mean']:.5g} | {value['geometry_consistent_future_frames']}/{value['total_future_frames']} |")
            lines+=['', '用模型自身预测的位置、速度、加速度解析计算杆角状态，再代入独立 SI 质量矩阵、科氏项、重力、已知力矩和外力方程；通过闭环雅可比的单位零空间消除未知约束反力。不拟合反力，不投影位置回真值。1 cm沿用旧诊断容差，不能作为正式物理通过阈值；原始误差另列。']
            lines+=['', '`physics_scores.json` 的 `keypoint_position_rmse_m` 采用每点欧氏距离 RMSE；状态训练表中的 `position_rmse_m` 则采用坐标分量 RMSE。当前二维 B/C 坐标下，两者相差 √2，不能将两个字段混用。', '',
                    '两组的质量矩阵伪逆乘原矩阵与单位矩阵的最大差均约为浮点精度，没有在本次权重中观察到质量矩阵秩丢失。学习质量系数存在质量/驱动力整体尺度自由度，不解读为已识别的 SI 质量。']
            audit=json.loads(physics_path.read_text())['length_metadata_audit']
            lines+=['', f"杆长元数据审计：在 seed=500 的 {audit['states_checked']} 个真实状态上，仅将给定杆长改为 2 倍，位置约束值最大改变约 {audit['position_constraint_max_abs_change_m']:.2f} m，而约束 Jacobian、Hessian 和模型加速度改变均为零。这验证了等式常数在求导后的消失；固定锚点和拓扑仍参与导数，正确杆长需要合法初态。"]
        if len(completed)==2:
            a=completed['euler']['summary']['position_rmse_m']['mean']*1000
            b=completed['rk4_4']['summary']['position_rmse_m']['mean']*1000
            lead=[f'该实验观察到视觉物理语义和时间差分偏差，并在真实状态输入的控制下测得训练积分器的影响。关键点排列校正后仍有约 {v["encoded_rmse_validation_permutation_m"]["mean"]*1000:.1f} mm 坐标误差；细分推理积分器没有改变原视频方法的 VPT=7.82。', '',
                  f'提供真实状态、固定训练预算后，Euler 与 RK4 的独立测试坐标分量误差分别为 {a:.2f} mm、{b:.2f} mm。该对照属于有状态标签的组件诊断，不能替代视频方法成绩，也没有检验完整视频模型更换训练积分器后的效果。现有证据不能将问题唯一归为闭链拓扑本身。', '']
            lines[2:2]=lead
    physics_path=state_root/'physics_scores.json'
    if physics_path.exists():
        physics=json.loads(physics_path.read_text())['conditions']
        primary=['## 首要证据：独立受约束动力学方程', '',
                 '该实验支持的核心判断是：几何闭合、图像重建与动力学正确性需要分别检验。以下两组使用真实状态标签训练和初始化，均完成 40 轮、18,000 次更新，使用相同的 50 条测试轨迹。它们是动力学组件对照，不是视频输入方法成绩。', '',
                 '| 组件训练积分器 | 独立切向广义力矩残差 RMS 均值 (N·m) ↓ | 杆长 RMSE 均值 (μm) |',
                 '|---|---:|---:|']
        for arm,value in physics.items():
            q=value['summary']
            primary.append(f"| {arm} | {q['independent_tangent_force_residual_rmse_Nm']['mean']:.6f} | {q['native_link_length_rmse_m']['mean']*1e6:.3f} |")
        primary+=['', '两组杆长误差都很小，仍存在独立受力平衡误差。RK4 降低了动力学残差，但不能因此宣称已通过正式物理检验；通过阈值尚未预先制定。', '',
                  '$$r_{\\mathrm{dyn}}=n(q)^\\top\\left[M(q)\\ddot q+c(q,\\dot q)+G(q)-Q_{\\mathrm{motor}}-Q_{\\mathrm{ext}}-Q_{\\mathrm{fric}}\\right],\\qquad J(q)n(q)=0,\\quad \\|n(q)\\|=1.$$', '',
                  '这里的位置、速度、加速度来自模型预测；质量矩阵、科氏项、重力、载荷惯量与外力来自独立的 SI 刚体方程。单位切向向量消除未知约束反力，未拟合反力掩盖误差。这是广义力矩的切向投影残差，不能直接称为驱动轴电机力矩误差。', '',
                  '该指标先要求物理坐标可解释、几何合法、初态和动作一致。低残差还需结合动作干预及条件变化后的响应检验，不能单独用来证明模型学会了完整动力学。', '',
                  '当前主训练工况摩擦为零、机构几何尺寸与标称载荷固定，初始姿态随机。因此现有结果没有验证摩擦辨识、未见载荷适应或跨机构动力学泛化，也没有证明学习质量参数等于真实 SI 质量。', '',
                  '后续评测优先级：独立力/力矩平衡与动作到加速度的响应；配对质量、惯量、载荷和摩擦变化；功—能平衡（需控制数值积分误差）。视频和位置指标用于诊断表征与运动学，不替代这些动力学证据。', '',
                  '已有证据支持动力学评测的必要性，尚不能推断全部模型或所有闭链机构的能力。以下保留视觉、采样与积分对照作为辅助定位。', '']
        lines[2:2]=primary
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    if args.report_only:
        print('Refreshed dynamics-first report; numerical results and media preserved.',flush=True)
        return

    font=FontProperties(fname=FONT)
    plt.rcParams['font.family']=font.get_name()
    fig,axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    choices=['persistence','odeint','framewise_oracle','rk4_16']
    axes[0].bar(range(4),[s['image_mse'][k]['mean'] for k in choices],color=['#8e9da9','#ef8d36','#328e9c','#7e68a7'])
    axes[0].set_xticks(range(4),['图像保持','原版预测','逐帧重建*','RK4/16'],fontproperties=font,rotation=15)
    axes[0].set_title('未来图像误差（越低越好）',fontproperties=font);axes[0].set_ylabel('MSE')
    choices=['dop853_exact','dop853_sampled_velocity','euler_1','euler_4','euler_16','rk4_4']
    values=[s['oracle_position_rmse_m'][k]['mean']*1000 for k in choices]
    axes[1].bar(range(6),values,color='#328e9c');axes[1].set_yscale('log')
    axes[1].set_xticks(range(6),['真速度','差分速度','Euler/1','Euler/4','Euler/16','RK4/4'],fontproperties=font,rotation=30)
    axes[1].set_title('独立方程回放的位置误差',fontproperties=font);axes[1].set_ylabel('mm (log scale)')
    for name,color in [('euler_1','#ef8d36'),('rk4_4','#328e9c'),('latent_persistence','#8e9da9')]:
        values=[s['local_latent_prediction'][name][k]['mean']*1000 for k in ['one_step_rmse_m','four_step_rmse_m']]
        axes[2].plot([1,4],values,'o-',label=name,color=color)
    axes[2].set_xticks([1,4]);axes[2].set_xlabel('预测步数',fontproperties=font);axes[2].set_ylabel('mm')
    axes[2].set_title('局部潜在点预测误差',fontproperties=font);axes[2].legend(fontsize=8)
    for axis in axes:
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18)
    fig.suptitle('固定权重 · 相同 50 条测试 · *逐帧重建使用真实未来帧',fontproperties=font,fontsize=12)
    fig.savefig(OUT/'diagnosis.png',dpi=160);plt.close(fig)
    if completed:
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for arm,color in [('euler','#ef8d36'),('rk4_4','#328e9c')]:
            if arm not in completed:continue
            history=json.loads((state_root/arm/'history.json').read_text())
            axes[0].plot([r['epoch'] for r in history],[r['dynamics_loss'] for r in history],label=arm,color=color)
            axes[1].plot([r['epoch'] for r in history],[r['validation_position_rmse_m']['mean']*1000 for r in history],label=arm,color=color)
        for axis in axes:
            axis.set_xlabel('训练轮数',fontproperties=font);axis.legend();axis.grid(alpha=.18)
            axis.spines[['top','right']].set_visible(False)
        axes[0].set_yscale('log');axes[0].set_title('四步动力学损失',fontproperties=font)
        axes[1].set_ylabel('mm');axes[1].set_title('验证集位置 RMSE',fontproperties=font)
        fig.suptitle('真实状态输入的组件诊断 · 每组40轮 · 不计作视频方法成绩',fontproperties=font,fontsize=12)
        fig.savefig(state_root/'training_comparison.png',dpi=160);plt.close(fig)
    trajectory_path=state_root/'trajectory_example.npz'
    if trajectory_path.exists():
        with np.load(trajectory_path) as saved:trajectory={k:saved[k] for k in saved.files}
        fig,axes=plt.subplots(2,2,figsize=(10,5.5),layout='constrained')
        for point,name in enumerate(['B','C']):
            for coordinate,axis_name in enumerate(['x','z']):
                axis=axes[point,coordinate]
                for key,label,color in [('reference_world_xz','reference','#263e50'),('euler_predicted_world_xz','Euler','#ef8d36'),('rk4_4_predicted_world_xz','RK4/4','#328e9c')]:
                    axis.plot(trajectory['time'],trajectory[key][:,point,coordinate],label=label,color=color,linewidth=1.6)
                axis.set_title(f'{name}: {axis_name} (m)');axis.set_xlabel('t (s)');axis.grid(alpha=.18)
                axis.spines[['top','right']].set_visible(False)
        axes[0,0].legend(fontsize=8)
        fig.suptitle('独立测试 seed=500 · 真实状态初始化 · 两组状态训练的最终权重',fontproperties=font,fontsize=12)
        fig.savefig(state_root/'trajectory_comparison.png',dpi=160);plt.close(fig)

    with np.load(OUT/'visual_example.npz') as saved:data={k:saved[k] for k in saved.files}
    face=ImageFont.truetype(FONT,18,index=2);small=ImageFont.truetype(FONT,15,index=2)
    labels=['真实图像 + 真值点','真实图像 + 编码点','逐帧重建（未来帧诊断）','原版未来预测']
    colors=['#ee852e','#39bad4','#dc6cb4']
    def frame(i):
        canvas=Image.new('RGB',(1024,318),'#152330');draw=ImageDraw.Draw(canvas)
        rgbs=[data['reference_rgb'][i],data['reference_rgb'][i],data['reconstructed_rgb'][i],data['predicted_rgb'][i]]
        for j,rgb in enumerate(rgbs):
            canvas.paste(Image.fromarray(rgb).resize((256,256),Image.Resampling.NEAREST),(256*j,30))
            draw.text((256*j+7,6),labels[j],font=face,fill='white')
            if j<2:
                coordinates=data['physical_normalized' if j==0 else 'encoded_normalized'][i]
                for p,color in zip(coordinates,colors):
                    x=256*j+(p[0]+1)*63/2*4+2;y=30+(1-p[1])*63/2*4+2
                    draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        draw.text((8,293),f"测试 seed=500 · t={data['time'][i]:.2f}s · 5倍慢放 · 橙/青/粉对应三个点",font=small,fill='#cad6df')
        return np.asarray(canvas)
    writer=cv2.VideoWriter(str(OUT/'visual_diagnosis.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1024,318))
    assert writer.isOpened()
    for i in range(50):writer.write(cv2.cvtColor(frame(i),cv2.COLOR_RGB2BGR))
    writer.release();Image.fromarray(frame(10)).save(OUT/'visual_diagnosis_preview.png')
    print('Wrote results.md, diagnosis.png, visual_diagnosis.mp4',flush=True)


if __name__=='__main__':main()
