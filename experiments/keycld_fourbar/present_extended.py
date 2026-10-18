"""Dynamics-first presentation of actual frozen-weight experiments."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/keycld_fourbar/extended_dynamics'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
NAMES={'nominal':'标称无摩擦','viscous':'粘性摩擦','coulomb':'库仑摩擦','mixed_friction':'混合摩擦',
       'heavy_payload':'载荷0.25 kg','dense_bars':'杆件密度×1.5','geometry_scaled':'尺寸×0.8','crank_slider':'曲柄滑块'}


def main():
    report=json.loads((OUT/'evaluation_report.json').read_text())
    probes=json.loads((OUT/'field_probes.json').read_text())
    controls=json.loads((OUT/'identifiability_controls.json').read_text())
    examples=json.loads((OUT/'equation_examples.json').read_text())
    manifest=json.loads((OUT/'execution_manifest.json').read_text())
    lines=['# 摩擦、SI 参数与跨机构动力学评估', '',
           '完成 8 工况 × 20 配对轨迹、两个冻结模型，共 320 次预测。配对种子使用相同初始曲柄角、初速度和恒定力矩；没有重新训练或按测试结果选择权重。', '',
           '主要证据来自独立动力学方程。视频模型的原始关键点未形成合法物理状态，不能据此计算有物理解释的力矩成绩；提供真实初态的 RK4 状态组件用于隔离视觉误差，属于状态监督诊断，不是视频方法成绩。', '',
           '参考仿真使用 MuJoCo 3.2.3；冻结模型使用 JAX。', '',
           '## 一、以曲柄轴 N·m 为单位的动力学泛化', '',
           '| 条件 | 真实状态组件完整合法轨迹 | 曲柄轴等效力矩误差 RMS 均值 ± 标准差 (N·m) ↓ | 视频模型完整合法轨迹 |',
           '|---|---:|---:|---:|']
    for name,groups in report['conditions'].items():
        state=groups['state_RK4'];video=groups['video'];score=state['motor_torque_error_rms_Nm']
        lines.append(f"| {NAMES[name]} | {state['complete_physical_runs']}/20 | {score['mean']:.6f} ± {score['std']:.6f} | {video['complete_physical_runs']}/20 |")
    lines+=['', f"输入力矩的全组 RMS 参考值为 {report['torque_reference_Nm']:.6f} N·m。每条序列只评估前三帧之后的 47 个未来状态，表中统计先对每条轨迹计算 RMS，再对 20 条取均值和标准差。", '',
            '该实验残差采用曲柄参数化 γ=dq/dθ、γ₁=1，直接得到驱动轴等效力矩误差。上轮的单位化零空间切向广义力矩残差采用不同归一化，不能直接比较数值。曲柄滑块包含一个线性位移坐标，γ 换算保证 Nm 单位一致。', '',
            '几何诊断容差在模型评分前固定为 100 μm，来源为数据审计界限的 5 倍，不能当作 64 像素视频的通用精度标准。本批视频输出的整段最大几何误差也均超过 1 cm，因此结论不依赖这一更紧容差。几何不合法时动力学指标记为不可用，不记为零。', '',
            '迁移条件只提供目标的几何与约束，保留全部学习权重；未提供目标质量、惯量和摩擦。尺寸缩放同时改变杆件质量和惯量，属于几何与相应物理量联合变化。曲柄滑块是 3R+1P 新拓扑，该实验未覆盖 Delta、Stewart 或多自由度复杂并联机构。', '',
            '## 二、摩擦辨识：同姿态、正反速度', '',
            r'$$\frac{\alpha(\theta,+\omega,\tau)-\alpha(\theta,-\omega,\tau)}{2}=-\frac{\mathcal F(\theta,\omega)}{\mathcal I(\theta)}.$$', '',
            '在每个工况的 73 个角度、4 个速度幅值上进行 292 组方向反转配对。相同姿态和力矩下，惯性曲率、重力与外力项抵消，速度奇对称分量检验实际耗散力。', '',
            '| 摩擦条件 | 真实奇对称加速度 RMS (rad/s²) | 模型奇对称加速度最大绝对值 | 奇对称响应相对误差 |',
            '|---|---:|---:|---:|']
    for name in ['viscous','coulomb','mixed_friction']:
        row=probes['conditions'][name]['friction']
        lines.append(f"| {NAMES[name]} | {row['true_odd_acceleration_rms_rad_s2']:.6f} | {row['learned_odd_acceleration_max_abs_rad_s2']:.6g} | {row['odd_acceleration_relative_error']*100:.2f}% |")
    lines+=['', '当前 KeyCLD 势能与动作矩阵只依赖位置，速度仅通过约束曲率的二次项进入加速度；因此它对速度反转给出相同加速度。其投影摩擦参数为零，缺少同时表达两个运动方向耗散的结构。这里没有将这解释为“已用摩擦数据训练后仍失败”：源训练工况摩擦为零，测得的是冻结模型的迁移响应及结构限制。', '',
            '库仑组综合力矩误差略低于标称组，不构成摩擦辨识成功的证据；其它误差项和运动速度可改变综合 RMS。方向反转检验仍显示库仑耗散响应全部缺失。', '',
            '## 三、真实 SI 质量辨识与惯量响应', '',
            r'$$M\mapsto kM,\quad V\mapsto kV,\quad B\mapsto kB\quad\Longrightarrow\quad\ddot x\ \text{不变}.$$', '',
            '原模型同时自由学习点质量、势能与动作映射，存在整体尺度自由度。数值检查将质量矩阵缩为 1/4 或扩大 4 倍，配套缩放势能及动作输出，加速度最大改变均为零。除此之外，网络中的三点等效质量也不是四个实际刚体的质量。不能直接把参数读数当作 kg 辨识结果。', '',
            r'$$k_\tau(\theta)=\frac{\hat\alpha(\theta,0,+1)-\hat\alpha(\theta,0,-1)}{2},\qquad\mathcal I_{\rm response}(\theta)=\frac{1}{k_\tau(\theta)}\ \ [\mathrm{kg\,m^2}].$$', '',
            '另用已知 N·m 输入的响应检查可观测等效惯量。该量混合了质量模型与动作映射的影响，不等于直接读取的神经质量参数。非正增益无法解释为本实验已知正向传动下的正惯量，显式计为失败。', '',
            '| 条件 | 正增益采样数 | 等效惯量相对误差中位数 ↓ | 相对误差均值 ↓ |',
            '|---|---:|---:|---:|']
    for name in ['nominal','heavy_payload','dense_bars','geometry_scaled','crank_slider']:
        row=probes['conditions'][name]['inertia'];error=row['equivalent_inertia_relative_error']
        lines.append(f"| {NAMES[name]} | {row['positive_gain_poses']}/{row['total_poses']} | {error['median']*100:.2f}% | {error['mean']*100:.2f}% |")
    lines+=['', '均值包含近零增益导致的大惯量误差，没有截断离群值；中位数仅描述正增益子集，须与失败数一起看。73 个角度网格包含周期端点的重复表示，计数为采样数而非独立机制数。当前没有一组在整个网格上满足正惯量响应，所以没有给出“可信的四刚体 SI 质量拟合成功”结论。', '',
            '## 四、数据是否足以辨识：独立物理回归对照', '',
            '使用已知均匀刚体形状、SI 力矩和精确真值状态，联合拟合 4 个质量、4 个粘性系数和 4 个平滑库仑力矩。12 个种子用于拟合，另 8 个用于检验；此对照没有使用神经模型输出，也不属于视频学习成绩。', '',
            '| 条件 | 回归矩阵秩 | 归一化条件数 | 最大质量相对误差 | 独立测试力矩 RMS (N·m) |',
            '|---|---:|---:|---:|---:|']
    for name,row in controls.items():
        mass_error=max(abs(x) for x in row['mass_relative_errors']) if row['mass_relative_errors'] is not None else None
        lines.append(f"| {NAMES[name]} | {row['regressor_rank']}/12 | {row['normalized_condition_number']:.2f} | {mass_error:.3g} | {row['test_torque_rms_Nm']:.3g} |")
    mixed=controls['mixed_friction']
    lines+=['', f"混合摩擦对照：最大粘性系数绝对误差 {max(abs(x) for x in mixed['viscous_absolute_errors']):.3g} N·m·s/rad，最大库仑力矩绝对误差 {max(abs(x) for x in mixed['coulomb_absolute_errors']):.3g} N·m。", '',
            '这些结果说明，在上述强物理先验与精确状态条件下，本批激励可辨识参数。它不证明从 64×64 视频也能达到同等精度。真值为模型明确指定的 SI 仿真参数，尚不是实物称重、惯量测量或摩擦测试得到的参数。', '',
            '## 五、具体动力学方程与误差项', '',
            r'四杆绝对角度 $q=(q_1,q_2,q_3)$、独立曲柄角 $\theta=q_1$：', '',
            r'$$\phi(q)=\begin{bmatrix}l_1\cos q_1+l_2\cos q_2-l_3\cos q_3-d\\l_1\sin q_1+l_2\sin q_2-l_3\sin q_3\end{bmatrix}=0.$$', '',
            r'曲柄滑块 $q=(q_1,q_2,s)$：', '',
            r'$$\phi(q)=\begin{bmatrix}l_1\cos q_1+l_2\cos q_2-s\\l_1\sin q_1+l_2\sin q_2\end{bmatrix}=0.$$', '',
            r'$$\gamma=\frac{dq}{d\theta},\quad J\gamma=0,\quad\gamma_1=1,\qquad\dot q=\gamma\omega,\quad\ddot q=\gamma\alpha+\gamma' + "'" + r'\omega^2.$$', '',
            r'$$\underbrace{\mathcal I(\theta)\alpha}_{惯性}+\underbrace{\mathcal H(\theta)\omega^2}_{速度二次项}+\underbrace{\mathcal G(\theta)}_{重力}+\underbrace{\mathcal F(\theta,\omega)}_{摩擦}=\underbrace{\tau_A}_{电机}+\underbrace{\mathcal Q_{ext}(\theta)}_{外力}.$$', '',
            r'$$\mathcal I=\gamma^TM\gamma,\qquad\mathcal H=\gamma^T(M\gamma' + "'" + r'+c(q,\gamma)),\qquad\mathcal G=\gamma^TG,\qquad\mathcal Q_{ext}=\gamma^TJ_F^TF.$$', '',
            r'四杆关节相对速度映射 $R=\begin{bmatrix}1&0&0\\-1&1&0\\0&-1&1\\0&0&1\end{bmatrix}$，令 $s_j=(R\gamma)_j$：', '',
            r'$$\mathcal F=\sum_{j=1}^{4}s_j\left[b_js_j\omega+\tau_{c,j}\tanh\left(\frac{s_j\omega}{\epsilon}\right)\right].$$', '',
            '四杆的粘性参数 b_j 单位为 N·m·s/rad；τ_c,j 为摩擦力矩 N·m，不是无量纲接触摩擦系数 μ。曲柄滑块第 4 项对应移动副，单位分别为 N·s/m 和 N；该实验滑块工况未施加摩擦。库仑项使用平滑 tanh，未模拟精确静摩擦粘滞。', '',
            r'预测的实际驱动轴力矩残差使用原始预测状态及其导数：$r_\tau=\gamma^T[M\hat{\ddot q}+c+G-Q_{ext}-Q_{fric}]-\tau_A$。它直接消除约束反力，没有通过拟合反力掩盖错误，也没有把预测位置投影回真值。', '',
            '固定例子：seed=1000，t=0.11 s。下表在各条件对应的真实曲柄角、速度处计算独立理想约束方程，再查询模型在同一状态的加速度；属于瞬时构成关系诊断，不是未来视频预测。', '',
            '| 条件 | I_eff (kg·m²) | 真实 α (rad/s²) | 模型 α (rad/s²) | 同状态力矩误差 (N·m) |',
            '|---|---:|---:|---:|---:|']
    for name,row in examples.items():
        lines.append(f"| {NAMES[name]} | {row['I_eff_kg_m2']:.6g} | {row['independent_alpha_rad_s2']:.4f} | {row['model_alpha_at_same_state_rad_s2']:.4f} | {row['matched_state_equivalent_motor_error_Nm']:.6f} |")
    e=examples['nominal'];w2=e['omega_rad_s']**2
    lines+=['', '标称条件的具体数值：', '',
            f"$$ {e['I_eff_kg_m2']:.8f}\\,\\alpha + ({e['h_velocity_coefficient']:.8f})\\times({e['omega_rad_s']:.6f})^2 + ({e['gravity_eff_Nm']:.8f}) + 0 = {e['motor_Nm']:.8f} + {e['external_eff_Nm']:.8f}. $$", '',
            f"独立方程得到 α={e['independent_alpha_rad_s2']:.6f} rad/s²；模型为 {e['model_alpha_at_same_state_rad_s2']:.6f} rad/s²，对应额外力矩误差 {e['matched_state_equivalent_motor_error_Nm']:.6f} N·m。完整各项系数与数值见 `equation_examples.json`。", '',
            '## 证据边界', '',
            '这轮结果针对一个训练种子和一个源机构。冻结状态组件没有质量/摩擦上下文输入，载荷泛化是该输入接口下的迁移测试；它不等同于提供完整交互历史的通用模型适应能力。新拓扑仅提供几何约束回调，未训练目标机构。',
            '数据可辨识性、原参数是否有 SI 含义、行为响应是否物理正确分别报告。尚未建立预先校准的正式物理通过阈值，不据此宣称整个模型族不具备学习能力或完成复杂并联机构覆盖。', '',
            f"执行设备：{manifest['devices']}；预测耗时 {manifest['elapsed_seconds']:.2f} s。权重哈希、版本和数据哈希均保存，原基线结果未覆盖。"]
    temporary=OUT/'results.md.partial';temporary.write_text('\n'.join(lines)+'\n');temporary.replace(OUT/'results.md')
    font=FontProperties(fname=FONT);fig,axes=plt.subplots(1,3,figsize=(15,4.4),layout='constrained')
    names=list(report['conditions']);values=[report['conditions'][name]['state_RK4']['motor_torque_error_rms_Nm']['mean'] for name in names]
    axes[0].bar(range(8),values,color=['#328e9c']+['#ed8b32']*3+['#7181a7']*4)
    axes[0].set_xticks(range(8),[NAMES[n] for n in names],fontproperties=font,rotation=45,ha='right')
    axes[0].set_ylabel('N·m');axes[0].set_title('独立曲柄轴力矩残差',fontproperties=font)
    for x,y in enumerate(values):axes[0].text(x,y+.08,f'{y:.2f}',ha='center',fontsize=8)
    fric_names=['viscous','coulomb','mixed_friction'];truth=[probes['conditions'][n]['friction']['true_odd_acceleration_rms_rad_s2'] for n in fric_names]
    axes[1].bar(np.arange(3)-.17,truth,width=.34,label='Independent physics',color='#328e9c')
    axes[1].bar(np.arange(3)+.17,[0.,0.,0.],width=.34,label='Frozen KeyCLD',color='#ed8b32')
    axes[1].set_xticks(range(3),[NAMES[n] for n in fric_names],fontproperties=font)
    axes[1].set_ylabel('rad/s²');axes[1].set_title('速度反转的摩擦响应',fontproperties=font);axes[1].legend(fontsize=8,loc='upper center')
    axes[1].set_ylim(0,max(truth)*1.20)
    for i,y in enumerate(truth):axes[1].text(i-.17,y+3,f'{y:.2f}',ha='center',fontsize=8);axes[1].text(i+.17,3,'0',ha='center',fontsize=8)
    selected=['nominal','heavy_payload','dense_bars','geometry_scaled','crank_slider']
    median=[probes['conditions'][n]['inertia']['equivalent_inertia_relative_error']['median']*100 for n in selected]
    axes[2].bar(range(5),median,color='#7181a7')
    axes[2].set_xticks(range(5),[NAMES[n] for n in selected],fontproperties=font,rotation=40,ha='right')
    axes[2].set_ylabel('%');axes[2].set_title('SI惯量响应误差（正增益子集）',fontproperties=font)
    axes[2].set_ylim(0,max(median)*1.30)
    for i,n in enumerate(selected):
        k=probes['conditions'][n]['inertia']['positive_gain_poses'];axes[2].text(i,median[i]+2,f'{median[i]:.1f}%\n{k}/73',ha='center',fontsize=8)
    for axis in axes:
        axis.spines[['top','right']].set_visible(False);axis.grid(axis='y',alpha=.18)
    fig.suptitle('冻结 RK4 状态组件 · 独立 SI 方程检验 · 不计作视频模型成绩',fontproperties=font,fontsize=12)
    fig.savefig(OUT/'dynamics_evaluation.png',dpi=160);plt.close(fig)
    with np.load(OUT/'example_inputs.npz') as saved:images={k:saved[k] for k in saved.files}
    canvas=Image.new('RGB',(768,432),'#152330');draw=ImageDraw.Draw(canvas);face=ImageFont.truetype(FONT,22,index=2)
    for i,(key,label) in enumerate([('nominal_rgb','源机构：4R 四杆'),('slider_rgb','目标机构：3R+1P 曲柄滑块')]):
        canvas.paste(Image.fromarray(images[key][0]).resize((384,384),Image.Resampling.NEAREST),(i*384,40))
        draw.text((i*384+8,8),label,font=face,fill='white')
    canvas.save(OUT/'mechanism_comparison.png')
    print('Wrote dynamics-first results.md and two scientific previews.',flush=True)


if __name__=='__main__':main()
