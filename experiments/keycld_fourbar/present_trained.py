"""Create readable diagnostic media from real held-out prediction archives."""
import json
import os
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/keycld_fourbar/trained'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'output/fourbar/.cache/matplotlib'))


def main():
    report=json.loads((OUT/'evaluation_report.json').read_text())
    cjk=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    font_path=str(cjk) if cjk.exists() else subprocess.check_output(['fc-match',':lang=zh','--format=%{file}'],text=True).strip()
    font=ImageFont.truetype(font_path,18,index=2 if cjk.exists() else 0)
    writer=cv2.VideoWriter(str(OUT/'trained_prediction_comparison.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(640,380))
    if not writer.isOpened():raise RuntimeError('Cannot create trained comparison video')
    for i in range(4):
        with np.load(OUT/f'test_example_{i}.npz') as data:
            truth=data['reference_rgb'];pred=data['prediction_rgb'];index=int(data['index']);times=data['time']
        for frame in range(len(times)):
            canvas=Image.new('RGB',(640,380),(25,30,36));draw=ImageDraw.Draw(canvas)
            draw.text((12,5),'真实运动（独立测试）',font=font,fill='white')
            draw.text((332,5),'KeyCLD 训练后预测',font=font,fill='white')
            canvas.paste(Image.fromarray(truth[frame]).resize((320,320)),(0,32))
            canvas.paste(Image.fromarray(pred[frame]).resize((320,320)),(320,32))
            draw.rectangle((0,352,640,380),fill=(25,30,36))
            draw.text((12,355),f'测试 {index} | t={times[frame]:.2f}s | 5倍慢放 | 前3帧观测',font=font,fill='white')
            image=cv2.cvtColor(np.asarray(canvas),cv2.COLOR_RGB2BGR);writer.write(image)
            if i==0 and frame==10:cv2.imwrite(str(OUT/'trained_comparison_preview.png'),image)
    writer.release()
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    history=json.loads((OUT/'training_history.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for name in ['reconstruction','dynamics','bce_loss']:
        axes[0].semilogy([row['epoch'] for row in history],[row[name] for row in history],label=name)
    axes[0].set_xlabel('Epoch');axes[0].set_ylabel('Official loss components');axes[0].legend()
    axes[1].plot([row['epoch'] for row in history],[row['validation_vpt_mean'] for row in history])
    axes[1].set_xlabel('Epoch');axes[1].set_ylabel('Validation VPT index (0..49)')
    fig.savefig(OUT/'training_curves.png',dpi=160);plt.close(fig)
    lines=['# 四杆 KeyCLD 首轮训练与测试','',
           '本次完成预定 40 轮、18,000 次更新。保留论文主指标 VPT，并增加独立测试；仅替换机构适配，不加入关节真值监督。',
           '', '| 条件 | 序列数 | VPT 均值 ± 标准差（索引） | 未来图像 MSE | 输出有限 |',
           '|---|---:|---:|---:|---:|']
    for name,value in report['conditions'].items():
        mse=value['future_image_mse_mean'];mse_text=f'{mse:.6g}' if mse is not None else '未计算'
        lines.append(f"| {name} | {value['run_count']} | {value['vpt_mean_index']:.2f} ± {value['vpt_std_index']:.2f} | {mse_text} | {value['finite_runs']}/{value['run_count']} |")
    checks=json.loads((OUT/'additional_checks.json').read_text())
    for name,value in checks['persistence_control'].items():
        lines.append(f"| persistence_{name} | {value['run_count']} | {value['persistence_vpt_mean_index']:.2f} ± {value['persistence_vpt_std_index']:.2f} | — | 图像保持对照 |")
    lines+=['','随机与已训练模型在同一批独立测试轨迹上计算。加重与摩擦组为额外独立随机样本，未用于训练或选权重，属于初步分布外检查，不是配对因果消融。',
            '', '当前同分布训练模型 VPT=7.82，静止图像保持对照 VPT=7.98，均值接近；该实验尚未证明有效动力学预测优于这一简单对照。不能将摩擦组较高的 VPT 单独解释成更强泛化：该组曲轴平均角行程约 0.90 rad，同分布组约 19.29 rad，运动难度不同。',
            '', 'VPT 沿用作者代码的 0..49 索引语义，采样间隔 0.01 s。全 50 帧组成约 0.5 秒序列，前三帧用于初始化。对比视频为 5 倍慢放，便于看清运动。',
            '', '论文关键点是学习到的潜在坐标，额外几何与力残差先列为诊断；未经物理语义校验不能作为铰中心准确率。当前只有一个训练随机种子，不能据此声称跨机构泛化或发表多模型排名。',
            '', '协议、数值收敛检查和实现差异见 `experiments/keycld_fourbar/training_protocol.json` 与 README。']
    (OUT/'results.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
