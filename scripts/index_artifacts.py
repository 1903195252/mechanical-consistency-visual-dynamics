"""Build navigable inventories for a collection without moving its runtime files."""
import argparse
import csv
import json
from pathlib import Path
import re
import stat

from artifacts import collection_packages, safe_path
from check_artifact_coverage import check

ROOT = Path(__file__).resolve().parents[1]
GIB = 2 ** 30
GROUPS = {
    'planar_data': '平面输入、图像与物理参考',
    'spatial_data': '空间输入、视频与物理参考',
    'planar64': '64 px KeyCLD / CON 权重与预测',
    'planar256': '256 px KeyCLD / CON 权重与预测',
    'readers': '独立测量头、标定与对照',
    'vjepa': 'V-JEPA + AC 适配权重、特征与测量',
    'lingbot': 'LingBot 适配权重、0/50 视频与测量',
    'torque_controls': '力矩激励与结构诊断',
    'metadata': '数据清单、协议与模型配置',
    'lingbot_official': 'LingBot Transformer / VAE / 文本编码器 / 分词器',
    'vjepa_official': 'V-JEPA 2.1 B/16 384 px 编码器',
    'vjepa_optimizer_states': '三种子 × 两家族完整 AC 续训状态',
}
ROLE_LABELS = {
    'dataset': '输入数据与物理真值',
    'final_checkpoint': '最终预测模型权重',
    'feedback_head': '模型内部反馈头',
    'measurement_head': '独立测量头',
    'official_weights': '官方预训练权重',
    'official_components': '官方配置与分词器等配套文件',
    'optimizer_state': '完整续训状态',
    'forecast': '保存的预测',
    'reference': '匹配参考视频与测量',
    'measurement': '坐标读出、重构与校准结果',
    'training_record': '训练配置、曲线与最终检查点信息',
    'protocol': '协议与数据清单',
    'score_or_diagnostic': '评分、逐例记录与诊断',
    'conditioning': '固定文本条件嵌入',
}


def role(record, package):
    path = record['path']; name = Path(path).name; group = record['group']
    if group == 'vjepa_optimizer_states': return 'optimizer_state'
    if group.endswith('_official'):
        return 'official_weights' if name.endswith(('.pt', '.safetensors')) else 'official_components'
    if name == 'text_embeddings.pt': return 'conditioning'
    if name.endswith('_readout.pt'): return 'feedback_head'
    if name.endswith(('_head.pt', 'relative_head.npz')) or ('/readers/' in path and name == 'final.pt'):
        return 'measurement_head'
    if name in ('params.p', 'trained_params.p') or name.endswith(('_predictor.pt', '.safetensors')):
        return 'final_checkpoint'
    if record['group'] in ('planar_data', 'spatial_data'):
        return 'dataset'
    if record['group'] == 'torque_controls' and any('/'+part+'/' in path for part in ('observations', 'truth')):
        return 'dataset'
    if '/reference/' in path: return 'reference'
    if record['group'] == 'readers' or '/readout/' in path or '/calibrated/' in path:
        return 'measurement'
    if '/predictions/' in path or '/forecast_features/' in path: return 'forecast'
    if 'training' in name or name.startswith('checkpoint_'): return 'training_record'
    if record['group'] == 'metadata' or 'manifest' in name or 'protocol' in name:
        return 'protocol'
    return 'score_or_diagnostic'


def checkpoint(record, package):
    kind = role(record, package)
    if kind not in ('final_checkpoint', 'feedback_head', 'measurement_head', 'official_weights', 'optimizer_state'):
        return None
    path = record['path']; group = record['group']; seed = re.search(r'seed_?(\d+)', path)
    if '/con_corpus/' in path or '/con_256_' in path: model = 'CON'
    elif '/keycld' in path: model = 'KeyCLD'
    elif 'vjepa' in path: model = 'V-JEPA + AC' if not group.endswith('_official') else 'V-JEPA 2.1'
    elif 'lingbot' in path: model = 'LingBot-VA'
    else: model = 'Measurement'
    family = re.search(r'H-[AB]', Path(path).name)
    resolution = 256 if group == 'planar256' else 64 if group in ('planar64', 'torque_controls') else ''
    if kind == 'measurement_head':
        model = 'Measurement'
        if '/reader_256/' in path: resolution = 256
    component = Path(path).stem
    if group.endswith('_official') and 'lingbot' in path: component = Path(path).parent.name
    elif kind == 'final_checkpoint' and Path(path).name == 'params.p': component = Path(path).parent.name
    elif '/checkpoint_step_50/' in path: component = 'transformer_step50'
    return dict(package=package, model=model, role=kind, component=component,
                seed=int(seed.group(1)) if seed else 42 if group == 'torque_controls' else '',
                family=family.group() if family else '', resolution_px=resolution,
                **{k: record[k] for k in ('path', 'bytes', 'group')})


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def write_tsv(path, fields, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, delimiter='\t', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


GB = 10 ** 9


def build_package(directory, package_id):
    directory = directory.resolve()
    metadata = json.loads((directory/'package.json').read_text())
    files = metadata['files']; seen = set(); parents = set(); weights = []; indexed = []
    for record in files:
        path = record['path']; safe_path(path); source = directory/path
        if path in seen: raise ValueError('Repeated runtime path: '+path)
        seen.add(path); info = source.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size != record['bytes']:
            raise ValueError('Missing or incomplete asset: '+path)
        if source.parent not in parents:
            if directory not in source.parent.resolve().parents: raise ValueError('External asset')
            parents.add(source.parent)
        item = checkpoint(record, package_id)
        if item: weights.append(item)
        indexed.append(dict(role=role(record, package_id), **record))
    actual = {str(p.relative_to(directory)) for p in (directory/'output').rglob('*') if p.is_file()}
    if actual != seen: raise ValueError('Package files differ from its description')
    size = sum(x['bytes'] for x in files)
    if size >= 50_000_000_000: raise ValueError('Package exceeds 50 GB')
    write_tsv(directory/'files.tsv', ['role', 'path', 'bytes'], indexed)
    write_tsv(directory/'checkpoints.tsv', ['model','role','component','seed','family','resolution_px','path','bytes'], weights)
    lines = ['# '+metadata['title'], '', metadata['description'], '',
             '**%.2f GB · %s 个实验文件**'%(size/GB,format(len(files),',')), '',
             '[文件目录](files.tsv) · [模型组件](checkpoints.tsv) · [模型总目录](../README.md)', '',
             '## 内容', '', '| 目录 | 内容 |', '|---|---|']
    roots = {}
    for record in files:
        path = Path(record['path']); folder = Path(*path.parts[:min(3,len(path.parts)-1)])
        roots.setdefault(str(folder),set()).add(GROUPS[record['group']])
    for folder, descriptions in sorted(roots.items()):
        lines.append('| [%s](%s/) | %s |'%(folder,folder,'；'.join(sorted(descriptions))))
    if weights:
        lines += ['', '## 模型组件', '', '| 模型 | 角色 | 组件 | 种子 | 家族 |', '|---|---|---|---|---|']
        for w in weights:
            lines.append('| %s | %s | [%s](%s) | %s | %s |'%(w['model'],ROLE_LABELS[w['role']],w['component'],w['path'],w['seed'],w['family']))
    lines += ['', '## 使用', '', '在源码仓库中导入这个分包：', '', '```bash',
              'python3 scripts/artifacts.py import /path/to/artifacts/'+package_id,'```','',
              '导入后恢复统一的 `output/` 运行路径。完整模型运行还需对应共享数据、预训练组件和依赖环境，见总目录。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    return dict(path=package_id,title=metadata['title'],description=metadata['description'],
                model=metadata['model'],experiment=metadata['experiment'],
                files=len(files),bytes=size,default=metadata.get('default',True)),weights


def build(directory):
    directory = directory.resolve()
    packages=collection_packages(directory,include_optional=True)
    inventory=[];weights=[]
    for package in packages:
        item, entries=build_package(package,str(package.relative_to(directory)))
        inventory.append(item);weights+=entries
    coverage=check(directory)
    if not coverage['complete']:raise ValueError(json.dumps(coverage))
    catalog=dict(schema=3,packages=inventory,checkpoints=weights)
    write_json(directory/'catalog.json',catalog)
    write_json(ROOT/'configs/artifact_catalog.json',catalog)
    write_tsv(directory/'checkpoints.tsv',['package','model','role','component','seed','family','resolution_px','path','bytes'],weights)
    titles={'data':'共享数据','keycld':'KeyCLD','con-model':'CON','vjepa':'V-JEPA + AC','lingbot':'LingBot-VA','measurement':'独立测量组件'}
    for group in titles:
        items=[x for x in inventory if x['path'].split('/')[0]==group]
        lines=['# '+titles[group],'','[返回总目录](../README.md)','','| 分包 | 内容 | GB | 导入 |','|---|---|---:|---|']
        for x in items:
            lines.append('| [%s](%s/README.md) | %s | %.2f | %s |'%(x['title'],x['path'].split('/')[1],x['description'],x['bytes']/GB,'默认' if x['default'] else '续训时按需'))
        (directory/group/'README.md').write_text('\n'.join(lines)+'\n')
    total=sum(x['bytes'] for x in inventory);largest=max(x['bytes'] for x in inventory)
    lines=['# 实验数据与模型','','按 **模型 → 实验类型** 划分；共享数据和独立测量组件单列。每个叶子目录是一个可单独传输、导入的分包。', '',
           '**%d 个分包，最大 %.2f GB；每包均小于 50 GB。** GB 按十进制计量。'%(len(inventory),largest/GB),'',
           '```text','artifacts/','├── data/         平面 64 px、平面 256 px、空间共享数据','├── keycld/       64 px、256 px、力矩激励与结构诊断','├── con-model/    64 px、256 px','├── vjepa/        预训练编码器、空间主实验、各种子续训状态','├── lingbot/      预训练组件、各种子适配权重、0/50 更新预测、参考视频','└── measurement/  平面与空间独立测量组件','```','',
           '## 模型入口','','| 模型 / 数据 | 分包数 | GB |','|---|---:|---:|']
    for group,title in titles.items():
        selected=[x for x in inventory if x['path'].split('/')[0]==group]
        lines.append('| [%s](%s/README.md) | %d | %.2f |'%(title,group,len(selected),sum(x['bytes'] for x in selected)/GB))
    lines+=['','[全部模型组件](checkpoints.tsv) · [分包目录](catalog.json)','','## 分包明细','','| 分包 | 实验类型 | GB |','|---|---|---:|']
    for x in inventory:lines.append('| [%s](%s/README.md) | %s | %.2f |'%(x['path'],x['path'],x['title'],x['bytes']/GB))
    lines+=['','## 一键准备','','在源码仓库目录运行：','','```bash',
            './setup.sh --profile all --device cuda --download-models --artifact-dir /path/to/artifacts','```','',
            '默认组合导入实验数据、最终模型、预测与测量结果以及官方预训练组件。V-JEPA 的续训状态按种子单独导入；普通推理和评分使用导出的预测器与内部反馈头。安装依赖仍需要网络。','','例如继续 seed 42 的 V-JEPA 训练：','','```bash',
            'python3 scripts/artifacts.py import /path/to/artifacts/vjepa/resume-seed42','```','',
            '## 单包传输','','分包可以直接复制。需要单文件上传时，将一个分包导出为 TAR：','','```bash',
            'python3 scripts/artifacts.py export /path/to/artifacts/con-model/planar-256 con-planar-256.tar',
            'python3 scripts/artifacts.py import con-planar-256.tar','```','',
            '导出工具对单个 TAR 同样执行 50 GB 上限。传输时选择需要的分包，无需合并成一个大归档。','','## 实验范围','',
            '- 平面：343 个机构，64／256 px，三个种子，包含视频、状态与积分器对照。',
            '- 空间：Stewart 与 Delta 各 50 个测试机构、三个工况、三个种子；V-JEPA + AC 与 LingBot 使用相同测试队列。',
            '- LingBot：0／50 更新，保留 1,800 段生成视频和 300 条匹配参考视频。',
            '- 测量：冻结读取头、标定配置、原生坐标与 CAD/IK 重构；失败案例随相应实验结果保留。','',
            '每包的 `package.json` 描述运行文件布局；`files.tsv` 和 `checkpoints.tsv` 提供用途与模型组件导航。模型文件和实验内部运行字段保持兼容。']
    (directory/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(packages=len(inventory),largest_GB=round(largest/GB,2),total_GB=round(total/GB,2),files=sum(x['files'] for x in inventory),core_coverage=coverage),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    build(parser.parse_args().directory)
