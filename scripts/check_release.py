#!/usr/bin/env python3
"""Check tracked source files before packaging or publishing a repository."""
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    output = subprocess.check_output(['git','ls-files','-z'],cwd=str(ROOT))
    names = [n.decode() for n in output.split(b'\0') if n]
    if not names:
        raise RuntimeError('Stage the intended release files before checking')
    forbidden_parts={'output','official','__pycache__','.venv','build','.idea','.vscode'}
    forbidden_suffixes={'.pyc','.log','.pid','.partial','.tmp','.pt','.pth','.p','.pkl','.ckpt','.h5','.sqlite','.db','.npz','.npy','.safetensors','.zip','.mp4','.key','.pem'}
    patterns={
        'private-key':r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
        'github-token':r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b',
        'api-key':r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b',
        'personal-path':r'/(?:home|media)/[A-Za-z0-9_-]+/',
        'cloud-path':r'/root/' + r'autodl[^\s"\']*',
        'cloud-host':r'connect\.[a-z0-9.]+\.(?:com|cn)|[a-z0-9.]+\.gpugeek\.com',
    }
    problems=[]
    for name in names:
        path=ROOT/name
        if path.is_symlink() or set(Path(name).parts)&forbidden_parts or path.suffix in forbidden_suffixes:
            problems.append((name,'excluded artifact'))
        if path.stat().st_size > 10*1024*1024:
            problems.append((name,'file larger than 10 MiB'))
        if path.suffix in {'.png','.pdf'}:
            continue
        text=path.read_text(encoding='utf-8')
        if path.suffix=='.py':
            try:ast.parse(text,filename=name)
            except SyntaxError:problems.append((name,'Python syntax'))
        if path.suffix=='.json':
            try:json.loads(text)
            except ValueError:problems.append((name,'JSON syntax'))
        for label,pattern in patterns.items():
            if re.search(pattern,text):problems.append((name,label))
    if problems:
        # Report locations only; secret values never appear in checker output.
        for name,label in problems:print(name+': '+label)
        raise SystemExit(1)
    print('Release checks passed for %d tracked files.' % len(names))


if __name__=='__main__':
    main()
