"""Verify or import experiment bundles containing a file manifest and output/ tree."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import struct

ROOT=Path(__file__).resolve().parents[1]


def safe_path(name):
    p=PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!='output' or '\\' in name:
        raise ValueError('Bundle path must be contained in output/: '+name)
    return p


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(8*1024**2),b''):
            h.update(b)
    return h.hexdigest()


def import_tree(directory, root=ROOT, verify_only=False):
    """Import a manifested directory with the same checks as an archive.

    Plain directories avoid a second archive-sized disk allocation when sharing
    large checkpoints. Each copied file is checked before atomic publication.
    """
    directory=Path(directory).resolve();root=Path(root).resolve()
    files=json.loads((directory/'artifact_manifest.json').read_text())['files']
    if len({f['path'] for f in files})!=len(files):
        raise ValueError('Duplicate manifest entries')
    pending=[]
    for f in files:
        name=f['path'];safe_path(name)
        source=(directory/name).resolve();target=(root/name).resolve()
        if directory not in source.parents or root not in target.parents:
            raise ValueError('Artifact path escapes its root: '+name)
        if not source.is_file() or source.stat().st_size!=f['bytes']:
            raise ValueError('Missing or incomplete artifact: '+name)
        if target.exists() and not verify_only:
            if not target.is_file() or target.stat().st_size!=f['bytes'] or sha(target)!=f['sha256']:
                raise FileExistsError('Existing artifact differs: '+name)
        else:pending.append((source,target,f))
    if not verify_only and shutil.disk_usage(root).free<sum(f['bytes'] for _,_,f in pending)+16*1024**2:
        raise OSError('Insufficient space for imported artifacts')
    for source,target,f in pending:
        if verify_only:
            if sha(source)!=f['sha256']:raise ValueError('Artifact checksum mismatch: '+f['path'])
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        fd,temp=tempfile.mkstemp(prefix=target.name+'.',suffix='.partial',dir=target.parent)
        h=hashlib.sha256()
        try:
            with os.fdopen(fd,'wb') as out,source.open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024**2),b''):
                    h.update(block);out.write(block)
            if h.hexdigest()!=f['sha256']:raise ValueError('Artifact checksum mismatch: '+f['path'])
            Path(temp).replace(target)
        finally:
            if Path(temp).exists():Path(temp).unlink()
    return len(files)


def import_bundle(bundle, root=ROOT, verify_only=False):
    root=Path(root).resolve()
    with tarfile.open(bundle,'r:*') as archive:
        members=archive.getmembers()
        names=[m.name for m in members]
        if len(names)!=len(set(names)):
            raise ValueError('Duplicate archive entries')
        metadata=archive.getmember('artifact_manifest.json')
        if not metadata.isfile() or metadata.size>32*1024**2:
            raise ValueError('Invalid artifact manifest')
        manifest=json.load(archive.extractfile(metadata))
        files=manifest['files']
        expected={f['path']:f for f in files}
        if len(expected)!=len(files):
            raise ValueError('Duplicate manifest entries')
        by_name={m.name:m for m in members if m.name!='artifact_manifest.json'}
        if set(by_name)!=set(expected):
            raise ValueError('Archive contents differ from manifest')
        pending=[]
        for name,f in expected.items():
            safe_path(name)
            member=by_name[name]
            if not member.isfile() or member.size!=f['bytes']:
                raise ValueError('Invalid archive file: '+name)
            target=(root/name).resolve()
            if root not in target.parents:
                raise ValueError('Destination escapes repository: '+name)
            if target.exists():
                if target.stat().st_size!=f['bytes'] or sha(target)!=f['sha256']:
                    raise FileExistsError('Existing artifact differs: '+name)
            else:
                pending.append((name,f))
        if not verify_only and shutil.disk_usage(root).free<sum(f['bytes'] for _,f in pending)+16*1024**2:
            raise OSError('Insufficient space for imported artifacts')
        for name,f in pending if not verify_only else expected.items():
            h=hashlib.sha256()
            target=root/name
            if not verify_only:
                target.parent.mkdir(parents=True,exist_ok=True)
                fd,temp=tempfile.mkstemp(prefix=target.name+'.',suffix='.partial',dir=target.parent)
                stream=os.fdopen(fd,'wb')
            else:
                stream=None;temp=None
            try:
                with archive.extractfile(by_name[name]) as source:
                    while True:
                        b=source.read(8*1024**2)
                        if not b:break
                        h.update(b)
                        if stream:stream.write(b)
                if stream:stream.close()
                if h.hexdigest()!=f['sha256']:
                    raise ValueError('Artifact checksum mismatch: '+name)
                if temp:Path(temp).replace(target)
            finally:
                if stream and not stream.closed:stream.close()
                if temp and Path(temp).exists():Path(temp).unlink()
        return len(files)


def pack_bundle(selection, destination, source_root=ROOT):
    """Package explicit source-to-runtime mappings; never include source locations."""
    root=Path(source_root).resolve()
    entries=json.loads(Path(selection).read_text())
    files=[];sources=[];seen=set()
    for entry in entries:
        name=entry['path'];safe_path(name)
        if name in seen:raise ValueError('Duplicate bundle destination: '+name)
        seen.add(name)
        source_name=entry.get('source',name);safe_path(source_name)
        source=(root/source_name).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError('Missing or external source: '+source_name)
        size=source.stat().st_size
        if source.suffix=='.safetensors':
            with source.open('rb') as stream:
                count=struct.unpack('<Q',stream.read(8))[0]
                if count>100*1024**2:raise ValueError('Invalid tensor header')
                header=json.loads(stream.read(count))
            expected=8+count+max(t['data_offsets'][1] for k,t in header.items() if k!='__metadata__')
            if expected!=size:raise ValueError('Incomplete tensor file: '+source_name)
        digest=sha(source)
        if entry.get('sha256',digest)!=digest:raise ValueError('Source checksum mismatch: '+source_name)
        files.append(dict(path=name,bytes=size,sha256=digest));sources.append(source)
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    if shutil.disk_usage(destination.parent).free<sum(f['bytes'] for f in files)+16*1024**2:
        raise OSError('Insufficient space for artifact bundle')
    fd,temp=tempfile.mkstemp(prefix=destination.name+'.',suffix='.partial',dir=destination.parent)
    os.close(fd)
    try:
        with tarfile.open(temp,'w') as archive:
            payload=json.dumps(dict(schema=1,files=files),indent=2).encode()
            item=tarfile.TarInfo('artifact_manifest.json');item.size=len(payload)
            archive.addfile(item,io.BytesIO(payload))
            for source,record in zip(sources,files):
                item=tarfile.TarInfo(record['path']);item.size=record['bytes'];item.mode=0o644
                with source.open('rb') as stream:archive.addfile(item,stream)
        # Detect concurrent modification between hashing and archive creation.
        with tempfile.TemporaryDirectory(dir=destination.parent) as empty:
            import_bundle(temp,empty,verify_only=True)
        Path(temp).replace(destination)
    finally:
        if Path(temp).exists():Path(temp).unlink()
    return len(files)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('import','verify'):
        sub.add_parser(name).add_argument('bundle',type=Path,nargs='+')
    pack=sub.add_parser('pack')
    pack.add_argument('selection',type=Path)
    pack.add_argument('destination',type=Path)
    pack.add_argument('--source-root',type=Path,default=ROOT)
    a=p.parse_args()
    if a.command=='pack':
        print('Packaged %d verified files'%pack_bundle(a.selection,a.destination,a.source_root))
        return
    for bundle in a.bundle:
        importer=import_tree if bundle.is_dir() else import_bundle
        print('%s: %d verified files'%(bundle,importer(bundle,verify_only=a.command=='verify')))


if __name__=='__main__':main()
