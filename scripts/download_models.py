"""Download pinned official model files with resumable, verified transfers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def valid(path, item):
    return (path.is_file() and path.stat().st_size == item['bytes']
            and digest(path) == item['sha256'])


def download(item, root=ROOT):
    root = Path(root).resolve()
    path = (root/item['path']).resolve()
    if root not in path.parents or Path(item['path']).is_absolute():
        raise ValueError('Invalid model destination')
    path.parent.mkdir(parents=True, exist_ok=True)
    if valid(path, item):
        return item['path']
    if path.exists():
        raise FileExistsError('Existing file differs; move it aside before downloading: '+item['path'])
    partial = path.with_name(path.name+'.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > item['bytes']:
        raise ValueError('Oversized partial file: '+str(partial))
    if shutil.disk_usage(path.parent).free < item['bytes']-offset+16*1024**2:
        raise OSError('Insufficient free space for '+item['path'])
    if offset < item['bytes']:
        headers = {'User-Agent':'mechanical-consistency-model-downloader'}
        if offset:
            headers['Range']='bytes=%d-'%offset
        request = urllib.request.Request(item['url'],headers=headers)
        with urllib.request.urlopen(request,timeout=90) as response:
            if response.status == 206:
                if not response.headers.get('Content-Range','').startswith('bytes %d-'%offset):
                    raise ValueError('Incorrect HTTP resume range')
            elif response.status == 200:
                offset=0
            else:
                raise ValueError('Unexpected HTTP status')
            with partial.open('ab' if offset else 'wb') as stream:
                while True:
                    block=response.read(4*1024**2)
                    if not block:
                        break
                    stream.write(block)
                    if stream.tell()>item['bytes']:
                        raise ValueError('Download exceeds registered size')
    if not valid(partial,item):
        raise ValueError('Checksum or size mismatch: '+str(partial))
    partial.replace(path)
    return item['path']


def model_files(model):
    if model=='vjepa':
        return json.loads((ROOT/'configs/pretrained_models.json').read_text())['vjepa']['files']
    spec=json.loads((ROOT/'configs/pretrained_models.json').read_text())['lingbot']
    manifest=json.loads((ROOT/'configs/lingbot_base_files.json').read_text())
    result=[]
    for f in manifest['Data']['Files']:
        if f['Type']!='blob' or f['Path'].startswith('assets/'):
            continue
        result.append(dict(path=spec['directory']+'/'+f['Path'],bytes=f['Size'],sha256=f['Sha256'],
            url=spec['endpoint']+'?'+urllib.parse.urlencode(dict(Revision=spec['revision'],FilePath=f['Path']))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=['vjepa','lingbot','all'])
    p.add_argument('--list',action='store_true')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    items=[f for m in (['vjepa','lingbot'] if a.model=='all' else [a.model]) for f in model_files(m)]
    print('%d files, %.2f GiB'%(len(items),sum(f['bytes'] for f in items)/2**30))
    if a.list:
        for f in items:
            print('%12d  %s'%(f['bytes'],f['path']))
        return
    if a.verify_only:
        missing=[f['path'] for f in items if not valid(ROOT/f['path'],f)]
        if missing:
            raise SystemExit('Missing or invalid files:\n'+'\n'.join(missing))
    else:
        if not 1<=a.workers<=16:
            p.error('--workers must be between 1 and 16')
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for name in pool.map(download,sorted(items,key=lambda f:f['bytes'])):
                print('Verified '+name,flush=True)
    print('All requested model files verified.')


if __name__=='__main__':
    main()
