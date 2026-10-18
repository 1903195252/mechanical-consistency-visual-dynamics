"""Resume official pinned model files, publishing only SHA256-verified files."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.parse
import urllib.request

from .common import ROOT,OUT,OLD_RUN,sha,write

DEST=ROOT/'output/model_expansion/weights/lingbot-va-base'
REV='5fbd004a61bac17d3764504f09e0b628fc2c0592'


def run():
    import fcntl
    DEST.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    metadata=json.loads((OLD_RUN/'official_weight_manifest.json').read_text())
    files=[f for f in metadata['Data']['Files'] if f['Type']=='blob' and not f['Path'].startswith('assets/')]
    progress=dict(revision=REV,total_files=len(files),total_bytes=sum(f['Size'] for f in files),verified=[],errors=[],pid=os.getpid())
    mutex=threading.Lock()
    def one(f):
        target=DEST/f['Path']
        if DEST.resolve() not in target.resolve().parents:raise ValueError('Invalid model path')
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and target.stat().st_size==f['Size'] and sha(target)==f['Sha256']:
            return f['Path']
        partial=target.with_name(target.name+'.partial')
        for attempt in range(3):
            try:
                if shutil.disk_usage(OUT).free<40*1024**3:raise OSError('Keep at least 40 GiB free on the workspace disk')
                offset=partial.stat().st_size if partial.exists() else 0
                if offset>f['Size']:raise ValueError('Oversized partial file')
                if offset<f['Size']:
                    url='https://modelscope.cn/api/v1/models/Robbyant/lingbot-va-base/repo?'+urllib.parse.urlencode(dict(Revision=REV,FilePath=f['Path']))
                    request=urllib.request.Request(url,headers={'Range':'bytes=%d-'%offset} if offset else {})
                    with urllib.request.urlopen(request,timeout=90) as response:
                        if offset and response.status==206:
                            if not response.headers.get('Content-Range','').startswith('bytes %d-'%offset):raise ValueError('Incorrect resume range')
                        elif response.status==200:offset=0
                        else:raise ValueError('Unexpected HTTP status')
                        with partial.open('ab' if offset else 'wb') as stream:
                            while True:
                                block=response.read(4*1024*1024)
                                if not block:break
                                stream.write(block)
                                if stream.tell()>f['Size']:raise ValueError('Response larger than pinned file')
                if partial.stat().st_size!=f['Size'] or sha(partial)!=f['Sha256']:
                    # Do not reuse corrupted complete bytes on the next attempt.
                    if partial.stat().st_size>=f['Size']:partial.unlink()
                    raise ValueError('Official weight size or digest mismatch')
                partial.replace(target)
                return f['Path']
            except Exception:
                if attempt==2:raise
                time.sleep(2**attempt)
    def task(f):
        try:
            name=one(f)
            with mutex:progress['verified'].append(name)
        except Exception as error:
            with mutex:progress['errors'].append(dict(file=f['Path'],error=str(error)))
        with mutex:
            progress['updated_unix']=time.time();write(OUT/'download_status.json',progress)
            print(json.dumps(dict(verified=len(progress['verified']),total=len(files),errors=len(progress['errors']))),flush=True)
    write(OUT/'download_status.json',progress)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(task,sorted(files,key=lambda f:f['Size'])))
    progress['complete']=len(progress['verified'])==len(files) and not progress['errors']
    write(OUT/'download_status.json',progress)


if __name__=='__main__':run()
