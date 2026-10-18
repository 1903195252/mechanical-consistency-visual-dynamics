"""Finish and validate shared CAD maps before concurrent RGB measurements."""
import json
import time
import zipfile
import numpy as np
from .common import ROOT,OUT,records,sha,write


def prepare():
    import fcntl
    from .delta_optical import OPTICAL,raymap
    lock=(OUT/'raymap_cache.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX)
    unique={r['assembly_id']:r for r in records('test') if r['category']=='H-B'}
    inventory={}
    for identity,row in sorted(unique.items()):
        target=OPTICAL/'raymaps'/(identity+'.npz')
        if target.exists():
            try:
                with zipfile.ZipFile(target) as archive:
                    if archive.testzip() is not None:raise ValueError('CAD map CRC failure')
            except (zipfile.BadZipFile,ValueError):
                archive=OUT/'archive/incomplete_raymaps';archive.mkdir(parents=True,exist_ok=True)
                target.replace(archive/(identity+'_%d.npz'%time.time()))
        target=raymap(row)
        with np.load(target) as data:
            owner,local=data['owner'],data['local']
            assert owner.shape==(256,256) and local.shape==(256,256,3)
            assert np.isfinite(local[owner>=0]).all()
            assert set(np.unique(owner)).issubset({-1,0,1,2,3})
        inventory[identity]=dict(path=str(target.relative_to(ROOT)),sha256=sha(target))
    report=dict(complete=True,assemblies=len(inventory),files=inventory,
        dataset_manifest_sha256=sha(OUT/'dataset_manifest.json'),
        measurement_protocol_sha256=sha(OUT/'measurement_protocol.json'),
        preparation='serial completion before concurrent reference/generated reads',
        coordinate_algorithm_changed=False,updated_unix=time.time())
    write(OUT/'raymap_cache_ready.json',report)
    return report


if __name__=='__main__':print(json.dumps(prepare()))
