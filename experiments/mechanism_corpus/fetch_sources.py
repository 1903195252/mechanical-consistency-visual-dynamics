"""Download pinned open resources with their original licenses and lineage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .catalog import OUTPUT, write_json

SOURCES = [
    {"name": "mujoco_menagerie", "repo": "https://github.com/google-deepmind/mujoco_menagerie.git",
     "revision": "0059d4335f8156206f63a35662313385f7ad6d74",
     "paths": ["universal_robots_ur5e","wonik_allegro","unitree_go2"],
     "candidate_categories": ["M-I","H-B","H-G"]},
    {"name": "mechanism", "repo": "https://github.com/gabemorris12/mechanism.git",
     "revision": "b68554a5415812b3c63fb5d8a4db4db7e9f8c98a",
     "paths": None, "candidate_categories": ["S-A","S-B","S-C","S-E","S-H"]},
    {"name": "freecad_gears", "repo": "https://github.com/looooo/freecad.gears.git",
     "revision": "83ec154b1925347622b61812f75d2ed51e956b9f",
     "paths": ["pygears","freecad","examples"],
     "candidate_categories": ["S-C","S-F","M-E","M-F","M-K","H-C","H-I"]},
    {"name": "sprk", "repo": "https://github.com/BerkeleyAutomation/sprk.git",
     "revision": "3072506978dd4a734a32168830cc271c5fc001f6",
     "paths": None, "candidate_categories": ["H-A"]},
]


def git(folder, *arguments, output=False):
    command = ["git","-c","http.version=HTTP/1.1","-C",str(folder),*arguments]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE if output else None,
                            stderr=subprocess.PIPE if output else None, text=True,
                            env={**os.environ,"GIT_LFS_SKIP_SMUDGE":"1"})
    return result.stdout if output else None


def main():
    root = OUTPUT / "upstream"
    root.mkdir(parents=True, exist_ok=True)
    registry = []
    for source in SOURCES:
        folder = root / source["name"]
        folder.mkdir(exist_ok=True)
        if not (folder / ".git").exists():
            git(folder,"init","--quiet")
            git(folder,"remote","add","origin",source["repo"])
        current = git(folder,"rev-parse","--verify","HEAD",output=True).strip() if (folder/".git"/"HEAD").exists() and \
            subprocess.run(["git","-C",str(folder),"rev-parse","--verify","HEAD"],
                           stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode == 0 else None
        if current != source["revision"]:
            if current:
                raise RuntimeError("Existing source revision differs; preserve it: "+source["name"])
            if source["paths"]:
                git(folder,"sparse-checkout","init","--cone")
                git(folder,"sparse-checkout","set",*source["paths"])
            git(folder,"config","core.repositoryformatversion","1")
            git(folder,"config","extensions.partialClone","origin")
            git(folder,"config","remote.origin.promisor","true")
            git(folder,"config","remote.origin.partialclonefilter","blob:none")
            git(folder,"fetch","--depth=1","--filter=blob:none","origin",source["revision"])
            git(folder,"checkout","--detach","FETCH_HEAD")
        assert git(folder,"rev-parse","HEAD",output=True).strip() == source["revision"]
        files=[]
        licenses=[]
        lfs=[]
        for name in git(folder,"ls-files","-z",output=True).split("\0"):
            path=folder/name
            if not name or not path.is_file() or path.is_symlink():
                continue
            content=path.read_bytes()
            files.append({"path":name,"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
            if path.name.lower().startswith(("license","copying")):
                licenses.append({"path":name,"text":content.decode(errors="replace")})
            if content[:70].startswith(b"version https://git-lfs.github.com/spec/v1"):
                lfs.append(name)
        record={**source,"downloaded":True,"tracked_files_present":len(files),
                "total_bytes":sum(row["bytes"] for row in files),"licenses":licenses,
                "unresolved_lfs_files":lfs,"files":files,
                "use_state":"downloaded_for_inspection_not_yet_a_corpus_assembly"}
        write_json(folder.parent/(source["name"]+"_provenance.json"),record)
        registry.append({key:value for key,value in record.items() if key!="files"})
        write_json(OUTPUT/"source_registry.json",registry)
        print(json.dumps({"source":source["name"],"files":len(files),"bytes":record["total_bytes"],
                          "license_files":[row["path"] for row in licenses],"unresolved_lfs":len(lfs)}),flush=True)


if __name__ == "__main__":
    main()
