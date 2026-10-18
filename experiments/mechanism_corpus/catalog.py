"""Persistent, auditable production catalog. Planned entries are not completed data."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output/mechanism_corpus"
SPEC_PATH = Path(__file__).with_name("specification.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".%d.partial" % os.getpid())
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def connect(output=OUTPUT):
    output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(output / "catalog.sqlite"), timeout=60)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize(output=OUTPUT):
    spec = json.loads(SPEC_PATH.read_text())
    assert len(spec["categories"]) == 7
    assert sum(x["assemblies"] for x in spec["tiers"].values()) == spec["target_assemblies"]
    assert spec["target_videos"] == 15 * spec["target_assemblies"]
    db = connect(output)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS assemblies(
          assembly_id TEXT PRIMARY KEY, category TEXT NOT NULL, tier TEXT NOT NULL,
          ordinal INTEGER NOT NULL, hardware_recipe INTEGER NOT NULL, seed INTEGER NOT NULL,
          split TEXT NOT NULL, builder TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'planned',
          geometry_digest TEXT, manifest_path TEXT, last_error TEXT,
          UNIQUE(category,ordinal));
        CREATE TABLE IF NOT EXISTS videos(
          video_id TEXT PRIMARY KEY, assembly_id TEXT NOT NULL REFERENCES assemblies(assembly_id),
          video_index INTEGER NOT NULL, condition TEXT NOT NULL, view TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'planned', path TEXT, validation_path TEXT, last_error TEXT,
          UNIQUE(assembly_id,video_index));
    """)
    fingerprint = hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()
    old = db.execute("SELECT value FROM metadata WHERE key='specification_sha256'").fetchone()
    if old and old["value"] != fingerprint:
        raise RuntimeError("Catalog specification changed; migrate explicitly before altering planned/completed identities")
    db.execute("INSERT OR IGNORE INTO metadata VALUES('specification_sha256',?)", (fingerprint,))
    for category in spec["categories"]:
        code = category["code"]
        tier = code[0]
        count = spec["tiers"][tier]["per_category"]
        for ordinal in range(1, count + 1):
            hardware = (ordinal - 1) % 12 + 1
            identity = "%s-%03d-P%02d" % (code, ordinal, hardware)
            seed_identity = spec.get("seed_namespace_aliases", {}).get(code, code) + identity[len(code):]
            seed = int.from_bytes(hashlib.sha256(seed_identity.encode()).digest()[:4], "big") & 0x7fffffff
            split = "train" if ordinal <= count*.8 else "validation" if ordinal <= count*.9 else "test"
            db.execute("""INSERT OR IGNORE INTO assemblies
                (assembly_id,category,tier,ordinal,hardware_recipe,seed,split,builder)
                VALUES(?,?,?,?,?,?,?,?)""",
                (identity, code, tier, ordinal, hardware, seed, split, category["implementation"]))
            for condition_index, condition in enumerate(spec["conditions"]):
                for view_index, view in enumerate(spec["views"]):
                    number = condition_index * 5 + view_index + 1
                    db.execute("""INSERT OR IGNORE INTO videos
                        (video_id,assembly_id,video_index,condition,view) VALUES(?,?,?,?,?)""",
                        (identity + "-V%02d" % number, identity, number, condition, view))
    db.commit()
    assert db.execute("SELECT count(*) FROM assemblies").fetchone()[0] == spec["target_assemblies"]
    assert db.execute("SELECT count(*) FROM videos").fetchone()[0] == spec["target_videos"]
    write_json(output / "specification.json", spec)
    rows = [dict(row) for row in db.execute("SELECT * FROM assemblies ORDER BY assembly_id")]
    manifest = output / "planned_assemblies.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    summary(db, output)
    return db


def summary(db, output=OUTPUT):
    tiers = {}
    for tier in ("S","H"):
        states = Counter(dict(row)["status"] for row in db.execute("SELECT status FROM assemblies WHERE tier=?", (tier,)))
        videos = Counter(row["status"] for row in db.execute(
            "SELECT v.status FROM videos v JOIN assemblies a USING(assembly_id) WHERE a.tier=?", (tier,)))
        tiers[tier] = {"assemblies": dict(states), "videos": dict(videos)}
    result = {"target_assemblies": db.execute("SELECT count(*) FROM assemblies").fetchone()[0],
              "target_videos": db.execute("SELECT count(*) FROM videos").fetchone()[0], "tiers": tiers,
              "naming_version": "SH-2.0",
              "counting_policy": "A planned row, preview, configuration or copied asset is not a completed assembly/video."}
    plan_path = output / 'video_completion_plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        codes = list(plan['categories'])
        marks = ','.join('?' for _ in codes)
        complete = db.execute("SELECT count(*) FROM assemblies WHERE category IN ("+marks+") AND status='verified'",codes).fetchone()[0]
        videos = db.execute("SELECT count(*) FROM videos v JOIN assemblies a USING(assembly_id) WHERE a.category IN ("+marks+") AND v.status='verified'",codes).fetchone()[0]
        splits = {r['split']:r['count'] for r in db.execute("SELECT split,count(*) AS count FROM assemblies WHERE category IN ("+marks+") AND status='verified' GROUP BY split",codes)}
        active = {'target_assemblies':plan['target_assemblies'],'target_videos':plan['target_videos'],
                  'verified_assemblies':complete,'verified_videos':videos,
                  'remaining_assemblies':plan['target_assemblies']-complete,'remaining_videos':plan['target_videos']-videos,
                  'verified_assembly_splits':splits,'categories':plan['categories']}
        result = {'production_status':plan['status'],'active_scope':active,
                  'target_status':'Active S/H corpus only; the original S/M/H design is preserved in archives/naming_SMH_v1.',**result}
    write_json(output / "status.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["initialize","status"])
    args = parser.parse_args()
    db = initialize() if args.command == "initialize" else connect()
    print(json.dumps(summary(db), ensure_ascii=False, indent=2))
    db.close()


if __name__ == "__main__":
    main()
