"""Freeze the expanded test cohort without changing the original corpus."""
import argparse
import hashlib
import json
import sqlite3

from .common import (ROOT, OUT, CORPUS, OLD_CORPUS, OLD_MANIFEST, OLD_RUN,
                     FAMILIES, CONDITIONS, SEEDS, STEPS, sha, write, frozen,
                     assert_split_isolation, generation_seed, TEST_ASSEMBLIES_PER_FAMILY,
                     NEW_ASSEMBLIES_PER_FAMILY, NEW_ASSEMBLIES, TEST_CASES)


def initialize():
    original = json.loads(OLD_MANIFEST.read_text())
    rows = [r for r in original['videos'] if r['split'] == 'train' or r['view'] == 'front']
    entries = []
    old = sqlite3.connect('file:' + str(OLD_CORPUS / 'catalog.sqlite') + '?mode=ro', uri=True)
    old.row_factory = sqlite3.Row
    for family in FAMILIES:
        existing = [r for r in rows if r['split'] == 'test' and r['category'] == family]
        assert len(existing) == 15 and len({r['assembly_id'] for r in existing}) == 5
        for ordinal in range(51, 51 + NEW_ASSEMBLIES_PER_FAMILY):
            hardware = (ordinal - 1) % 12 + 1
            identity = '%s-%03d-P%02d' % (family, ordinal, hardware)
            seed_id = 'M' + identity[1:]  # Continue the existing geometry seed namespace.
            seed = int.from_bytes(hashlib.sha256(seed_id.encode()).digest()[:4], 'big') & 0x7fffffff
            if old.execute('SELECT 1 FROM assemblies WHERE assembly_id=?', (identity,)).fetchone():
                raise ValueError('New identity already exists: ' + identity)
            entry = dict(assembly_id=identity, category=family, tier='H', ordinal=ordinal,
                         hardware_recipe=hardware, seed=seed, split='test',
                         builder='stewart' if family == 'H-A' else 'delta')
            entries.append(entry)
            folder = CORPUS / 'assemblies' / family / identity
            for index, condition in enumerate(CONDITIONS):
                number = 5 * index + 1
                video_id = identity + '-V%02d' % number
                video = folder / 'videos' / (video_id + '.mp4')
                rows.append(dict(assembly_id=identity, source_assembly_id=identity,
                    category=family, split='test', video_id=video_id,
                    condition=condition, view='front', video=str(video.relative_to(ROOT)),
                    public_inputs=str(video.with_suffix('.inputs.json').relative_to(ROOT)),
                    commands=str((folder / 'conditions' / condition / 'commands.npz').relative_to(ROOT)),
                    evaluator_directory=str((folder / 'conditions' / condition).relative_to(ROOT)),
                    observed_prefix_frames=3, depth_available=False,
                    camera_projection='orthographic'))
    assert_split_isolation(rows)
    spec = dict(schema_version=2, model='LingBot-VA official full 5.09B shared backbone',
        original_manifest_sha256=sha(OLD_MANIFEST), seeds=list(SEEDS), checkpoints=list(STEPS),
        families={f: dict(test_assemblies=TEST_ASSEMBLIES_PER_FAMILY, test_trajectories=TEST_CASES//2, existing_test_assemblies=5,
                         additional_test_assemblies=NEW_ASSEMBLIES_PER_FAMILY) for f in FAMILIES},
        training=dict(joint_families=True, full_parameter_adaptation=True, updates_per_seed=STEPS[-1],
            total_updates=STEPS[-1]*len(SEEDS), batch_size=8, microbatch_size=1, accumulation_steps=8,
            learning_rate=1e-5, warmup_updates=10, sequence_exposures_per_seed=STEPS[-1]*8,
            input_resolution=[256,256], native_video_action_objective=True,
            starting_checkpoint='official pretrained weights independently for each seed',
            checkpoint_selection='Both registered steps; no test-set selection',
            training_videos=sum(r['split']=='train' for r in rows),
            exact_resume='FP32 model + AdamW + scheduler + RNG + data cursor; BF16 exports are evaluation-only'),
        inference=dict(test_cases=TEST_CASES, model_seed_checkpoint_cases=TEST_CASES*len(SEEDS)*len(STEPS), future_frames=117,
            total_future_frames=TEST_CASES*len(SEEDS)*len(STEPS)*117, fps=30, observed_frames=3, chunks=8,
            denoising_steps=5, guidance_scale=5,
            generation_seed='SHA256(lingbot60:<training_seed>:<video_id>), constant across checkpoints',
            zero_step='One shared base weight set; three paired generation seed sets'),
        evaluation=dict(scoring_version='continuous-si-1', primary='fixed-engineering-tolerance continuous Scomp',
            auxiliary='strict joint qualification and physical coverage',
            reader='frozen CAD-coordinate head plus frozen V-JEPA 2.1-B encoder',
            tracks=['native_coordinates','CAD_IK_reconstruction'],
            instrument_controls=['matched_reference_RGB','exact_states','sampled_state_derivatives'],
            fit_test_labels=False, subtract_reference_RMSE=False,
            statistical_unit='assembly; three conditions are clustered within each identity',
            condition_and_family_balanced=True),
        production=dict(new_assemblies=NEW_ASSEMBLIES, new_videos=NEW_ASSEMBLIES*len(CONDITIONS), views=['front'], source_resolution=[640,640],
            model_resize='640 to 256 using INTER_AREA, identical to the existing experiment',
            full_physics_checks=True, full_360_pose_mesh_checks=True,
            original_corpus_modified=False, failed_cases='Retain failures; do not silently replace sampled identities'))
    frozen(OUT / 'protocol.json', spec)
    frozen(OUT / 'assembly_plan.json', dict(original_manifest_sha256=sha(OLD_MANIFEST), entries=entries))
    frozen(OUT / 'planned_dataset.json', dict(ready=False, videos=rows,
        note='A planned record is not an accepted video; dataset_manifest.json is published after all checks pass.'))
    CORPUS.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(CORPUS / 'catalog.sqlite'))
    for table in ('metadata', 'assemblies', 'videos'):
        sql = old.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()[0]
        db.execute(sql.replace('CREATE TABLE ', 'CREATE TABLE IF NOT EXISTS ', 1))
    for entry in entries:
        columns = ','.join(entry)
        db.execute('INSERT OR IGNORE INTO assemblies (' + columns + ') VALUES (' + ','.join('?' for _ in entry) + ')', tuple(entry.values()))
        for index, condition in enumerate(CONDITIONS):
            number = 5 * index + 1
            db.execute('INSERT OR IGNORE INTO videos (video_id,assembly_id,video_index,condition,view) VALUES (?,?,?,?,?)',
                       (entry['assembly_id']+'-V%02d'%number, entry['assembly_id'], number, condition, 'front'))
    db.commit(); db.close(); old.close()
    jobs = [dict(seed=seed, checkpoint=step, cases=TEST_CASES,
                 output='runs/seed%d/step%04d'%(seed,step)) for seed in SEEDS for step in STEPS]
    frozen(OUT / 'evaluation_plan.json', dict(jobs=jobs, total_cases=TEST_CASES*len(SEEDS)*len(STEPS),
        sample_generation_seeds={str(seed):generation_seed(seed,rows[-1]['video_id']) for seed in SEEDS}))
    # Frozen instrumentation is referenced by hash; none is trained on new tests.
    files = [OLD_MANIFEST, OLD_RUN/'protocol.json', OLD_RUN/'official_weight_manifest.json',
             OLD_RUN/'action_norm.json', OLD_RUN/'text_embeddings.pt',
             ROOT/'output/model_expansion/weights/vjepa2_1_vitb_dist_vitG_384.pt']
    files += [ROOT/'output/closed_chain_eval/visual_calibration/optimization/geometry_spatial_reader'/(f+'_head.pt') for f in FAMILIES]
    files += [ROOT/'output/model_expansion/spatial'/(f+'_calibrate.json') for f in FAMILIES]
    frozen(OUT/'input_bindings.json',dict(files={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files}))
    print(json.dumps(dict(protocol=str(OUT/'protocol.json'),new_assemblies=NEW_ASSEMBLIES,test_trajectories=TEST_CASES,jobs=len(jobs))))


def publish():
    from .production import acceptance_status
    status = acceptance_status()
    if status['accepted_new_assemblies'] != NEW_ASSEMBLIES:
        raise ValueError('Dataset incomplete: %d/%d new assemblies accepted' % (status['accepted_new_assemblies'], NEW_ASSEMBLIES))
    planned = json.loads((OUT/'planned_dataset.json').read_text())
    rows = []
    for old in planned['videos']:
        row = dict(old)
        for key in ('video','public_inputs','commands'):
            path = ROOT/row[key]
            if not path.is_file():raise FileNotFoundError(path)
            row[key+'_sha256'] = sha(path)
        rows.append(row)
    assert_split_isolation(rows)
    frozen(OUT/'dataset_manifest.json',dict(ready=True,protocol_sha256=sha(OUT/'protocol.json'),videos=rows))
    return status


def publish_training():
    original=json.loads(OLD_MANIFEST.read_text())['videos']
    original=[r for r in original if r['split']=='train']
    rows=[r for r in json.loads((OUT/'planned_dataset.json').read_text())['videos'] if r['split']=='train']
    if rows!=original or len(rows)!=270:raise ValueError('Training partition differs from the original frozen corpus')
    verified=[]
    for old in rows:
        row=dict(old)
        for key in ('video','public_inputs','commands'):row[key+'_sha256']=sha(ROOT/row[key])
        verified.append(row)
    frozen(OUT/'training_manifest.json',dict(ready=True,scope='training-only',
        original_manifest_sha256=sha(OLD_MANIFEST),protocol_sha256=sha(OUT/'protocol.json'),videos=verified))
    print(json.dumps(dict(training_videos=len(verified),test_records_included=0)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['initialize','publish','publish-training'])
    args=parser.parse_args()
    if args.command=='initialize':initialize()
    elif args.command=='publish-training':publish_training()
    else:print(json.dumps(publish()))
