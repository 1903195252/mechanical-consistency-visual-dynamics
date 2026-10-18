"""Canonical S/H identities with explicit, immutable legacy artifact lineage."""
from pathlib import Path


def canonical_id(identity):
    if identity.startswith(('M-A', 'M-B')):
        return 'H' + identity[1:]
    return identity


def canonical_entry(db, entry):
    identity = canonical_id(entry['assembly_id'])
    row = db.execute('SELECT * FROM assemblies WHERE assembly_id=?', (identity,)).fetchone()
    if row is None:
        raise ValueError('Assembly is outside the active S/H corpus: ' + identity)
    return dict(row)


def video_aliases(folder, source_id, identity):
    """Expose renamed files without altering media, signed sidecars or hashes."""
    if source_id == identity:
        return
    directory = Path(folder) / 'videos'
    if not directory.exists():
        return
    for source in directory.glob(source_id + '-V*'):
        if not source.is_file() or source.is_symlink():
            continue
        alias = source.with_name(identity + source.name[len(source_id):])
        if not alias.exists() and not alias.is_symlink():
            alias.symlink_to(source.name)


def canonical_path(path):
    return None if path is None else '/'.join(canonical_id(part) for part in str(path).split('/'))
