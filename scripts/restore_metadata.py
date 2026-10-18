"""Install included metadata without overwriting a different runtime file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = json.loads((ROOT/'configs/metadata_layout.json').read_text())
    for destination, source in entries.items():
        target = ROOT/destination
        if target.exists() and target.read_bytes() != (ROOT/source).read_bytes():
            raise FileExistsError('Existing metadata differs: '+destination)
    for destination, source in entries.items():
        target = ROOT/destination
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/source).read_bytes())
    print('Restored %d metadata files. Dataset arrays and weights are separate artifacts.' % len(entries))


if __name__ == '__main__':
    main()
