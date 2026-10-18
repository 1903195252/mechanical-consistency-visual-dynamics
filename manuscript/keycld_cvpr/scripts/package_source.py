"""Package the current paper sources and assets, excluding datasets and build scratch."""
from pathlib import Path
import json
import re
import zipfile

PAPER = Path(__file__).resolve().parents[1]


def main():
    checked = json.loads((PAPER / 'format_check.json').read_text())
    assert checked['main']['main_body_pages'] <= 8
    sources = [(PAPER / name).read_text() for name in ('main.tex', 'supplement.tex')]
    files = {PAPER / name for name in ('main.tex', 'supplement.tex', 'preamble.tex', 'references.bib',
             'cvpr.sty', 'ieeenat_fullname.bst', 'README.md', 'build.sh', 'format_check.json')}
    for text in sources:
        for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text):
            graphic = PAPER / name
            assert graphic.exists(), name
            files.add(graphic)
            editable = graphic.with_suffix('.svg')
            if editable.exists(): files.add(editable)
    files.add(PAPER / 'figures/audit_mechanism.drawio')
    for folder, pattern in [('tables','*.tex'), ('scripts','*.py'), ('sources','*.json'), ('figures/assets','*.png')]:
        files.update((PAPER / folder).glob(pattern))
    target = PAPER / 'keycld_cvpr_source.zip'
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(files): archive.write(path, str(path.relative_to(PAPER)))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert archive.read('main.tex') == (PAPER/'main.tex').read_bytes()
        assert archive.read('supplement.tex') == (PAPER/'supplement.tex').read_bytes()
        for name in archive.namelist():
            assert not name.startswith('build/') and not name.endswith('.npz')
    temporary.replace(target)
    print(json.dumps({'archive': str(target.relative_to(PAPER)), 'files':len(files),
                      'size_MiB':round(target.stat().st_size/1024**2,2)},indent=2))


if __name__ == '__main__': main()
