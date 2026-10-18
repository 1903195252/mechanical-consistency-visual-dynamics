"""Sync scientific content from the canonical draft, retaining arXiv authors."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parent
CANONICAL = ROOT.parent / "keycld_cvpr"
SOURCE = ROOT / "source"
TABLE_INPUT = re.compile(r"\\input\{(tables/[^}]+)\}")


def inline_tables(text):
    """Keep table rows contiguous across LaTeX file-hook versions."""
    def expand(match):
        return (CANONICAL / (match.group(1) + ".tex")).read_text().rstrip("\n")

    expanded = TABLE_INPUT.sub(expand, text)
    assert not TABLE_INPUT.search(expanded), "Nested table inputs need expansion"
    return expanded


def main():
    canonical_preamble = (CANONICAL / "preamble.tex").read_text()
    title = re.search(r"\\newcommand\{\\keytitle\}\{([^}]+)\}", canonical_preamble).group(1)
    preamble_path = SOURCE / "preamble.tex"
    preamble = preamble_path.read_text()
    preamble = re.sub(r"(\\newcommand\{\\keytitle\}\{)[^}]+\}",
                      lambda match: match.group(1) + title + "}", preamble)
    preamble = re.sub(r"(pdftitle=\{)[^}]+\}",
                      lambda match: match.group(1) + title + "}", preamble)
    preamble_path.write_text(preamble)
    arxiv = (SOURCE / "main.tex").read_text()
    header = arxiv[:arxiv.index(r"\begin{abstract}")]
    suffix = arxiv[arxiv.index(r"\label{maintextend}"):]
    paper = (CANONICAL / "main.tex").read_text()
    bibliography_style = re.search(r"\\bibliographystyle\{([^}]+)\}", paper).group(1)
    suffix = re.sub(r"\\bibliographystyle\{[^}]+\}",
                    lambda match: "\\bibliographystyle{" + bibliography_style + "}", suffix)
    # Preserve canonical float spacing as well as scientific content; author
    # metadata remains specific to the signed version.
    float_layout = re.compile(r"\\setlength\{\\(?:dbltextfloatsep|textfloatsep)\}\{[^}]+\}\n")
    header = float_layout.sub("", header)
    spacing = "".join(float_layout.findall(paper[:paper.index(r"\begin{abstract}")]))
    header = header.replace(r"\maketitle", spacing + r"\maketitle")
    body = inline_tables(paper[paper.index(r"\begin{abstract}"):paper.index(r"\label{maintextend}")])
    (SOURCE / "main.tex").write_text(header + body + suffix)
    old_appendix = (SOURCE / "appendix.tex").read_text()
    prefix = old_appendix[:old_appendix.index(r"\label{supplementstart}")] + "\\label{supplementstart}\n\n"
    supplementary = (CANONICAL / "supplement.tex").read_text()
    appendix = supplementary.split(r"\appendix", 1)[1]
    appendix = inline_tables(appendix[:appendix.rindex(r"\FloatBarrier")])
    (SOURCE / "appendix.tex").write_text(prefix + appendix.strip() + "\n\\FloatBarrier\n")
    for name in ("cvpr.sty", bibliography_style + ".bst", "references.bib"):
        shutil.copyfile(CANONICAL / name, SOURCE / name)
    for path in SOURCE.glob("*.bst"):
        if path.stem != bibliography_style:
            path.unlink()
    referenced_assets = set()
    for content in (body, appendix):
        paths = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", content)
        for name in paths:
            referenced_assets.add(Path(name))
            target = SOURCE / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(CANONICAL / name, target)
    # These directories contain generated publication assets. Keep the upload
    # limited to the figures and table inputs used by the synchronized text.
    for folder in ("figures", "tables"):
        for path in (SOURCE / folder).iterdir():
            if path.is_file() and path.relative_to(SOURCE) not in referenced_assets:
                path.unlink()
    print("Canonical manuscript and figures synchronized; table rows embedded; author metadata retained.")


if __name__ == "__main__":
    main()
