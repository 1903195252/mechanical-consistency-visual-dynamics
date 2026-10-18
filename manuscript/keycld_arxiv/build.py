#!/usr/bin/env python3
"""Build, check, and package this standalone arXiv manuscript."""
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
BUILD = ROOT / "build"
AUTHORS = ("Qibin Gu", "Yunfeng Yan")
NATIVE_AUTHOR_NAMES = ("顾琪斌", "闫云凤")
AFFILIATIONS = ("Nanjing Tech University", "Zhejiang University")
EMAILS = ("guqibin123@njetch.edu.cn", "yyff@zju.edu.cn")
POLICY_CHECKED_DATE = "2026-10-17"
POLICY_URLS = [
    "https://info.arxiv.org/help/submit_tex.html",
    "https://info.arxiv.org/help/00README.html",
    "https://info.arxiv.org/help/prep.html",
]


def run(args, cwd, log_name, env=None):
    log_path = BUILD / log_name
    with log_path.open("w") as log:
        result = subprocess.run(args, cwd=str(cwd), env=env,
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError("Command failed; see " + str(log_path))


def check_pdf(pdf, log):
    problems = re.findall(
        r"^.*(?:Overfull|Undefined control sequence|undefined references|"
        r"Citation .*undefined|Reference .*undefined|multiply defined|"
        r"destination with the same identifier|Rerun to get|"
        r"rerunfilecheck Warning).*$", log, re.M)
    assert not problems, problems
    text = subprocess.check_output(["pdftotext", "-layout", str(pdf), "-"], text=True)
    pages = [page for page in text.split("\f") if page.strip()]
    assert all(author in pages[0] for author in AUTHORS)
    assert all(affiliation in pages[0] for affiliation in AFFILIATIONS)
    assert all(email in pages[0] for email in EMAILS)
    assert pages[0].index(AUTHORS[0]) < pages[0].index(AUTHORS[1])
    assert pages[0].index(EMAILS[0]) < pages[0].index(EMAILS[1])
    assert not re.search(r"CONFIDENTIAL REVIEW COPY|Anonymous Authors|"
                         r"Anonymous CVPR submission|Paper ID DRAFT", text)
    assert "Supplementary Material" in text
    assert "Figure S8." in text and "Table S19." in text
    info = subprocess.check_output(["pdfinfo", str(pdf)], text=True)
    assert re.search(r"^Author:\s+" + re.escape(", ".join(AUTHORS)) + r"\s*$", info, re.M)
    assert re.search(r"^JavaScript:\s+no\s*$", info, re.M)
    fonts = subprocess.check_output(["pdffonts", str(pdf)], text=True)
    assert "Type 3" not in fonts
    font_rows = fonts.splitlines()[2:]
    for row in font_rows:
        flags = re.search(r"\s+(yes|no)\s+(yes|no)\s+(yes|no)\s+\d+\s+\d+\s*$", row)
        assert flags and flags.group(1) == "yes", row
    references = [i + 1 for i, page in enumerate(pages)
                  if re.search(r"^\s*References(?:\s{2,}|$)", page, re.M)]
    assert len(references) == 1, references
    appendix_page = next(i + 1 for i, page in enumerate(pages)
                         if "Supplementary Material" in page)
    return {"pages": len(pages), "references_start_page": references[0],
            "supplement_start_page": appendix_page,
            "fonts_embedded": True, "type3_fonts": False,
            "review_marks_removed": True, "author_contacts_verified": True,
            "unresolved_build_issues": []}


def main():
    BUILD.mkdir(exist_ok=True)
    tex_files = sorted(SOURCE.rglob("*.tex"))
    standalone = [str(p.relative_to(SOURCE)) for p in tex_files
                  if r"\documentclass" in p.read_text()]
    assert standalone == ["main.tex"], standalone
    assert not any("/home/" in p.read_text() or "/media/" in p.read_text()
                   for p in tex_files)
    assert not any(re.search(r"\\input\{tables/", p.read_text()) for p in tex_files), \
        "Run sync_source.py to embed table rows before packaging"
    command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
               "-halt-on-error", "-file-line-error", "-recorder",
               "-output-directory=" + str(BUILD), "main.tex"]
    run(command, SOURCE, "main.pass1.log")
    env = os.environ.copy()
    for variable in ("BIBINPUTS", "BSTINPUTS"):
        env[variable] = str(SOURCE) + os.pathsep + env.get(variable, "")
    run(["bibtex", "main"], BUILD, "main.bibtex.log", env)
    for number in range(2, 5):
        run(command, SOURCE, "main.pass%d.log" % number)
    summary = check_pdf(BUILD / "main.pdf", (BUILD / "main.log").read_text())
    assert summary["references_start_page"] == 9, summary
    shutil.copyfile(BUILD / "main.bbl", SOURCE / "main.bbl")
    bib = (SOURCE / "main.bbl").read_text()
    bibliography_keys = re.findall(r"\\bibitem(?:\[[\s\S]*?\])?\{([^}]+)\}", bib)
    source_keys = re.findall(r"@\w+\{([^,]+),", (SOURCE / "references.bib").read_text())
    assert len(bibliography_keys) == len(source_keys)
    assert set(bibliography_keys) == set(source_keys)
    assert r'\bibliographystyle{ieeenat_fullname}' in (SOURCE / "main.tex").read_text()

    files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    names = [str(path.relative_to(SOURCE)) for path in files]
    allowed = {".tex", ".sty", ".bst", ".bib", ".bbl", ".pdf", ".png", ".json"}
    assert all(path.suffix in allowed for path in files)
    assert all(not path.is_symlink() for path in files)
    assert all(not any(part.startswith(".") for part in Path(name).parts)
               for name in names)
    assert all(name.startswith("figures/") for name in names
               if Path(name).suffix in (".pdf", ".png"))
    target = ROOT / "arxiv_source.zip"
    candidate = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(candidate, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path, name in zip(files, names):
            archive.write(path, name)
    # Build the exact upload in isolation, using its supplied .bbl without BibTeX.
    with tempfile.TemporaryDirectory(prefix="upload-check-", dir=str(BUILD)) as work:
        work = Path(work)
        with zipfile.ZipFile(candidate) as archive:
            assert archive.testzip() is None
            archive.extractall(work)
        isolated_command = ["pdflatex", "-no-shell-escape", "-interaction=nonstopmode",
                            "-halt-on-error", "-file-line-error", "main.tex"]
        for number in range(1, 4):
            run(isolated_command, work, "upload-check.pass%d.log" % number)
        isolated = check_pdf(work / "main.pdf", (work / "main.log").read_text())
        assert isolated == summary
    candidate.replace(target)
    shutil.copyfile(BUILD / "main.pdf", ROOT / "main.pdf")

    main_source = (SOURCE / "main.tex").read_text()
    preamble = (SOURCE / "preamble.tex").read_text()
    title = re.search(r"\\newcommand\{\\keytitle\}\{([^}]+)\}", preamble).group(1)
    abstract = re.search(r"\\begin\{abstract\}\s*(.*?)\s*\\end\{abstract\}",
                         main_source, re.S).group(1)
    abstract = " ".join(abstract.replace(r"\%", "%").replace("$", "").split())
    affiliations = "\n".join(name + ": " + affiliation
                              for name, affiliation in zip(AUTHORS, AFFILIATIONS))
    contacts = "\n".join(f"{name} ({native}): {email}"
                         for name, native, email in zip(AUTHORS, NATIVE_AUTHOR_NAMES, EMAILS))
    combined_source = main_source + (SOURCE / "appendix.tex").read_text()
    figure_count = (len(re.findall(r"\\(?:begin\{figure\*?\}|captionof\{figure\})", combined_source))
                    - len(re.findall(r"\\ContinuedFloat\b", combined_source)))
    table_count = len(re.findall(r"\\(?:begin\{table\*?\}|captionof\{table\})", combined_source))
    metadata = ("Title: " + title + "\n\nAuthors: " + ", ".join(AUTHORS) +
                "\n\nAffiliations:\n" + affiliations +
                "\n\nAuthor contacts:\n" + contacts +
                "\n\nAbstract:\n" + abstract + "\n\nComments: " +
                str(summary["pages"]) + " pages, including supplementary material; "
                f"{figure_count} figures and {table_count} tables.\n")
    (ROOT / "submission_metadata.txt").write_text(metadata)
    summary.update({
        "mode": "arxiv-package-check", "checked_date": str(date.today()),
        "authors": [{"name": name, "native_name": native,
                     "affiliation": affiliation, "email": email}
                    for name, native, affiliation, email in zip(
                        AUTHORS, NATIVE_AUTHOR_NAMES, AFFILIATIONS, EMAILS)],
        "compiler": subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0],
        "engine": "pdflatex", "entry_point": "main.tex",
        "source_files": len(files),
        "figure_files": sum(name.startswith("figures/") for name in names),
        "table_inputs": sum(name.startswith("tables/") for name in names),
        "table_rows_embedded": True,
        "bibliography_entries": len(bibliography_keys), "isolated_zip_build_passed": True,
        "bibliography_style": "ieeenat_fullname",
        "bibliography_order": "author_alphabetical",
        "arxiv_server_build": "Not run; local source and exact ZIP builds passed",
        "official_guidance": POLICY_URLS,
        "official_guidance_checked_date": POLICY_CHECKED_DATE,
        "project_state_tracking": "Not configured (ccfa.yaml absent)",
        "archive_size_bytes": target.stat().st_size,
        "archive_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path, name in zip(files, names)},
    })
    (ROOT / "submission_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "pages", "supplement_start_page", "source_files", "archive_size_bytes",
        "isolated_zip_build_passed")}, indent=2))


if __name__ == "__main__":
    main()
