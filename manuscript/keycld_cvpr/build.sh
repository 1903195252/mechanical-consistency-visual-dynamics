#!/usr/bin/env bash
set -euo pipefail
paper_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$paper_dir"
mkdir -p build
export TMPDIR="$paper_dir/build"
if [[ -f "$paper_dir/../../output/closed_chain_eval/continuous/scores.json" ]]; then
  python3 scripts/prepare_unified_scoring.py > build/unified-scoring.log
  python3 scripts/prepare_resolution_artifacts.py > build/resolution-data.log
fi
for name in main supplement; do
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "$name.tex" > "build/$name.pass1.log" 2>&1
  (
    cd build
    BIBINPUTS="$paper_dir:" BSTINPUTS="$paper_dir:" bibtex "$name" > "$name.bibtex.log" 2>&1
  )
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "$name.tex" > "build/$name.pass2.log" 2>&1
  pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "$name.tex" > "build/$name.pass3.log" 2>&1
  for pass in 4 5; do
    if rg -q 'Rerun to get|Linenumber reference failed|undefined references' "build/$name.log"; then
      pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build "$name.tex" > "build/$name.pass$pass.log" 2>&1
    else
      break
    fi
  done
  cp "build/$name.pdf" "$name.pdf"
done
pdfinfo main.pdf | head -18
pdfinfo supplement.pdf | head -18
