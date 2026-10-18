#!/usr/bin/env bash
set -euo pipefail
FOURBAR_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$FOURBAR_PROJECT_ROOT"
FOURBAR_RENDER_ONLY=false
if [[ "${1:-}" == --render-only && $# == 1 ]]; then
  FOURBAR_RENDER_ONLY=true
elif [[ $# != 0 ]]; then
  echo 'Usage: bash experiments/fourbar/run.sh [--render-only]' >&2
  exit 2
fi
export TMPDIR="$FOURBAR_PROJECT_ROOT/output/fourbar/.tmp"
export XDG_CACHE_HOME="$FOURBAR_PROJECT_ROOT/output/fourbar/.cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
FOURBAR_PYTHON="${FOURBAR_PYTHON:-$(command -v python3)}"
if [[ ! -x "$FOURBAR_PYTHON" ]]; then
  echo 'Create the project environment described in experiments/fourbar/README.md first.' >&2
  exit 1
fi
if [[ -z "${BLENDER_BIN:-}" ]]; then
  if command -v blender >/dev/null 2>&1; then
    BLENDER_BIN="$(command -v blender)"
  else
    echo 'Set BLENDER_BIN to the Blender executable.' >&2
    exit 1
  fi
fi
if [[ "$FOURBAR_RENDER_ONLY" != true ]]; then
  "$FOURBAR_PYTHON" experiments/fourbar/test_mechanics.py
  "$FOURBAR_PYTHON" experiments/fourbar/simulate.py --all-cases --render
  "$FOURBAR_PYTHON" experiments/fourbar/export_physics.py
fi
"$BLENDER_BIN" -b --factory-startup --python experiments/fourbar/blender_scene.py -- --case nominal --render --beauty
for FOURBAR_CASE in heavy_payload high_friction; do
  "$BLENDER_BIN" -b --factory-startup --python experiments/fourbar/blender_scene.py -- --case "$FOURBAR_CASE" --preview-only --beauty
done
ffmpeg -y -loglevel warning -framerate 30 -start_number 1 \
  -i output/fourbar/nominal/blender_frames/frame_%04d.png \
  -c:v libx264 -crf 18 -pix_fmt yuv420p output/fourbar/nominal/blender_rgb.mp4
"$FOURBAR_PYTHON" experiments/fourbar/finalize.py
