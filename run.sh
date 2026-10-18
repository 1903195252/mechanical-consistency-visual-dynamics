#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$repo_dir"
mode="${1:-verify}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
  verify) environment=audit ;;
  audit|planar|spatial) environment="$mode" ;;
  *) echo "Usage: ./run.sh {verify|audit|planar|spatial} [python arguments]" >&2; exit 2 ;;
esac
python_bin="$repo_dir/.venv-$environment/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "Environment missing. Run ./setup.sh --profile $environment first." >&2
  exit 2
fi
export PYTHONPATH="$repo_dir:$repo_dir/output/keycld_corpus/runtime:$repo_dir/official${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="$repo_dir/output/runtime/matplotlib"
export WANDB_MODE=disabled
export XLA_PYTHON_CLIENT_PREALLOCATE=false
if [[ "$mode" == verify ]]; then
  "$python_bin" -m unittest experiments.closed_chain_eval.test_core experiments.closed_chain_eval.test_continuous experiments.closed_chain_eval.test_instrument_optimization experiments.closed_chain_eval.test_visual_calibration -v
  "$python_bin" experiments/fourbar/test_mechanics.py
  "$python_bin" scripts/verify_results.py
  "$python_bin" -m unittest discover -s tests -v
else
  exec "$python_bin" "$@"
fi
