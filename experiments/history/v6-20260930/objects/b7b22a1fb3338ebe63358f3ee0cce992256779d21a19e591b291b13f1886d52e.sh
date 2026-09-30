#!/usr/bin/env bash
set -euo pipefail

root=/home/cms/experiments/v6-sparse-relevance-deeper-20260927
source_root=/home/cms/experiments/v6-target-broadcast-dgx2-20260927/source/src
parent_sha=86f380c39c4da737eb92de301e1c69ac11c47304aa23b86ee034a40bfb3ad952
parent=/home/cms/experiments/openml12-joint30m-squared-20260927/half2/run/checkpoints/$parent_sha.pt
cd "$source_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$root/run_selected.py', run_name='__main__')" \
  --probe-root "$root/seed2" --reference-run "$root/seed2/run-h8-u150" \
  --parent "$parent" --parent-sha "$parent_sha" \
  --output "$root/seed2/broadcast-only-h8-u150" --updates 150 \
  --variants broadcast-only --device cuda:0
