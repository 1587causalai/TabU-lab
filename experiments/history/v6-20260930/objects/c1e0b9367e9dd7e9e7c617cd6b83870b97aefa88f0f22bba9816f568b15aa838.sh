#!/usr/bin/env bash
set -euo pipefail

base=/home/cms/experiments/v6-broadcast-oldloss-openml12-20260927
manifest=/home/cms/experiments/openml12-joint2h-squared-20260927/manifests/joint-openml12-squared.json
parent_sha=86f380c39c4da737eb92de301e1c69ac11c47304aa23b86ee034a40bfb3ad952
parent=/home/cms/experiments/openml12-joint30m-squared-20260927/half2/run/checkpoints/$parent_sha.pt
cd "$base/source/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$base/smoke.py', run_name='__main__')" \
  --manifest "$manifest" --parent "$parent" --parent-sha "$parent_sha" --device cuda:0
