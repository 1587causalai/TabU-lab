#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo 'usage: eval_checkpoint.sh CHECKPOINT LABEL [TABLE ...]' >&2
  exit 2
fi

checkpoint=$1
label=$2
shift 2
base=/home/cms/experiments/v6-broadcast-oldloss-openml12-20260927
manifest=/home/cms/experiments/openml12-joint2h-squared-20260927/manifests/joint-openml12-squared.json
bank=/home/cms/experiments/openml12-frozen-icl-20260927
expected_sha=$(basename "$checkpoint" .pt)
cd "$base/source/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
command=(~/.local/bin/wehub-python --profile train-20260920 -u -c
  "import runpy; runpy.run_path('$base/evaluate.py', run_name='__main__')"
  --manifest "$manifest" --checkpoint "$checkpoint" --expected-sha "$expected_sha"
  --device cuda:0 --bank-root "$bank" --output "$base/evaluations/$label")
if [ "$#" -gt 0 ]; then
  command+=(--only "$@")
fi
"${command[@]}"
