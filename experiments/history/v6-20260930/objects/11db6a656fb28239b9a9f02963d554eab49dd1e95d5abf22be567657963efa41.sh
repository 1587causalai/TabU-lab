#!/usr/bin/env bash
set -euo pipefail

root=/home/cms/experiments/v6-target-broadcast-dgx2-20260927
bank=/home/cms/experiments/openml12-frozen-icl-20260927
manifest=/home/cms/experiments/openml12-joint2h-squared-20260927/manifests/joint-openml12-squared.json
cd "$root/source/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for label in start end; do
  if [[ "$label" == start ]]; then
    digest=8a48ea48f9774518be1c71ec0d5b2b700a53373d2f093de2b8d48ea3f15d33f0
  else
    digest=d27940148168ccca51ab1f1a5881a122eed0125ed14e78e6a49a60eb7d67755a
  fi
  ~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$root/evaluate_v6.py', run_name='__main__')" \
    --manifest "$manifest" \
    --checkpoint "$root/stage15m-v6/checkpoints/$digest.pt" \
    --expected-sha "$digest" \
    --device cuda:0 \
    --bank-root "$bank" \
    --output "$root/eval-$label-puma" \
    --only pumadyn32nh
done
