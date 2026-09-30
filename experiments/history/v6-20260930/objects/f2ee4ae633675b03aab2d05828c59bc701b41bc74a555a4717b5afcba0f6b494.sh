#!/usr/bin/env bash
set -euo pipefail

root=/home/cms/experiments/v6-target-broadcast-dgx2-20260927
bank=/home/cms/experiments/openml12-frozen-icl-20260927
manifest=/home/cms/experiments/openml12-joint2h-squared-20260927/manifests/joint-openml12-squared.json
old_manifest=/home/cms/experiments/joint618-60min-20260926/manifests/joint618.json
old_bank=/home/cms/experiments/openml12-joint30m-squared-20260927/fixed-query-bank.json
end_digest=d27940148168ccca51ab1f1a5881a122eed0125ed14e78e6a49a60eb7d67755a
mid_digest=1d0a7734647b9b0742a3f07ddc2eaef616b6935462f37486636b5cc252fa53f3
cd "$root/source/src"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$root/evaluate_v6.py', run_name='__main__')" \
  --manifest "$manifest" \
  --checkpoint "$root/stage15m-v6/checkpoints/$mid_digest.pt" \
  --expected-sha "$mid_digest" \
  --device cuda:0 --bank-root "$bank" --output "$root/eval-mid488-puma" --only pumadyn32nh

~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$root/evaluate_v6.py', run_name='__main__')" \
  --manifest "$manifest" \
  --checkpoint "$root/stage15m-v6/checkpoints/$end_digest.pt" \
  --expected-sha "$end_digest" \
  --device cuda:0 --bank-root "$bank" --output "$root/eval-end-openml12"

for scope in smoke full; do
  extra=()
  if [[ "$scope" == smoke ]]; then extra=(--smoke); fi
  ~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$root/evaluate_old618_v6.py', run_name='__main__')" \
    --host dgx2 --label "end-$scope" \
    --current-manifest "$manifest" --old-manifest "$old_manifest" \
    --checkpoint "$root/stage15m-v6/checkpoints/$end_digest.pt" \
    --expected-sha "$end_digest" --bank "$old_bank" \
    --device cuda:0 --output "$root/eval-end-old618-$scope" "${extra[@]}"
done
