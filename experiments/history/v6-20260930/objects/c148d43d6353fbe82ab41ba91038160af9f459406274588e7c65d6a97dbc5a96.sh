#!/usr/bin/env bash
set -euo pipefail

root=/home/cms/experiments/v6-sparse-relevance-600-20260927
source_root=/home/cms/experiments/v6-target-broadcast-dgx2-20260927/source/src
cd "$source_root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
for spec in 'v6 full32 250' 'v6 signal6 300'; do
  read -r model arm update <<< "$spec"
  ~/.local/bin/wehub-python --profile train-20260920 -u -c "import runpy; runpy.run_path('$root/eval_saved.py', run_name='__main__')" \
    --root "$root" --run "$root/run-h8-600" \
    --model "$model" --arm "$arm" --update "$update" --device cuda:0
done
