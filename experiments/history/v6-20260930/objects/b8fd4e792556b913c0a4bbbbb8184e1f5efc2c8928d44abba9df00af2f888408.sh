#!/usr/bin/env bash
set -euo pipefail

base=/home/cms/experiments/v6-broadcast-oldloss-openml12-20260927
run=$base/stage1h/run
python3 - "$run/campaign.json" <<'PY'
import json, sys
campaign = json.load(open(sys.argv[1]))
if campaign["outcome"] != "training_completed" or campaign["successful_update_seconds"] < 3600:
    raise SystemExit("training has not reached its terminal one-hour budget")
PY

initial_sha=$(python3 - "$run/campaign.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["initial_checkpoint_sha256"])
PY
)
terminal_sha=$(python3 - "$run/campaign.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["checkpoint_sha256"])
PY
)

mkdir -p "$base/evaluations"
eval_one() {
  local label=$1 sha=$2
  if [ -f "$base/evaluations/$label/terminal.json" ]; then
    echo "already evaluated: $label"
  else
    bash "$base/eval_checkpoint.sh" "$run/checkpoints/$sha.pt" "$label"
  fi
}

eval_one initial "$initial_sha"
eval_one u0900 e8f0483c56bb7b51b4ae5e52f38e5af19f405aa2fda73d795cfee6541966af02
eval_one u1800 bd705a79f4684de931e3268339ed710fe6d2816b11e58d15ddfe45b40e0dbaf7
eval_one terminal "$terminal_sha"
