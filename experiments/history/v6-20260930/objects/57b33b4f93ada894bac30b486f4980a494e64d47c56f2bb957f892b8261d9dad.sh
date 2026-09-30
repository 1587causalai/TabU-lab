#!/usr/bin/env bash
set -euo pipefail

root=/home/windb/experiments/puma-synthetic-diagnosis-20260927
base=/home/windb/experiments/puma-synthetic-probe-20260927
cd "$root"
export PYTHONPATH="$base/source/src"

cleanup() {
  status=$?
  trap - EXIT
  if [[ -f "$root/miner-guard.pid" ]]; then
    kill "$(cat "$root/miner-guard.pid")" 2>/dev/null || true
  fi
  nohup bash /home/windb/.srb/start_dt_v2.sh >/dev/null 2>&1 < /dev/null &
  printf 'exit=%s\n' "$status" > "$root/readonly.status"
  echo "READONLY_EXIT status=$status"
  exit "$status"
}
trap cleanup EXIT

parent="$base/inputs/e3da2393c28f6f05cab7e9d143710e69db9618015205b47f1fd3e9a041543863.pt"
signal600=$(readlink -f "$base/runs/signal6/resume-150-to-600/checkpoint-progress.pt")
full600=$(readlink -f "$base/runs/full32/resume-150-to-600/checkpoint-progress.pt")
full1000=$(readlink -f "$base/runs/full32/resume-600-to-1000/checkpoint-progress.pt")

for arm in signal6 full32; do
  echo "BEGIN stage=$arm-parent $(date -Is)"
  ~/.local/bin/wehub-python --profile train -u stage_probe.py --arm "$arm" --checkpoint "$parent" --label parent
  echo "DONE stage=$arm-parent $(date -Is)"
done
for entry in "signal6:u600:$signal600" "full32:u600:$full600" "full32:u1000:$full1000"; do
  IFS=: read -r arm label checkpoint <<< "$entry"
  echo "BEGIN stage=$arm-$label $(date -Is)"
  ~/.local/bin/wehub-python --profile train -u stage_probe.py --arm "$arm" --checkpoint "$checkpoint" --label "$label"
  echo "DONE stage=$arm-$label $(date -Is)"
done

for entry in "full32-u600:relevant6:$full600" "full32-u600:first6:$full600" \
             "signal6-u600:relevant6:$signal600" "full32-u1000:relevant6:$full1000"; do
  IFS=: read -r name keep checkpoint <<< "$entry"
  echo "BEGIN mask=$name-$keep $(date -Is)"
  ~/.local/bin/wehub-python --profile train -u mask_probe.py --weights-name "$name" --keep "$keep" --checkpoint "$checkpoint"
  echo "DONE mask=$name-$keep $(date -Is)"
done

for entry in "u600:$full600" "u1000:$full1000"; do
  IFS=: read -r label checkpoint <<< "$entry"
  echo "BEGIN bandwidth=$label $(date -Is)"
  ~/.local/bin/wehub-python --profile train -u bandwidth_probe.py --arm full32 --checkpoint "$checkpoint" --bandwidths 0.5 1.0 2.0
  echo "DONE bandwidth=$label $(date -Is)"
done
