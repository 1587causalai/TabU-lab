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
  printf 'exit=%s\n' "$status" > "$root/remaining.status"
  echo "REMAINING_EXIT status=$status"
  exit "$status"
}
trap cleanup EXIT

for arm in dense32 linear32; do
  echo "BEGIN arm=$arm $(date -Is)"
  ~/.local/bin/wehub-python --profile train -u diagnosis.py fit --arm "$arm" --updates 150
  checkpoint=$(readlink -f "$root/runs/$arm/updates-150/checkpoint-progress.pt")
  test -f "$checkpoint"
  ~/.local/bin/wehub-python --profile train -u diagnosis.py evaluate --arm "$arm" --checkpoint "$checkpoint" --readout
  echo "DONE arm=$arm $(date -Is)"
done

echo "BEGIN arm=full32_extension $(date -Is)"
~/.local/bin/wehub-python --profile train -u diagnosis.py resume-full32 --updates 400
checkpoint=$(readlink -f "$base/runs/full32/resume-600-to-1000/checkpoint-progress.pt")
test -f "$checkpoint"
~/.local/bin/wehub-python --profile train -u diagnosis.py evaluate --arm full32 --checkpoint "$checkpoint" --readout
echo "DONE arm=full32_extension $(date -Is)"
