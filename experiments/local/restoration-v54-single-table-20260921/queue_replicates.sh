#!/bin/sh
# Start r1/r2 only after the previous replicate completed successfully.
# A failed or wall-budget-exhausted replicate stops this queue without retrying.
set -eu
ROOT=$1
TABLE=$2
DEVICE=$3
cd "$ROOT/src"
for rep in 1 2; do
    previous=$((rep - 1))
    previous_terminal="../runs/${TABLE}.r${previous}/terminal.json"
    while [ ! -f "$previous_terminal" ]; do
        sleep 30
    done
    if ! grep -q '"outcome": "completed"' "$previous_terminal"; then
        echo "queue stopped: ${TABLE}.r${previous} did not complete" >&2
        exit 7
    fi
    manifest="../experiments/local/restoration-v54-single-table-20260921/manifests/${TABLE}.r${rep}.json"
    output="../runs/${TABLE}.r${rep}"
    log="../receipts/${TABLE}.r${rep}.stdout.log"
    if [ -e "$output" ]; then
        echo "queue stopped: output already exists: $output" >&2
        exit 8
    fi
    ~/.local/bin/wehub-python --profile train -u -m tabu_lab.cli curriculum-v54 run \
        --manifest "$manifest" --output-root "$output" --device "$DEVICE" \
        >"$log" 2>&1
 done
