#!/bin/sh
# Advance independent r1/r2 only after the preceding replicate completed.
set -eu

ROOT=$1
TABLE=$2
DEVICE=$3
PANEL=../experiments/local/restoration-v54-single-table-20260921-bd9a42d

cd "$ROOT/src"
for rep in 1 2; do
    previous=$((rep - 1))
    previous_terminal="../runs-bd9a42d/${TABLE}.r${previous}/terminal.json"
    while [ ! -f "$previous_terminal" ]; do
        sleep 30
    done
    if ! grep -q '"outcome": "completed"' "$previous_terminal"; then
        echo "queue stopped: ${TABLE}.r${previous} did not complete" >&2
        exit 7
    fi
    manifest="$PANEL/manifests/${TABLE}.r${rep}.json"
    output="../runs-bd9a42d/${TABLE}.r${rep}"
    log="../receipts-bd9a42d/${TABLE}.r${rep}.stdout.log"
    if [ -e "$output" ]; then
        echo "queue stopped: output already exists: $output" >&2
        exit 8
    fi
    if [ "$DEVICE" = mps ]; then
        export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0
    fi
    ~/.local/bin/wehub-python --profile train -u -m tabu_lab.cli curriculum-v54 run \
        --manifest "$manifest" --output-root "$output" --device "$DEVICE" \
        >"$log" 2>&1
done
