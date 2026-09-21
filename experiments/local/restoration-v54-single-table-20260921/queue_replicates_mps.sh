#!/bin/sh
# MPS/FP32 r1/r2 queue. The earlier CPU queue is a superseded invalid launch.
set -eu
ROOT=$1
TABLE=$2
cd "$ROOT/src"
for rep in 1 2; do
    previous=$((rep - 1))
    previous_terminal="../runs-mps/${TABLE}.r${previous}/terminal.json"
    while [ ! -f "$previous_terminal" ]; do
        sleep 30
    done
    if ! grep -q '"outcome": "completed"' "$previous_terminal"; then
        echo "queue stopped: ${TABLE}.r${previous} did not complete" >&2
        exit 7
    fi
    manifest="../experiments/local/restoration-v54-single-table-20260921/manifests/${TABLE}.r${rep}.json"
    output="../runs-mps/${TABLE}.r${rep}"
    log="../receipts-mps/${TABLE}.r${rep}.stdout.log"
    if [ -e "$output" ]; then
        echo "queue stopped: output already exists: $output" >&2
        exit 8
    fi
    export PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0
    ~/.local/bin/wehub-python --profile train -u -m tabu_lab.cli curriculum-v54 run \
        --manifest "$manifest" --output-root "$output" --device mps \
        >"$log" 2>&1
done
