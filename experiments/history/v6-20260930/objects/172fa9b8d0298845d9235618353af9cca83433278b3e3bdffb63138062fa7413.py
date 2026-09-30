"""A second, independent Puma-like sparse-relevance teacher and row bank.

The signal-to-noise recipe and evaluation split match the original paired
probe.  Only the generator seed and six relevant column positions change.
This script prepares immutable data; it does not train or evaluate a model.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SEED = 2026092702
ROWS = 8192
TRAIN = 6553
RELEVANT = (2, 9, 11, 14, 16, 27)
ARMS = ("signal6", "full32")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        output.write("\n")


def raw(j: int, u: float) -> float:
    return 2.356 * u if j < 12 else 75.0 * u if j < 17 else 1.375 + 1.125 * u


def main() -> None:
    rng = random.Random(SEED)
    unit = [[rng.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(ROWS)]
    noise = [rng.gauss(0.0, 1.0) for _ in range(ROWS)]

    def teacher(u: list[float]) -> float:
        return (1.2 * (u[14] > 0) * math.sin(math.pi * u[2])
                + 0.8 * (u[27] > 0) * u[11]
                + 0.6 * u[9] * u[16])

    signal = [teacher(u) for u in unit]
    mean = statistics.fmean(signal[:TRAIN])
    scale = statistics.pstdev(signal[:TRAIN])
    assert scale > 0
    target = [0.030 * (g - mean) / scale + 0.020 * e
              for g, e in zip(signal, noise, strict=True)]
    names = ([f"theta{i+1}" for i in range(6)]
             + [f"thetad{i+1}" for i in range(6)]
             + [f"tau{i+1}" for i in range(5)]
             + [f"mass_or_damping{i+1}" for i in range(15)])
    split = dict(train=list(range(TRAIN)), test=list(range(TRAIN, ROWS)))
    paths = {}
    for arm in ARMS:
        columns = RELEVANT if arm == "signal6" else tuple(range(32))
        data = dict(
            schema="puma-synthetic-probe-v1",
            generator_seed=SEED,
            rows=ROWS,
            relevant_columns=list(RELEVANT),
            target_formula="0.030*z(g_train)+0.020*normal(0,1)",
            values=[[*(raw(j, unit[i][j]) for j in columns), target[i]]
                    for i in range(ROWS)],
            features=[dict(key=names[j], kind="numeric", domain=[]) for j in columns]
                     + [dict(key="acceleration_target", kind="numeric", domain=[])],
            splits=split,
        )
        path = ROOT / "data" / f"{arm}.json"
        write_new(path, data)
        paths[arm] = dict(path=f"data/{arm}.json", sha256=sha256(path),
                          feature_columns=len(columns), target_column=len(columns))
    receipt = dict(
        schema="tabu.sparse-relevance.seed2-data.v1",
        generator_seed=SEED,
        rows=ROWS,
        train_rows=TRAIN,
        test_rows=ROWS - TRAIN,
        relevant_columns=list(RELEVANT),
        teacher="1.2*I(u14>0)*sin(pi*u2) + 0.8*I(u27>0)*u11 + 0.6*u9*u16",
        clean_signal_train_sd=0.030,
        noise_sd_nominal=0.020,
        signal_to_noise_sd_ratio_nominal=1.5,
        target_train_sd=statistics.pstdev(target[:TRAIN]),
        data=paths,
    )
    write_new(ROOT / "data-receipt.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
