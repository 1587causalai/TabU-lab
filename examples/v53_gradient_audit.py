"""Matched CPU audit of input geometry; train-row probes, no held-out claims."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from tabu_lab.curriculum_v53.data import load_table
from tabu_lab.curriculum_v53.evaluation import evaluate_probe
from tabu_lab.curriculum_v53.runner import train_step
from tabu_lab.models.restoration_v53 import V53Config, V53LossConfig, V53Model
from tabu_lab.restoration_optimizers import OptimizerConfig, adamw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=32)
    parser.add_argument("--probe-masks", type=int, default=4)
    args = parser.parse_args()
    if args.updates < 1 or args.probe_masks < 1:
        parser.error("updates and probe-masks must be positive")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = Path(__file__).resolve().parents[1] / "src/tabu_lab"
    report = {
        "scope": __doc__, "device": "cpu", "dtype": "float64", "torch": str(torch.__version__),
        "updates_per_variant": args.updates, "probe_masks": args.probe_masks,
        "source_sha256": {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(source.rglob("*.py"))
                          if "restoration" in str(p) or "curriculum_v53" in str(p)},
        "variants": [],
    }
    for path in args.manifest:
        spec = yaml.safe_load(path.read_text())
        table = load_table(spec["tables"][0], args.data_root)
        stage = spec["stages"][0]
        probe = dict(spec["probes"][0], masks=args.probe_masks)
        if probe["partition"] != "train":
            raise ValueError("this audit accepts train-row probes only")
        shared_parameters_hash = None
        for name, projection, tau in (
            ("historical", "legacy_scaled", 1.),
            ("small_tau_only", "legacy_scaled", 1e-6),
            ("isometry_only", "isometric_qr", 1.),
            ("new_default", "isometric_qr", 1e-6),
        ):
            config = V53Config.from_dict(dict(
                spec["model"], input_projection=projection,
                backbone=dict(spec["model"]["backbone"], tau_presence=tau),
            ))
            plan = SimpleNamespace(config=config, optimizer=OptimizerConfig(**spec["optimizer"]),
                                   spec=spec, tables=(table,))
            torch.manual_seed(spec["seeds"]["model"])
            model = V53Model(config).double()
            digest = hashlib.sha256(b"".join(
                p.detach().numpy().tobytes() for n, p in model.named_parameters()
                if not n.startswith("encoder.projection.")
            )).hexdigest()
            if shared_parameters_hash is None:
                shared_parameters_hash = digest
            assert shared_parameters_hash == digest, "unmatched initialization outside W_enc"
            optimizer = adamw(model, plan.optimizer)
            before = evaluate_probe(model, plan, probe, "cpu")
            rows = []
            for index in range(args.updates):
                row = train_step(model, optimizer, plan, table, stage["recipe"][table.kind], index,
                                 "cpu", V53LossConfig(**stage["loss"]), namespace=stage["name"])
                rows.append({k: row[k] for k in (
                    "loss", "gradient_norm", "clipped_gradient_norm", "episode")})
            after = evaluate_probe(model, plan, probe, "cpu")
            norms = [r["gradient_norm"] for r in rows]
            factors = [r["clipped_gradient_norm"] / r["gradient_norm"] for r in rows]
            item = {
                "table": table.name, "variant": name, "config": config.as_dict(),
                "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "initial_nonprojection_parameters_sha256": digest,
                "grad_norm_median": statistics.median(norms), "grad_norm_max": max(norms),
                "clipped_fraction": sum(n > plan.optimizer.grad_clip for n in norms) / len(norms),
                "clip_factor_median": statistics.median(factors),
                "evaluation_before": before, "evaluation_after": after, "updates": rows,
            }
            report["variants"].append(item)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(json.dumps({k: item[k] for k in (
                "table", "variant", "grad_norm_median", "grad_norm_max", "clipped_fraction",
                "clip_factor_median")}), flush=True)


if __name__ == "__main__":
    main()
