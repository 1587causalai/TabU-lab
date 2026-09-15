"""Bounded, preregistered comparison against a hashed frozen serial oracle."""

from __future__ import annotations

import json
import platform
import runpy
import statistics
import time
from pathlib import Path

from .tar_fit import gpu_preflight, sha


def run_benchmark(args):
    spec = json.loads(Path(args.preregistration).read_text())
    if spec.get("schema") != "tabu.tar.batched-benchmark.1":
        raise ValueError("unsupported benchmark preregistration")
    if sha(args.reference) != spec["reference_sha256"]:
        raise ValueError("reference source digest mismatch")
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="local_unissued",
        outcome="started",
        device=args.device,
        preregistration_sha256=sha(args.preregistration),
        reference_sha256=sha(args.reference),
        runner_sha256=sha(__file__),
    )
    (root / "resolved.json").write_text(
        json.dumps(dict(spec=spec, request=receipt), indent=2) + "\n"
    )
    try:
        if args.device == "cuda:0":
            receipt["preflight"] = gpu_preflight()
            if not receipt["preflight"]["ready"]:
                receipt["outcome"] = "blocked_resources"
                return receipt
        import torch

        from tabu_lab.models.tar import TabUTARModel, TARConfig, TARFeature, score
        from tabu_lab.models.tar.checkpoint import source_digest
        from tabu_lab.models.tar.episodes import sample_supervised_episode

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = TARConfig(**spec["configuration"])
        new = TabUTARModel(cfg, device=args.device)
        old = TabUTARModel(cfg, device=args.device)
        reference = runpy.run_path(str(args.reference))["SerialAxisBlock"]
        old.layers = torch.nn.ModuleList(
            [reference(cfg, device=args.device, dtype=torch.float32) for _ in range(cfg.blocks)]
        )
        old.load_state_dict(new.state_dict())
        receipt.update(
            config=cfg.as_dict(),
            parameter_count=cfg.parameter_count,
            model_source_sha256=source_digest(),
            environment=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                cuda=torch.version.cuda,
                architecture=platform.machine(),
            ),
            datasets={},
        )

        def sync():
            if args.device == "cuda:0":
                torch.cuda.synchronize()

        for name, digest in spec["datasets"].items():
            data_path = Path(args.preregistration).parent / "data" / f"{name}.json"
            if sha(data_path) != digest:
                raise ValueError("dataset digest mismatch")
            data = json.loads(data_path.read_text())
            values = torch.tensor(data["values"], dtype=torch.float64)
            ids = data["splits"]["context"] + data["splits"]["fit"]
            features = (
                *(TARFeature(column_id=i) for i in range(values.shape[1] - 1)),
                TARFeature(data["target_kind"], tuple(data["domain"]), values.shape[1] - 1),
            )
            ep, truth, provenance = sample_supervised_episode(
                values[ids],
                features,
                row_ids=ids,
                context_size=len(data["splits"]["context"]),
                seed=20260906,
                namespace=f"{name}/training",
                episode_id=0,
            )

            def step(model, ep=ep, truth=truth):
                model.zero_grad(set_to_none=True)
                output = model(ep)
                loss = score(output, truth)
                loss.backward()
                return output, loss

            actual, loss = step(new)
            expected, old_loss = step(old)
            tolerance = spec["tolerance"]
            torch.testing.assert_close(actual.carriers, expected.carriers, **tolerance)
            torch.testing.assert_close(actual.responses, expected.responses, **tolerance)
            torch.testing.assert_close(loss, old_loss, **tolerance)
            max_gradient_error = 0.0
            for a, b in zip(actual.predictions, expected.predictions, strict=True):
                torch.testing.assert_close(a.value, b.value, **tolerance)
                if a.probabilities is not None:
                    torch.testing.assert_close(a.probabilities, b.probabilities, **tolerance)
            for a, b in zip(new.parameters(), old.parameters(), strict=True):
                ga = a.grad if a.grad is not None else torch.zeros_like(a)
                gb = b.grad if b.grad is not None else torch.zeros_like(b)
                torch.testing.assert_close(ga, gb, **spec["gradient_tolerance"])
                max_gradient_error = max(max_gradient_error, float((ga - gb).abs().max()))
            row = dict(
                episode=provenance,
                equivalence="passed",
                loss=float(loss.detach()),
                max_carrier_abs_error=float((actual.carriers - expected.carriers).abs().max()),
                max_gradient_abs_error=max_gradient_error,
            )
            del actual, expected, loss, old_loss
            for label, model in [("serial", old), ("batched", new)]:
                for _ in range(spec["warmup_steps"]):
                    step(model)
                timings = []
                model.zero_grad(set_to_none=True)
                sync()
                if args.device == "cuda:0":
                    torch.cuda.reset_peak_memory_stats()
                for _ in range(spec["timed_steps"]):
                    sync()
                    start = time.perf_counter()
                    step(model)
                    sync()
                    timings.append(time.perf_counter() - start)
                measurement = dict(seconds=timings, median_seconds=statistics.median(timings))
                if args.device == "cuda:0":
                    measurement["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                if spec["profile"]:
                    activities = [torch.profiler.ProfilerActivity.CPU]
                    if args.device == "cuda:0":
                        activities.append(torch.profiler.ProfilerActivity.CUDA)
                    with torch.profiler.profile(activities=activities) as prof:
                        step(model)
                        sync()
                    trace = root / f"{name}-{label}-trace.json"
                    prof.export_chrome_trace(str(trace))
                    events = json.loads(trace.read_text())["traceEvents"]
                    kernels = [x for x in events if x.get("cat") == "kernel"]
                    measurement["kernel_count"] = len(kernels)
                    measurement["kernel_duration_us"] = sum(x.get("dur", 0) for x in kernels)
                    measurement["cpu_operator_count"] = sum(
                        x.get("cat") == "cpu_op" for x in events
                    )
                row[label] = measurement
            row["speedup"] = row["serial"]["median_seconds"] / row["batched"]["median_seconds"]
            receipt["datasets"][name] = row
            print(json.dumps(dict(dataset=name, **row)), flush=True)
        receipt["outcome"] = "benchmark_completed"
        return receipt
    except BaseException as exc:
        receipt.update(outcome="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        (root / "result.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        (root / "checksums.json").write_text(
            json.dumps(
                {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()},
                indent=2,
            )
            + "\n"
        )
