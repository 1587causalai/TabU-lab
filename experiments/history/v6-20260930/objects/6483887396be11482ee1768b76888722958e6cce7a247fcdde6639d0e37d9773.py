"""Freeze the versioned Anchor120 recipe; preserve failures and rejections."""

# Derived from TabU-lab (Apache-2.0); new source snapshot and manifest identity.
import json
import platform
import shutil
from collections import Counter
from pathlib import Path
import numpy as np
from ..api import sample_request
from ..recipes.anchor120 import corpus_requests
from ..io.json import canonical_hash, file_hash, write_json
from .adapt import select_target, adapt_table, realized_metadata


def generate_corpus(recipe, *, seed=20260907, n_rows=256, output):
    if recipe not in ("anchor120", "anchor120.1"):
        raise ValueError("unknown recipe; use anchor120.1")
    if type(n_rows) is not int or n_rows < 16:
        raise ValueError("Anchor120 requires n_rows >= 16")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 120000:
        raise ValueError("recipe seed is outside the supported range")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / "generator-source"
    source_root = Path(__file__).resolve().parents[1]
    source_files = {}
    for path in sorted(source_root.rglob("*")):
        if (
            path.is_file()
            and (path.suffix == ".py" or "resources" in path.parts)
            and "__pycache__" not in path.parts
        ):
            name = str(path.relative_to(source_root))
            dest = frozen / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
            source_files[name] = file_hash(dest)
    import sklearn

    (root / "data").mkdir()
    (root / "raw").mkdir()
    entries = corpus_requests(n_rows, seed)
    write_json(root / "requested.json", entries)
    manifest = dict(
        schema="tfm-data.corpus.1",
        recipe="anchor120.1",
        package=dict(name="tfm-data", version="0.1.0"),
        status="local_unissued",
        generator_files=source_files,
        runtime=dict(
            python=platform.python_version(),
            numpy=np.__version__,
            sklearn=sklearn.__version__,
        ),
        rows=n_rows,
        split_seed=seed,
        records=[],
    )
    seen_tables = set()
    rejected = []
    try:
        for entry in entries:
            accepted = None
            for attempt in range(100):
                request = {
                    **entry["request"],
                    "seed": entry["request"]["seed"] + attempt,
                }
                try:
                    episode = sample_request(**request)
                    target = select_target(episode, entry)
                    dataset = adapt_table(
                        episode, target, name=entry["dataset"], split_seed=seed
                    )
                    digest = canonical_hash(dataset["values"])
                    if digest in seen_tables:
                        raise ValueError("duplicate complete table")
                    # Byte-exact value/metadata replay under the frozen implementation.
                    replay = sample_request(**request)
                    if canonical_hash(replay) != canonical_hash(episode):
                        raise RuntimeError("episode replay mismatch")
                    accepted = (episode, dataset, target, digest)
                    break
                except ValueError as exc:
                    rejection = dict(
                        dataset=entry["dataset"],
                        attempt=attempt,
                        seed=request["seed"],
                        reason=str(exc),
                    )
                    rejected.append(rejection)
                    with (root / "rejections.jsonl").open("a") as stream:
                        stream.write(json.dumps(rejection) + "\n")
            if accepted is None:
                raise ValueError(f"stratum exhausted 100 attempts: {entry['dataset']}")
            episode, dataset, target, digest = accepted
            seen_tables.add(digest)
            name = entry["dataset"]
            write_json(root / "raw" / f"{name}.json", episode)
            write_json(root / "data" / f"{name}.json", dataset)
            record = dict(
                dataset=name,
                family=entry["family"],
                profile=entry["profile"],
                target_type=entry["target_type"],
                request=request,
                accepted_attempt=attempt,
                data_sha256=file_hash(root / "data" / f"{name}.json"),
                raw_sha256=file_hash(root / "raw" / f"{name}.json"),
                values_sha256=digest,
                replay_verified=True,
                realized=realized_metadata(episode, target, dataset, entry),
            )
            manifest["records"].append(record)
            print(
                json.dumps(
                    dict(
                        dataset=name,
                        accepted_attempt=attempt,
                        target=record["realized"]["target_source_type"],
                    )
                ),
                flush=True,
            )
        manifest.update(
            outcome="corpus_frozen",
            count=len(manifest["records"]),
            family_counts=dict(Counter(r["family"] for r in manifest["records"])),
            target_counts=dict(Counter(r["target_type"] for r in manifest["records"])),
            rejection_count=len(rejected),
        )
    except Exception as exc:
        manifest.update(outcome="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest["auxiliary_files"] = {
            p.name: file_hash(p)
            for p in (root / "requested.json", root / "rejections.jsonl")
            if p.exists()
        }
        write_json(root / "manifest.json", manifest)
    return manifest
