"""Freeze a mechanism-stratified corpus from the local episode API implementation."""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import platform
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from tabu_lab.tar_data import dataset_features, full_train_test_split, validate_full_dataset

KINDS = ("numeric", "ordinal", "binary", "categorical", "high_cardinality")
SCM_PROFILES = {
    "numeric_predictors": (100, 0, 0, 0, 0),
    "mixed": (55, 15, 15, 10, 5),
    "discrete_rich": (20, 20, 20, 25, 15),
}
DISCO_PROFILES = {
    "numeric_dominant": (80, 7, 8, 5, 0),
    "mixed": (55, 15, 15, 10, 5),
    "discrete_rich": (20, 20, 20, 25, 15),
}


def canonical_hash(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def corpus_requests(rows=256, seed=20260907):
    """120 strata: requested axes are not substitutes for realized-world audits."""
    entries = []
    targets = ("numeric", "binary", "ordinal", "categorical")
    for ti, pi, capi, ni in itertools.product(range(4), range(3), range(2), range(2)):
        profile = list(SCM_PROFILES)[pi]
        request = dict(
            source="scm",
            api_version="v1",
            n_features=(8, 12, 20)[(ti + pi + capi + ni) % 3],
            sigma=(0.05, 0.3)[ni],
            scm_options=dict(
                target_type=targets[ti],
                max_parents=(1, 4)[capi],
                type_weights=dict(zip(KINDS, SCM_PROFILES[profile], strict=True)),
            ),
        )
        entries.append(
            dict(family="scm_mixed_v1", profile=profile, target_type=targets[ti], request=request)
        )
    for pi, gi, ki, ni, ti in itertools.product(range(3), range(2), range(2), range(2), range(2)):
        profile = list(DISCO_PROFILES)[pi]
        target = (
            "numeric" if ti == 0 else ("binary", "ordinal", "categorical")[(pi + gi + ki + ni) % 3]
        )
        request = dict(
            source="discoscm",
            n_features=(12, 20, 32)[(pi + ki + ni + ti) % 3],
            unit_dim=(4, 16)[ki],
            sigma=(0.05, 0.3)[ni],
            graph_family=("sparse", "star")[gi],
            max_parents=4 if gi == 0 else None,
            token_heritability=(0.5, 0.9)[(pi + gi + ki + ni) % 2],
            independent_frac=0.1 if pi == 2 else 0.0,
            enable_heavy_tails=(pi == 2 and ti == 0),
            type_weights=dict(zip(KINDS, DISCO_PROFILES[profile], strict=True)),
        )
        entries.append(
            dict(family="discoscm", profile=profile, target_type=target, request=request)
        )
    for target_fn, width, noise in itertools.product(
        ("root_noise", "linear", "tanh"), (6, 20), (0.1, 0.3)
    ):
        entries.append(
            dict(
                family="scm_numeric_v0",
                profile=target_fn,
                target_type="numeric",
                target_function=target_fn,
                request=dict(source="scm", api_version="v0", n_features=width, sigma=noise),
            )
        )
    for source, width in itertools.product(
        (
            "sklearn_make_regression",
            "sklearn_make_classification",
            "sklearn_friedman1",
            "sklearn_low_rank",
        ),
        (6, 12, 20),
    ):
        entries.append(
            dict(
                family="sklearn_synthetic",
                profile=source,
                target_type="binary" if source == "sklearn_make_classification" else "numeric",
                request=dict(source=source, n_features=width),
            )
        )
    for index, entry in enumerate(entries):
        entry["dataset"] = f"{entry['family']}_{index:03d}"
        entry["request"].update(
            n_units=rows,
            n_episodes=1,
            batch_size=1,
            missing_frac=0.0,
            query_frac=0.0,
            query_mode="label_cell",
            return_mechanism=True,
            seed=seed + index * 1000,
        )
    return entries


def select_target(episode, entry):
    table = episode["table"]
    if entry["family"] == "discoscm":
        law = episode["response_law"]
        if law["graph_family"] != entry["request"]["graph_family"]:
            raise ValueError("requested graph family did not materialize")
        candidates = [
            i
            for i, kind in enumerate(table["column_types"])
            if kind == entry["target_type"] and not law["columns"][i]["independent"]
        ]
        if not candidates:
            raise ValueError("requested non-independent target type absent")
        # Choose by schema only, never target values or model fit.
        return candidates[0]
    if (
        entry["family"] == "scm_numeric_v0"
        and episode["mechanism"]["node_fn"][-1] != entry["target_function"]
    ):
        raise ValueError("requested target function did not materialize")
    return len(table["column_types"]) - 1


def adapt_table(episode, target, *, name, split_seed):
    table = episode["table"]
    width = len(table["column_types"])
    order = [c for c in range(width) if c != target] + [target]
    values = [[row[c] for c in order] for row in table["values"]]
    features = []
    for col in order:
        kind = table["column_types"][col]
        features.append(
            dict(
                kind=kind if kind in ("numeric", "ordinal") else "nominal",
                domain=[]
                if kind == "numeric"
                else [str(i) for i in range(int(table["n_classes"][col]))],
            )
        )
    dataset = dict(
        schema="tabu.tar.typed-fit-table.1",
        dataset=name,
        values=values,
        features=features,
        target_kind=features[-1]["kind"],
        domain=features[-1]["domain"],
        output_to_source_column=order,
        splits=full_train_test_split(
            values, seed=split_seed, stratified=features[-1]["kind"] != "numeric"
        ),
    )
    dataset_features(dataset)
    validate_full_dataset(dataset, len(values))
    y = np.asarray([values[i][-1] for i in dataset["splits"]["train"]])
    if np.unique(y).size < 2 or (features[-1]["kind"] == "numeric" and float(y.var()) < 1e-10):
        raise ValueError("training target is degenerate")
    return dataset


def realized_metadata(episode, target, dataset, entry):
    table = episode["table"]
    rows = np.asarray(dataset["values"], dtype=np.float64)[dataset["splits"]["train"]]
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[:-1]), []).append(float(row[-1]))
    result = dict(
        target_source_column=target,
        target_source_type=table["column_types"][target],
        columns=len(table["column_types"]),
        source_type_counts=dict(Counter(table["column_types"])),
        predictor_type_counts=dict(
            Counter(t for i, t in enumerate(table["column_types"]) if i != target)
        ),
        declared_target_classes=table["n_classes"][target],
        observed_train_target_values=int(np.unique(rows[:, -1]).size),
        train_target_variance=float(rows[:, -1].var()),
        train_target_frequencies=None
        if dataset["target_kind"] == "numeric"
        else dict(Counter(str(int(value)) for value in rows[:, -1])),
        unique_train_predictors=len(groups),
        duplicate_predictor_groups=sum(len(v) > 1 for v in groups.values()),
        conflicting_predictor_groups=sum(len(set(v)) > 1 for v in groups.values()),
        max_abs_train_value=float(np.abs(rows).max()),
    )
    if entry["family"] == "scm_mixed_v1":
        mechanism = episode["mechanism"]
        target_column = mechanism["columns"][target]
        result.update(
            world_hash=mechanism["world_hash"],
            edge_count=len(mechanism["graph"]["edges"]),
            target_parent_count=len(target_column["parents"]),
            target_parent_types=[e["parent_type"] for e in target_column["mechanism"]["edges"]],
            edge_transforms=dict(
                Counter(
                    e["transform"]
                    for col in mechanism["columns"]
                    for e in col["mechanism"]["edges"]
                )
            ),
            replay_kind="manifest_world_and_event_seed",
        )
    elif entry["family"] == "discoscm":
        law, population = episode["response_law"], episode["population"]
        result.update(
            graph_family=law["graph_family"],
            edge_count=sum(len(p) for p in law["parents"]),
            target_parent_count=len(law["parents"][target]),
            target_independent=law["columns"][target]["independent"],
            independent_columns=sum(c["independent"] for c in law["columns"]),
            unit_dim=table["unit_dim"],
            mixture=population["mixture"],
            token_activations=dict(Counter(law["activations"])),
            noise_families=dict(Counter(law["noise_families"])),
            target_response=law["columns"][target]["g"],
            replay_kind="pinned_request_and_runtime",
        )
    else:
        result.update(mechanism=episode["mechanism"], replay_kind="pinned_request_and_runtime")
    return result


def freeze_corpus(args):
    root, generator_root = Path(args.output_root), Path(args.generator_root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root / "generator-source"
    frozen.mkdir()
    source_files = ("generator.py", "scm_v1.py", "sources.py", "LICENSE", "requirements.txt")
    for name in source_files:
        shutil.copy2(generator_root / name, frozen / name)
    sys.path.insert(0, str(frozen.resolve()))
    generator = importlib.import_module("generator")
    for name in ("generator", "sources", "scm_v1"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve() != (frozen / f"{name}.py").resolve():
            raise ValueError(f"another {name} module was already loaded")
    import sklearn

    (root / "data").mkdir()
    (root / "raw").mkdir()
    entries = corpus_requests(args.rows, args.seed)
    write_json(root / "requested.json", entries)
    manifest = dict(
        schema="tabu.tar.diverse-fit-corpus.1",
        status="local_unissued",
        generator_files={name: file_hash(frozen / name) for name in source_files},
        runtime=dict(
            python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__
        ),
        rows=args.rows,
        split_seed=args.seed,
        records=[],
    )
    seen_tables = set()
    rejected = []
    try:
        for entry in entries:
            accepted = None
            for attempt in range(100):
                request = {**entry["request"], "seed": entry["request"]["seed"] + attempt}
                try:
                    episode = generator.sample_episodes(**request)[0]
                    target = select_target(episode, entry)
                    dataset = adapt_table(
                        episode, target, name=entry["dataset"], split_seed=args.seed
                    )
                    digest = canonical_hash(dataset["values"])
                    if digest in seen_tables:
                        raise ValueError("duplicate complete table")
                    # Byte-exact value/metadata replay under the frozen implementation.
                    replay = generator.sample_episodes(**request)[0]
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
        write_json(root / "manifest.json", manifest)
    return manifest
