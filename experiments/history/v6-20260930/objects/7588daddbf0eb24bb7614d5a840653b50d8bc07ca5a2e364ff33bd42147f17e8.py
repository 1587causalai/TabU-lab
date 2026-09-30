# Derived from TabU-lab (Apache-2.0); pure-data imports only.
from collections import Counter
import numpy as np
from ..core.typed import dataset_features, full_train_test_split, validate_full_dataset


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
    if np.unique(y).size < 2 or (
        features[-1]["kind"] == "numeric" and float(y.var()) < 1e-10
    ):
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
            target_parent_types=[
                e["parent_type"] for e in target_column["mechanism"]["edges"]
            ],
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
        result.update(
            mechanism=episode["mechanism"], replay_kind="pinned_request_and_runtime"
        )
    return result
