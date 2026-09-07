"""Versioned mixed-type observational SCM generator.

This module deliberately has no Unit-token semantics.  It samples one
ordinary SCM world, draws i.i.d. rows by resampling exogenous event noise,
and only then attaches query/missing annotations.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

CONTRACT_VERSION = "scm.mixed_observational.v1"
FRAMEWORK = "Mixed-Type-Observational-SCM"
SCM_TYPES = (
    "numeric",
    "ordinal",
    "binary",
    "categorical",
    "high_cardinality",
)
AUTO_TARGET_TYPES = ("numeric", "ordinal", "binary", "categorical")
DEFAULT_TYPE_WEIGHTS = {
    "numeric": 70.0,
    "ordinal": 5.0,
    "binary": 10.0,
    "categorical": 5.0,
    "high_cardinality": 5.0,
}
EDGE_FUNCTIONS = ("identity", "tanh", "sin")
EDGE_FUNCTION_PROBS = (0.50, 0.30, 0.20)
MAX_RESAMPLE_ATTEMPTS = 8

_WORLD_TAG = 0x574F524C
_EVENT_TAG = 0x45564E54
_MASK_TAG = 0x4D41534B


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _subseed(master_seed: int, tag: int, attempt: int = 0) -> int:
    """Derive a stable uint64 seed without coupling independent RNG streams."""
    master = int(master_seed) & ((1 << 64) - 1)
    words = np.random.SeedSequence(
        [master & 0xFFFFFFFF, master >> 32, int(tag), int(attempt)]
    ).generate_state(2, dtype=np.uint32)
    return (int(words[0]) << 32) | int(words[1])


def _normalized_type_weights(
    weights: dict[str, float] | None,
) -> tuple[dict[str, float], np.ndarray]:
    source = DEFAULT_TYPE_WEIGHTS if weights is None else weights
    used = {kind: max(float(source.get(kind, 0.0)), 0.0) for kind in SCM_TYPES}
    vector = np.asarray([used[kind] for kind in SCM_TYPES], dtype=np.float64)
    total = float(vector.sum())
    if total <= 0.0:
        raise ValueError("scm_options.type_weights must have positive total weight")
    return used, vector / total


def _sample_target_type(
    rng: np.random.Generator,
    target_type: str,
    used_weights: dict[str, float],
) -> str:
    target = str(target_type or "auto").strip().lower()
    if target != "auto":
        if target not in SCM_TYPES:
            raise ValueError(
                "scm_options.target_type must be auto or one of "
                f"{', '.join(SCM_TYPES)}"
            )
        return target
    probs = np.asarray([used_weights[k] for k in AUTO_TARGET_TYPES], dtype=np.float64)
    if float(probs.sum()) <= 0.0:
        raise ValueError(
            "auto SCM target requires positive weight for numeric, ordinal, "
            "binary, or categorical"
        )
    probs = probs / probs.sum()
    return str(rng.choice(AUTO_TARGET_TYPES, p=probs))


def _n_classes_for(kind: str, n_rows: int, rng: np.random.Generator) -> int | None:
    if kind == "numeric":
        return None
    if kind == "binary":
        return 2
    if kind == "ordinal":
        return int(rng.integers(3, 9))
    if kind == "categorical":
        return int(rng.integers(8, 33))
    high = int(max(48, min(256, max(int(n_rows), 48))))
    low = int(max(32, min(64, high - 1)))
    return int(rng.integers(low, high + 1))


def _standardize_vector(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    out = out - float(out.mean())
    scale = float(out.std())
    if scale < 1e-8:
        out = np.linspace(-1.0, 1.0, out.size, dtype=np.float64)
        scale = float(out.std())
    return out / max(scale, 1e-8)


def _category_embedding(
    rng: np.random.Generator, n_classes: int
) -> list[list[float]]:
    rank = min(8, max(1, int(n_classes) - 1))
    embedding = rng.standard_normal((int(n_classes), rank))
    embedding = embedding - embedding.mean(axis=0, keepdims=True)
    scale = embedding.std(axis=0, keepdims=True)
    embedding = embedding / np.maximum(scale, 1e-8)
    return [[float(x) for x in row] for row in embedding]


def _signed_edge_scales(rng: np.random.Generator, count: int) -> np.ndarray:
    if count <= 0:
        return np.zeros(0, dtype=np.float64)
    magnitudes = rng.uniform(0.5, 2.0, size=count)
    signs = np.where(rng.random(count) < 0.5, -1.0, 1.0)
    raw = magnitudes * signs
    return 1.75 * raw / max(float(np.linalg.norm(raw)), 1e-8)


def _parent_transform(rng: np.random.Generator, parent_type: str) -> str:
    if parent_type == "numeric":
        return str(rng.choice(EDGE_FUNCTIONS, p=EDGE_FUNCTION_PROBS))
    if parent_type in ("binary", "ordinal"):
        return "scaled_scalar"
    return "category_embedding"


def sample_world(
    rng: np.random.Generator,
    *,
    n_features: int,
    n_rows: int,
    sigma: float,
    type_weights: dict[str, float] | None = None,
    target_type: str = "auto",
    max_parents: int = 3,
    target_output_column: int | None = None,
) -> dict[str, Any]:
    """Sample a JSON-serializable mixed-type SCM world."""
    d = int(n_features)
    if d < 2:
        raise ValueError("mixed SCM requires at least two features")
    noise_scale = float(sigma)
    if not (noise_scale > 0.0):
        raise ValueError("mixed SCM requires sigma > 0")
    parent_cap = max(1, int(max_parents))
    output_target = d - 1 if target_output_column is None else int(target_output_column)
    if not (0 <= output_target < d):
        raise ValueError("query_column must be smaller than n_features for v1 SCM")

    used_weights, type_probs = _normalized_type_weights(type_weights)
    topo = [int(x) for x in rng.permutation(d)]
    target_node = int(topo[-1])
    chosen_target_type = _sample_target_type(rng, target_type, used_weights)

    kinds = [str(x) for x in rng.choice(SCM_TYPES, size=d, p=type_probs)]
    kinds[target_node] = chosen_target_type
    n_classes = [_n_classes_for(kinds[j], n_rows, rng) for j in range(d)]

    parents: list[list[int]] = [[] for _ in range(d)]
    for position, node in enumerate(topo):
        if position == 0:
            continue
        cap = min(parent_cap, position)
        if node == target_node:
            count = int(rng.integers(1, cap + 1))
        else:
            count = int(rng.integers(0, cap + 1))
        if count:
            parents[node] = [
                int(x)
                for x in rng.choice(
                    np.asarray(topo[:position], dtype=np.int64),
                    size=count,
                    replace=False,
                )
            ]

    output_to_node = [int(x) for x in rng.permutation([j for j in range(d) if j != target_node])]
    output_to_node.insert(output_target, target_node)
    node_to_output = [0] * d
    for output_col, node in enumerate(output_to_node):
        node_to_output[node] = int(output_col)

    category_embeddings: list[list[list[float]] | None] = [None] * d
    for node in range(d):
        if kinds[node] in ("categorical", "high_cardinality"):
            assert n_classes[node] is not None
            category_embeddings[node] = _category_embedding(rng, int(n_classes[node]))

    nodes: list[dict[str, Any]] = [{} for _ in range(d)]
    for node in topo:
        kind = kinds[node]
        node_parents = parents[node]
        edge_scales = _signed_edge_scales(rng, len(node_parents))
        edges: list[dict[str, Any]] = []

        if kind in ("categorical", "high_cardinality"):
            child_classes = int(n_classes[node] or 2)
            for edge_index, parent in enumerate(node_parents):
                parent_type = kinds[parent]
                transform = _parent_transform(rng, parent_type)
                scale = float(edge_scales[edge_index])
                edge: dict[str, Any] = {
                    "parent": int(parent),
                    "parent_type": parent_type,
                    "transform": transform,
                    "scale": scale,
                }
                if transform == "category_embedding":
                    embedding = category_embeddings[parent]
                    assert embedding is not None
                    rank = len(embedding[0])
                    loading = rng.standard_normal((child_classes, rank))
                    loading = loading - loading.mean(axis=0, keepdims=True)
                    loading = loading / max(float(loading.std()), 1e-8)
                    edge["class_loading"] = [
                        [float(x) for x in row] for row in loading
                    ]
                else:
                    class_weights = _standardize_vector(
                        rng.standard_normal(child_classes)
                    )
                    edge["class_weights"] = [float(x) for x in class_weights]
                edges.append(edge)
            logits_bias = _standardize_vector(rng.normal(0.0, 0.05, child_classes))
            mechanism: dict[str, Any] = {
                "family": "gumbel_max",
                "bias_logits": [float(0.05 * x) for x in logits_bias],
                "noise_family": "gumbel",
                "noise_scale": noise_scale,
                "edges": edges,
            }
        else:
            for edge_index, parent in enumerate(node_parents):
                parent_type = kinds[parent]
                transform = _parent_transform(rng, parent_type)
                edge = {
                    "parent": int(parent),
                    "parent_type": parent_type,
                    "transform": transform,
                    "weight": float(edge_scales[edge_index]),
                }
                if transform == "category_embedding":
                    embedding = category_embeddings[parent]
                    assert embedding is not None
                    loading = _standardize_vector(
                        rng.standard_normal(len(embedding[0]))
                    )
                    edge["loading"] = [float(x) for x in loading]
                edges.append(edge)
            mechanism = {
                "family": "additive_score",
                "intercept": float(rng.normal(0.0, 0.10)),
                "noise_family": "gaussian",
                "noise_scale": noise_scale,
                "edges": edges,
            }
            if kind == "binary":
                mechanism["thresholds"] = [0.0]
            elif kind == "ordinal":
                classes = int(n_classes[node] or 3)
                mechanism["thresholds"] = [
                    float(x) for x in np.linspace(-0.9, 0.9, classes - 1)
                ]

        nodes[node] = {
            "node": int(node),
            "type": kind,
            "n_classes": n_classes[node],
            "parents": [int(x) for x in node_parents],
            "category_embedding": category_embeddings[node],
            "mechanism": mechanism,
        }

    return {
        "contract_version": CONTRACT_VERSION,
        "n_features": d,
        "type_weights": used_weights,
        "topological_order": topo,
        "parents": parents,
        "target_node": target_node,
        "target_output_column": output_target,
        "output_to_node": output_to_node,
        "node_to_output": node_to_output,
        "nodes": nodes,
    }


def _scalar_parent_values(
    values: np.ndarray,
    world: dict[str, Any],
    edge: dict[str, Any],
) -> np.ndarray:
    parent = int(edge["parent"])
    parent_values = values[:, parent]
    transform = str(edge["transform"])
    if transform == "identity":
        return np.clip(parent_values, -6.0, 6.0)
    if transform == "tanh":
        return np.tanh(parent_values)
    if transform == "sin":
        return np.sin(parent_values)
    if transform == "scaled_scalar":
        classes = int(world["nodes"][parent]["n_classes"] or 2)
        return 2.0 * parent_values / max(classes - 1, 1) - 1.0
    raise ValueError("category embeddings require an edge-specific loading")


def _category_rows(
    values: np.ndarray,
    world: dict[str, Any],
    parent: int,
) -> np.ndarray:
    embedding = world["nodes"][parent]["category_embedding"]
    if embedding is None:
        raise ValueError("categorical parent is missing its embedding")
    matrix = np.asarray(embedding, dtype=np.float64)
    indices = values[:, parent].astype(np.int64)
    return matrix[indices]


def sample_rows(
    world: dict[str, Any],
    *,
    n_rows: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw i.i.d. observational rows from a fixed world."""
    n = int(n_rows)
    d = int(world["n_features"])
    values = np.zeros((n, d), dtype=np.float64)

    for node in world["topological_order"]:
        spec = world["nodes"][node]
        kind = str(spec["type"])
        mechanism = spec["mechanism"]
        edges = mechanism["edges"]
        noise_scale = float(mechanism["noise_scale"])

        if mechanism["family"] == "gumbel_max":
            classes = int(spec["n_classes"])
            logits = np.tile(
                np.asarray(mechanism["bias_logits"], dtype=np.float64),
                (n, 1),
            )
            for edge in edges:
                scale = float(edge["scale"])
                if edge["transform"] == "category_embedding":
                    parent = int(edge["parent"])
                    encoded = _category_rows(values, world, parent)
                    loading = np.asarray(edge["class_loading"], dtype=np.float64)
                    logits += scale * (encoded @ loading.T) / np.sqrt(loading.shape[1])
                else:
                    encoded = _scalar_parent_values(values, world, edge)
                    class_weights = np.asarray(edge["class_weights"], dtype=np.float64)
                    logits += scale * encoded[:, None] * class_weights[None, :]
            logits += rng.gumbel(0.0, noise_scale, size=(n, classes))
            values[:, node] = np.argmax(logits, axis=1)
            continue

        score = np.full(n, float(mechanism["intercept"]), dtype=np.float64)
        for edge in edges:
            weight = float(edge["weight"])
            if edge["transform"] == "category_embedding":
                parent = int(edge["parent"])
                encoded = _category_rows(values, world, parent)
                loading = np.asarray(edge["loading"], dtype=np.float64)
                contribution = (encoded @ loading) / np.sqrt(loading.size)
            else:
                contribution = _scalar_parent_values(values, world, edge)
            score += weight * contribution
        score += rng.normal(0.0, noise_scale, size=n)

        if kind == "numeric":
            values[:, node] = score
        elif kind == "binary":
            values[:, node] = (score > 0.0).astype(np.float64)
        elif kind == "ordinal":
            thresholds = np.asarray(mechanism["thresholds"], dtype=np.float64)
            values[:, node] = np.sum(score[:, None] > thresholds[None, :], axis=1)
        else:
            raise ValueError(f"unknown scalar SCM node type {kind!r}")

    output_to_node = np.asarray(world["output_to_node"], dtype=np.int64)
    return values[:, output_to_node]


def _typed_values(
    values: np.ndarray,
    column_types: list[str],
) -> list[list[float | int]]:
    result: list[list[float | int]] = []
    for row in values:
        result.append(
            [
                float(value) if column_types[j] == "numeric" else int(value)
                for j, value in enumerate(row)
            ]
        )
    return result


def _valid_target(world: dict[str, Any], values: np.ndarray) -> bool:
    target_col = int(world["target_output_column"])
    target_node = int(world["target_node"])
    target_type = str(world["nodes"][target_node]["type"])
    target = values[:, target_col]
    if not np.isfinite(values).all():
        return False
    if target_type == "numeric":
        return bool(float(np.var(target)) > 1e-8)
    return bool(np.unique(target).size >= 2)


def _mechanism_manifest(
    world: dict[str, Any],
    *,
    master_seed: int,
    world_seed: int,
    event_seed: int,
    mask_seed: int,
    attempt: int,
    values_hash: str,
) -> dict[str, Any]:
    edges = [
        [int(parent), int(node)]
        for node, node_parents in enumerate(world["parents"])
        for parent in node_parents
    ]
    columns: list[dict[str, Any]] = []
    for output_col, node in enumerate(world["output_to_node"]):
        item = dict(world["nodes"][node])
        item["output_col"] = int(output_col)
        columns.append(item)
    target_node = int(world["target_node"])
    target_spec = world["nodes"][target_node]
    return {
        "framework": FRAMEWORK,
        "contract_version": CONTRACT_VERSION,
        "scope": "observational_only",
        "row_meaning": "iid_sample",
        "uses_unit_token": False,
        "prior": {"type_weights": dict(world["type_weights"])},
        "graph": {
            "topological_order": list(world["topological_order"]),
            "parents": list(world["parents"]),
            "edges": edges,
            "target_is_sink": True,
        },
        "output": {
            "output_to_node": list(world["output_to_node"]),
            "node_to_output": list(world["node_to_output"]),
        },
        "target": {
            "node": target_node,
            "output_col": int(world["target_output_column"]),
            "type": target_spec["type"],
            "n_classes": target_spec["n_classes"],
        },
        "columns": columns,
        "rng": {
            "bit_generator": "PCG64",
            "master_seed": int(master_seed),
            "world_seed": int(world_seed),
            "event_seed": int(event_seed),
            "mask_seed": int(mask_seed),
        },
        "resample_attempt": int(attempt),
        "world_hash": canonical_hash(world),
        "values_hash": values_hash,
    }


def world_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the pure world object carried by a v1 mechanism manifest."""
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ValueError(f"manifest is not a {CONTRACT_VERSION} contract")
    output_to_node = [int(x) for x in manifest["output"]["output_to_node"]]
    node_to_output = [int(x) for x in manifest["output"]["node_to_output"]]
    d = len(output_to_node)
    nodes: list[dict[str, Any] | None] = [None] * d
    for column in manifest["columns"]:
        node_spec = dict(column)
        node_spec.pop("output_col", None)
        nodes[int(node_spec["node"])] = node_spec
    if any(node is None for node in nodes):
        raise ValueError("manifest columns do not cover every logical SCM node")
    target = manifest["target"]
    world = {
        "contract_version": CONTRACT_VERSION,
        "n_features": d,
        "type_weights": dict(manifest["prior"]["type_weights"]),
        "topological_order": [
            int(x) for x in manifest["graph"]["topological_order"]
        ],
        "parents": [
            [int(x) for x in parent_list]
            for parent_list in manifest["graph"]["parents"]
        ],
        "target_node": int(target["node"]),
        "target_output_column": int(target["output_col"]),
        "output_to_node": output_to_node,
        "node_to_output": node_to_output,
        "nodes": nodes,
    }
    if canonical_hash(world) != manifest.get("world_hash"):
        raise ValueError("manifest world_hash does not match its mechanism content")
    return world


def sample_mixed_scm_episode(
    *,
    n_units: int,
    n_features: int,
    missing_frac: float,
    query_frac: float,
    sigma: float,
    seed: int,
    return_mechanism: bool,
    query_mode: str = "label_cell",
    query_column: int | None = None,
    type_weights: dict[str, float] | None = None,
    target_type: str = "auto",
    max_parents: int = 3,
) -> dict[str, Any]:
    """Sample, validate, and pack one v1 mixed-type SCM episode."""
    from sources import pack_grid

    master_seed = int(seed)
    mask_seed = _subseed(master_seed, _MASK_TAG)
    accepted: tuple[dict[str, Any], np.ndarray, int, int, int] | None = None
    for attempt in range(MAX_RESAMPLE_ATTEMPTS):
        world_seed = _subseed(master_seed, _WORLD_TAG, attempt)
        event_seed = _subseed(master_seed, _EVENT_TAG, attempt)
        world = sample_world(
            np.random.default_rng(world_seed),
            n_features=int(n_features),
            n_rows=int(n_units),
            sigma=float(sigma),
            type_weights=type_weights,
            target_type=target_type,
            max_parents=int(max_parents),
            target_output_column=query_column,
        )
        values = sample_rows(
            world,
            n_rows=int(n_units),
            rng=np.random.default_rng(event_seed),
        )
        if _valid_target(world, values):
            accepted = (world, values, attempt, world_seed, event_seed)
            break
    if accepted is None:
        raise ValueError(
            "mixed SCM could not produce a finite non-degenerate target after "
            f"{MAX_RESAMPLE_ATTEMPTS} attempts"
        )

    world, values, attempt, world_seed, event_seed = accepted
    output_nodes = world["output_to_node"]
    column_types = [str(world["nodes"][node]["type"]) for node in output_nodes]
    n_classes = [world["nodes"][node]["n_classes"] for node in output_nodes]
    typed_values = _typed_values(values, column_types)
    values_hash = canonical_hash(typed_values)
    mechanism = _mechanism_manifest(
        world,
        master_seed=master_seed,
        world_seed=world_seed,
        event_seed=event_seed,
        mask_seed=mask_seed,
        attempt=attempt,
        values_hash=values_hash,
    )
    target_output = int(world["target_output_column"])
    return pack_grid(
        values,
        np.random.default_rng(mask_seed),
        missing_frac=float(missing_frac),
        query_frac=float(query_frac),
        column_types=column_types,
        n_classes=n_classes,
        seed=master_seed,
        return_mechanism=bool(return_mechanism),
        mechanism=mechanism,
        query_mode=query_mode,
        query_column=target_output,
        source="scm",
    )
