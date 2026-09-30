"""Single-table generation, with source-specific options and retained raw envelope."""

import copy
import math
import numpy as np
from .generators import discoscm as disco
from .generators.mixed_scm import sample_mixed_scm_episode
from .generators.numeric_scm import scm_anm
from .generators.sklearn_synthetic import sklearn_synthetic, SKLEARN_SYNTH_CANONICAL
from .generators.registry import SOURCES
from .core.validation import validate_table


def _validate_request(source, seed, n_rows, options):
    if source not in SOURCES:
        raise ValueError(f"unknown source: {source}; choose from {list(SOURCES)}")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32-1]")
    if type(n_rows) is not int or n_rows < 3:
        raise ValueError("n_rows must be an integer >= 3")
    if not isinstance(options, dict):
        raise ValueError("options must be a dictionary")
    defaults = SOURCES[source]["defaults"]
    unknown = set(options) - set(defaults)
    if unknown:
        raise ValueError(f"unsupported options for {source}: {sorted(unknown)}")
    opts = {**defaults, **options}
    for key in (
        "sigma",
        "missing_frac",
        "query_frac",
        "independent_frac",
        "token_heritability",
        "beta_min",
        "beta_max",
    ):
        if key not in opts:
            continue
        val = opts[key]
        if (
            isinstance(val, bool)
            or not isinstance(val, (int, float))
            or not math.isfinite(val)
            or val < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
        if (
            key
            in ("missing_frac", "query_frac", "independent_frac", "token_heritability")
            and val > 1
        ):
            raise ValueError(f"{key} must be <= 1")
    for key, low, high in [
        ("n_features", 2, 128 if source == "mixed_scm" else 1000),
        ("unit_dim", 2, 1024),
        ("max_parents", 1, 8 if source == "mixed_scm" else 1000),
    ]:
        val = opts.get(key)
        if val is not None and (type(val) is not int or not low <= val <= high):
            raise ValueError(f"{key} must be an integer in [{low}, {high}]")
    if source == "mixed_scm" and opts["max_parents"] is None:
        raise ValueError("mixed_scm max_parents must be an integer")
    if (
        source == "sklearn_make_classification"
        and opts["n_features"] is not None
        and opts["n_features"] < 4
    ):
        raise ValueError(
            "classification requires n_features >= 4 for two clusters per class"
        )
    if opts.get("query_mode") not in (None, "any_cell", "label_cell"):
        raise ValueError("query_mode must be any_cell or label_cell")
    if opts.get("graph_family") not in (None, "sparse", "star"):
        raise ValueError("graph_family must be sparse or star")
    if "target_type" in opts and opts["target_type"] not in ("auto", *disco.COL_TYPES):
        raise ValueError("invalid target_type")
    for key in ("return_mechanism", "enable_heavy_tails", "column_normalize"):
        if key in opts and type(opts[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    weights = opts.get("type_weights")
    if weights is not None:
        if not isinstance(weights, dict) or set(weights) - set(disco.COL_TYPES):
            raise ValueError("invalid type_weights keys")
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in weights.values()
            )
            or sum(weights.values()) <= 0
        ):
            raise ValueError(
                "type_weights must be finite, nonnegative and have positive sum"
            )
    if opts.get("beta_min", 0) > opts.get("beta_max", float("inf")):
        raise ValueError("beta_min exceeds beta_max")
    if source == "mixed_scm" and opts["sigma"] <= 0:
        raise ValueError("mixed_scm requires sigma > 0")
    if source == "discoscm" and opts["beta_min"] <= 0:
        raise ValueError("beta_min must be positive")
    return opts


def generate_table(source, *, seed, n_rows=256, options=None):
    """Return {'raw': legacy episode envelope, 'metadata': new package identity}.

    n_features includes the target, when a source designates one. Full truth
    remains in raw.table.values; masks must be applied by consumers.
    """
    options = copy.deepcopy({} if options is None else options)
    opts = _validate_request(source, seed, n_rows, {} if options is None else options)
    shape_rng = np.random.default_rng(np.random.SeedSequence([seed, 0x53485045, 0]))
    width = opts.pop("n_features")
    if source == "discoscm":
        width, _ = disco._resolve_n_features(width, shape_rng)
        unit_dim, _ = disco._resolve_unit_dim(opts.pop("unit_dim"), shape_rng)
    else:
        width = 20 if width is None else width
    qcol = opts.get("query_column")
    if qcol is not None and (type(qcol) is not int or not 0 <= qcol < width):
        raise ValueError("query_column must be within table width")
    opts["query_mode"] = opts["query_mode"] or SOURCES[source]["query_mode"]
    base = dict(n_units=n_rows, n_features=width, seed=seed, **opts)
    rng = np.random.default_rng(seed)
    if source == "discoscm":
        raw = disco.sample_episode(
            rng,
            unit_dim=unit_dim,
            debug=False,
            dag_edge_p=disco.DEFAULT_DAG_EDGE_P,
            **base,
        )
    elif source == "mixed_scm":
        raw = sample_mixed_scm_episode(**base)
    elif source == "numeric_scm":
        raw = scm_anm(rng=rng, source="scm", **base)
    else:
        raw = sklearn_synthetic(
            rng=rng, source=source, source_name=SKLEARN_SYNTH_CANONICAL[source], **base
        )
    validate_table(raw)
    return dict(
        raw=raw,
        metadata=dict(
            schema="tfm-data.table.1",
            package="tfm-data",
            version="0.1.0",
            source=source,
            seed=seed,
            n_rows=n_rows,
            options={} if options is None else options,
        ),
    )


def sample_request(**request):
    """Internal adapter for the retained Anchor120 request schema."""
    req = dict(request)
    source = req.pop("source")
    version = req.pop("api_version", "v0")
    if source == "scm":
        source = "mixed_scm" if version == "v1" else "numeric_scm"
    seed = req.pop("seed")
    rows = req.pop("n_units")
    for name in ("batch_size", "n_episodes"):
        req.pop(name, None)
    req.update(req.pop("scm_options", {}) or {})
    return generate_table(source, seed=seed, n_rows=rows, options=req)["raw"]
