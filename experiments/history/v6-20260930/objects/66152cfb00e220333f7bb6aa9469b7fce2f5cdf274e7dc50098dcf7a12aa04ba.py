"""Source-specific public parameters. This registry also drives CLI reference output."""

from .sklearn_synthetic import SKLEARN_SYNTH_CANONICAL

COMMON = dict(
    n_features=None,
    missing_frac=0.05,
    query_frac=0.15,
    query_mode=None,
    return_mechanism=True,
)
DISCO = dict(
    unit_dim=None,
    sigma=0.3,
    enable_heavy_tails=False,
    column_normalize=True,
    type_weights=None,
    independent_frac=0.05,
    max_parents=None,
    token_heritability=0.75,
    beta_min=0.5,
    beta_max=2.0,
    graph_family=None,
    query_column=None,
)
SOURCES = {
    "mixed_scm": dict(
        family="mixed_scm",
        query_mode="label_cell",
        defaults={
            **COMMON,
            "sigma": 0.3,
            "target_type": "auto",
            "type_weights": None,
            "max_parents": 3,
            "query_column": None,
        },
    ),
    "discoscm": dict(
        family="discoscm", query_mode="any_cell", defaults={**COMMON, **DISCO}
    ),
    "numeric_scm": dict(
        family="numeric_scm", query_mode="label_cell", defaults={**COMMON, "sigma": 0.3}
    ),
    **{
        name: dict(
            family="sklearn_synthetic",
            query_mode="any_cell" if name == "sklearn_low_rank" else "label_cell",
            defaults=dict(COMMON),
        )
        for name in SKLEARN_SYNTH_CANONICAL
    },
}


def list_generators():
    return SOURCES
