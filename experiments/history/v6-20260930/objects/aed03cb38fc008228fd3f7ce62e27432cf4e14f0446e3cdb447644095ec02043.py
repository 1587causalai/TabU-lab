# Derived from TabU-lab (Apache-2.0); unchanged version-1 request ordering.
import itertools

VERSION = "anchor120.1"
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
            dict(
                family="scm_mixed_v1",
                profile=profile,
                target_type=targets[ti],
                request=request,
            )
        )
    for pi, gi, ki, ni, ti in itertools.product(
        range(3), range(2), range(2), range(2), range(2)
    ):
        profile = list(DISCO_PROFILES)[pi]
        target = (
            "numeric"
            if ti == 0
            else ("binary", "ordinal", "categorical")[(pi + gi + ki + ni) % 3]
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
            dict(
                family="discoscm", profile=profile, target_type=target, request=request
            )
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
                request=dict(
                    source="scm", api_version="v0", n_features=width, sigma=noise
                ),
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
                target_type="binary"
                if source == "sklearn_make_classification"
                else "numeric",
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
