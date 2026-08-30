from __future__ import annotations

import pytest
import torch

from tabu_lab.contracts import FeatureKind, FeatureRole, OriginState, TruthSidecar, origin_mask
from tabu_lab.experiments.feasibility import assess_nw_targets
from tabu_lab.experiments.s1_table_synthetic import (
    COMPLETION_CONTRACTS,
    build_s1_completion_corpus,
    build_s1_supervised_corpus,
    build_s1_table_corpus,
)
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig


@pytest.fixture(scope="module")
def completion_corpus():  # type: ignore[no-untyped-def]
    return build_s1_completion_corpus("tabuf")


@pytest.fixture(scope="module")
def tabul_corpus():  # type: ignore[no-untyped-def]
    return build_s1_supervised_corpus("tabul")


@pytest.fixture(scope="module")
def tabufl_corpus():  # type: ignore[no-untyped-def]
    return build_s1_supervised_corpus("tabufl")


def _typed_target_counts(episode) -> tuple[int, int]:  # type: ignore[no-untyped-def]
    numeric_features = torch.tensor(
        tuple(spec.kind is FeatureKind.NUMERIC for spec in episode.evidence.feature_specs),
        dtype=torch.bool,
    ).unsqueeze(0)
    targets = episode.truth.target_mask
    return int((targets & numeric_features).sum()), int((targets & ~numeric_features).sum())


def test_completion_corpus_is_shared_typed_and_multi_episode(completion_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = completion_corpus

    assert corpus.dataset.shape == (256, 6)
    assert tuple(spec.kind for spec in corpus.dataset.feature_specs) == (
        FeatureKind.NUMERIC,
        FeatureKind.NUMERIC,
        FeatureKind.NUMERIC,
        FeatureKind.NUMERIC,
        FeatureKind.CATEGORICAL,
        FeatureKind.CATEGORICAL,
    )
    assert tuple(len(corpus.episodes(name)) for name in ("train", "validation", "test")) == (
        24,
        4,
        4,
    )
    assert corpus.schedule.episode_count == 32
    assert corpus.schedule.targets_per_episode == 12
    assert corpus.fit_value_mask.sum(dim=0).tolist() == [144, 144, 144, 144, 0, 0]
    assert corpus.numeric_normalizer.statistics.counts.tolist() == [144, 144, 144, 144, 0, 0]

    for partition in ("train", "validation", "test"):
        for episode in corpus.episodes(partition):
            assert episode.truth.target_count == 12
            assert _typed_target_counts(episode) == (8, 4)
            assert not bool(origin_mask(episode.evidence.origin_states, OriginState.QUERY).any())
            assert bool(
                origin_mask(
                    episode.evidence.origin_states,
                    OriginState.ARTIFICIAL_MASK,
                )[episode.truth.target_mask].all()
            )
            assert assess_nw_targets(
                episode.feasibility_targets,
                report_id=f"completion-{partition}-{episode.ordinal}",
            ).ready


def test_legacy_completion_contracts_receive_bit_identical_corpus(completion_corpus) -> None:  # type: ignore[no-untyped-def]
    expected = completion_corpus.corpus_hash
    assert tuple(contract for contract in COMPLETION_CONTRACTS if contract != "tabu.cell.base") == (
        "tabuf",
        "tabu.unit_row",
        "tabu.unit_pair",
    )
    assert build_s1_completion_corpus("tabu.unit_row").corpus_hash == expected
    assert build_s1_completion_corpus("tabu.unit_pair").corpus_hash == expected


def test_tabubase_completion_corpus_is_an_independent_asset() -> None:
    corpus = build_s1_completion_corpus("tabu.cell.base")
    assert corpus.dataset.dataset_id == "s1-tabu-cell-base-completion-v1"
    assert corpus.corpus_hash != build_s1_completion_corpus("tabuf").corpus_hash
    assert corpus.builder_options["profile"] == "completion.artificial_mask.v1"


def test_supervised_corpora_freeze_typed_schema_and_fit_only_statistics(
    tabul_corpus,
    tabufl_corpus,
) -> None:  # type: ignore[no-untyped-def]
    for corpus in (tabul_corpus, tabufl_corpus):
        assert corpus.dataset.shape == (512, 8)
        assert tuple(spec.role for spec in corpus.dataset.feature_specs) == (
            FeatureRole.PREDICTOR,
            FeatureRole.PREDICTOR,
            FeatureRole.PREDICTOR,
            FeatureRole.PREDICTOR,
            FeatureRole.PREDICTOR,
            FeatureRole.PREDICTOR,
            FeatureRole.RESPONSE,
            FeatureRole.RESPONSE,
        )
        assert corpus.builder_options == {
            "label_address_plan": "predictor_unit_linked_per_label_v2",
            "label_columns": (6, 7),
        }
        assert tuple(len(corpus.episodes(name)) for name in ("train", "validation", "test")) == (
            16,
            4,
            4,
        )
        # The numeric response statistic has exactly the 128 permanent context
        # values; the 256 train-query truths and every held-out truth are absent.
        assert corpus.fit_value_mask[:, 6].sum().item() == 128
        assert corpus.numeric_normalizer.statistics.counts[6].item() == 128
        assert not bool(corpus.fit_value_mask[:, 7].any())

    assert tabul_corpus.fit_value_mask.sum(dim=0).tolist() == [
        384,
        384,
        384,
        384,
        384,
        384,
        128,
        0,
    ]
    # TabUFL additionally excludes the union of its 16 F targets per train
    # episode.  The rotation is balanced to within one target per predictor.
    assert tabufl_corpus.fit_value_mask.sum(dim=0).tolist() == [
        341,
        341,
        341,
        341,
        342,
        342,
        128,
        0,
    ]


def test_tabul_episodes_have_numeric_and_categorical_query_ledgers(tabul_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = tabul_corpus
    assert corpus.schedule.targets_per_episode == 32
    for partition in ("train", "validation", "test"):
        for episode in corpus.episodes(partition):
            query = origin_mask(episode.evidence.origin_states, OriginState.QUERY)
            assert episode.target_family_masks.keys() == {"L"}
            assert torch.equal(episode.target_family_masks["L"], episode.truth.target_mask)
            assert int(query.any(dim=1).sum()) == 16
            assert _typed_target_counts(episode) == (16, 16)
            assert assess_nw_targets(
                episode.feasibility_targets,
                report_id=f"tabul-{partition}-{episode.ordinal}",
            ).ready


def test_tabufl_runs_f_and_l_in_the_same_truth_isolated_episode(tabufl_corpus) -> None:  # type: ignore[no-untyped-def]
    corpus = tabufl_corpus
    assert corpus.schedule.targets_per_episode == 48
    for partition in ("train", "validation", "test"):
        for episode in corpus.episodes(partition):
            feature_targets = episode.target_family_masks["F"]
            label_targets = episode.target_family_masks["L"]
            query_rows = origin_mask(episode.evidence.origin_states, OriginState.QUERY).any(dim=1)
            assert int(feature_targets.sum()) == 16
            assert int(label_targets.sum()) == 32
            assert int(query_rows.sum()) == 16
            assert torch.equal(feature_targets | label_targets, episode.truth.target_mask)
            assert not bool((feature_targets & label_targets).any())
            # One F target lives on every query row, but query rows are excluded
            # from both F and L terminal support ledgers.
            assert bool(feature_targets[query_rows].any(dim=1).all())
            width = episode.evidence.forward_values.shape[1]
            for target in episode.feasibility_targets:
                for arm in target.arms:
                    assert all(
                        not bool(query_rows[support_id // width]) for support_id in arm.support_ids
                    )


def test_heldout_sources_are_train_context_or_current_query_predictors_only(
    completion_corpus,
    tabul_corpus,
) -> None:  # type: ignore[no-untyped-def]
    completion = completion_corpus.validation_episodes[0]
    completion_train = set(completion_corpus.typed_split.partition("train").row_ids)
    completion_source_rows = {
        completion.evidence.row_ids[row]
        for row in torch.nonzero(completion.evidence.source_mask.any(dim=1), as_tuple=False)
        .flatten()
        .tolist()
    }
    assert completion_source_rows <= completion_train

    supervised = tabul_corpus.validation_episodes[0]
    context_rows = set(tabul_corpus.typed_split.partition("train").row_ids[:128])
    query_rows = origin_mask(supervised.evidence.origin_states, OriginState.QUERY).any(dim=1)
    response = torch.tensor(
        tuple(spec.role is FeatureRole.RESPONSE for spec in supervised.evidence.feature_specs),
        dtype=torch.bool,
    )
    for row, row_id in enumerate(supervised.evidence.row_ids):
        sources = supervised.evidence.source_mask[row]
        if row_id in context_rows:
            assert bool(sources.all())
        elif bool(query_rows[row]):
            assert bool(sources[~response].all())
            assert not bool(sources[response].any())
        else:
            assert not bool(sources.any())


def test_truth_mutation_cannot_change_model_facing_evidence(completion_corpus) -> None:  # type: ignore[no-untyped-def]
    episode = completion_corpus.train_episodes[0]
    before = episode.evidence.evidence_hash
    changed_values = episode.truth.target_values.clone()
    changed_values[episode.truth.target_mask] += 17.0
    changed_truth = TruthSidecar(
        episode_id=episode.truth.episode_id,
        recipe_hash=episode.truth.recipe_hash,
        row_ids=episode.truth.row_ids,
        feature_names=episode.truth.feature_names,
        target_values=changed_values,
        target_mask=episode.truth.target_mask,
        metadata=episode.truth.metadata,
    )

    assert changed_truth.truth_hash != episode.truth.truth_hash
    assert episode.evidence.evidence_hash == before
    assert bool((episode.evidence.forward_values[episode.truth.target_mask] == 0).all())
    assert not hasattr(episode.evidence, "truth")
    assert not hasattr(episode.evidence, "target_values")


@pytest.mark.parametrize(
    ("contract_id", "corpus_fixture"),
    (
        ("tabuf", "completion_corpus"),
        ("tabu.unit_row", "completion_corpus"),
        ("tabu.unit_pair", "completion_corpus"),
        ("tabul", "tabul_corpus"),
        ("tabufl", "tabufl_corpus"),
    ),
)
def test_first_s1_episode_forwards_through_each_table_contract(
    contract_id: str,
    corpus_fixture: str,
    request: pytest.FixtureRequest,
) -> None:
    corpus = request.getfixturevalue(corpus_fixture)
    config = ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=2,
        max_features=16,
        dropout=0.0,
    )
    options = {"config": config}
    if contract_id in {"tabul", "tabufl"}:
        options.update(
            label_columns=(6, 7),
            label_address_plan="predictor_unit_linked_per_label_v2",
        )
    model = build_model(contract_id, **options)
    episode = corpus.train_episodes[0]

    with torch.no_grad():
        prediction = model(episode.evidence)

    targets = episode.truth.target_mask
    assert int(prediction.auxiliaries["target_mask"].sum()) == episode.truth.target_count
    assert bool(prediction.auxiliaries["support_available"][targets].all())


def test_s1_table_dispatch_rejects_non_table_contracts() -> None:
    with pytest.raises(ValueError, match="no S1 table corpus"):
        build_s1_table_corpus("tabu4graph")
    with pytest.raises(ValueError, match="unsupported S1 completion"):
        build_s1_completion_corpus("tabul")
    with pytest.raises(ValueError, match="unsupported S1 supervised"):
        build_s1_supervised_corpus("tabuf")
