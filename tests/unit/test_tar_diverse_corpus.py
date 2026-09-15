from collections import Counter

import pytest

from tabu_lab.tar_data import dataset_features
from tabu_lab.tar_diverse_corpus import adapt_table, corpus_requests, select_target


def test_corpus_has_120_strata_with_all_256_rows_and_valid_star_requests():
    entries = corpus_requests()
    assert len(entries) == len({x["dataset"] for x in entries}) == 120
    assert Counter(x["family"] for x in entries) == dict(
        scm_mixed_v1=48, discoscm=48, scm_numeric_v0=12, sklearn_synthetic=12
    )
    assert all(x["request"]["n_units"] == 256 for x in entries)
    for x in entries:
        if x["request"].get("graph_family") == "star":
            assert x["request"]["max_parents"] is None
            assert x["request"]["n_features"] >= 8
    assert {x["target_type"] for x in entries} == {"numeric", "ordinal", "binary", "categorical"}


def test_adapter_keeps_full_declared_domains_and_target_permutation():
    raw = dict(
        table=dict(
            column_types=["categorical", "ordinal", "numeric", "binary"],
            n_classes=[20, 5, None, 2],
            values=[[i % 2, i % 3, float(i), i % 2] for i in range(256)],
        )
    )
    data = adapt_table(raw, 0, name="mixed", split_seed=7)
    features = dataset_features(data)
    assert [f.kind for f in features] == ["ordinal", "numeric", "nominal", "nominal"]
    assert len(features[0].domain) == 5 and len(features[-1].domain) == 20
    assert data["values"][10] == [1, 10.0, 0, 0]
    assert len(data["splits"]["train"]) == 204 and len(data["splits"]["test"]) == 52
    data["features"][-1]["domain"] = ["0", "1"]
    with pytest.raises(ValueError, match="contradicts"):
        dataset_features(data)


def test_discoscm_requested_star_must_be_realized():
    entry = dict(family="discoscm", request=dict(graph_family="star"), target_type="numeric")
    ep = dict(table=dict(column_types=["numeric"]), response_law=dict(graph_family="sparse"))
    with pytest.raises(ValueError, match="did not materialize"):
        select_target(ep, entry)
