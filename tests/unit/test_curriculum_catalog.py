"""Provenance, stage policy and truth-isolation checks; no optimization runs."""

import copy
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace

import pytest

from tabu_lab.curriculum import (
    EpisodeRequest,
    EpisodeSeeds,
    Provenance,
    bind_legacy_corpus,
    bind_table,
    reference_stage,
    require_disjoint,
    select_stage,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bind(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"schema": "fixture-source.v1", "seed": 17}))
    provenance = Provenance(source, digest(source), "fixture", 17, 23)

    def create(name="old", *, offset=0, world=None, payload=None, kind="synthetic"):
        if payload is None:
            payload = {
                "schema": "tabu.tar.typed-fit-table.1", "dataset": name,
                "values": [[offset + i / 2, i % 2, 3 * i + offset] for i in range(10)],
                "features": [{"kind": "numeric"}, {"kind": "nominal", "domain": ["a", "b"]},
                             {"kind": "numeric"}],
                "splits": {"train": list(range(6)), "validation": [6, 7], "test": [8, 9]},
            }
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload))
        return bind_table(path, expected_sha256=digest(path), table_id=name, cohort=name,
                          kind=kind, provenance=replace(provenance, world_id=world))

    return create


def test_core_import_needs_no_torch_or_model():
    code = (
        "import sys; import tabu_lab.curriculum; "
        "assert 'torch' not in sys.modules; "
        "assert not any(n.startswith('tabu_lab.models') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_binding_records_versions_and_splits_without_values(bind):
    table = bind()
    record = table.identity()
    assert record["provenance"]["generation_seed"] == 17
    assert table.rows("train") == tuple(range(6))
    assert table.rows("test") == (8, 9)
    assert "path" not in record and "values" not in record
    assert "manifest_path" not in record["provenance"]
    table.verify()


def test_byte_and_source_drift_rejected(bind):
    table = bind()
    table.path.write_text(table.path.read_text() + "\n")
    with pytest.raises(ValueError, match="dataset digest mismatch"):
        table.verify()
    table = bind("other")
    table.provenance.manifest_path.write_text("{}")
    with pytest.raises(ValueError, match="source manifest digest mismatch"):
        table.verify()


@pytest.mark.parametrize("mode", ["overlap", "missing", "duplicate", "outside", "boolean"])
def test_bad_splits_rejected(bind, mode):
    payload = json.loads(bind().path.read_text())
    if mode == "overlap":
        payload["splits"]["test"] = [0, 9]
    elif mode == "missing":
        payload["splits"]["test"] = [8]
    elif mode == "duplicate":
        payload["splits"]["test"] = [8, 8, 9]
    elif mode == "outside":
        payload["splits"]["test"] = [8, 10]
    else:
        payload["splits"]["train"][0] = False
    with pytest.raises(ValueError, match="split"):
        bind("bad", payload=payload)


@pytest.mark.parametrize("change", ["rename", "resplit", "reorder"])
def test_old_new_aliases_cannot_evade_separation(bind, change):
    old = bind()
    payload = json.loads(old.path.read_text())
    payload["dataset"] = "renamed"
    if change == "resplit":
        payload["splits"]["train"][0], payload["splits"]["test"][0] = 8, 0
    if change == "reorder":
        payload["values"].reverse()
    new = bind("new", payload=payload)
    assert old.sha256 != new.sha256
    with pytest.raises(ValueError, match="content_sha256"):
        require_disjoint([old], [new])


def test_table_and_world_separation_have_distinct_evidence(bind):
    old, new = bind(), bind("new", offset=100)
    result = require_disjoint([old], [new])
    assert not result["world_separation_established"]
    assert result["unknown_world_ids"] == ["old", "new"]
    with pytest.raises(ValueError, match="missing world"):
        require_disjoint([old], [new], level="world")
    old = replace(old, provenance=replace(old.provenance, world_id="world-A"))
    new = replace(new, provenance=replace(new.provenance, world_id="world-A"))
    with pytest.raises(ValueError, match="world_id"):
        require_disjoint([old], [new])
    new = replace(new, provenance=replace(new.provenance, world_id="world-B"))
    # Different caller-declared strings cannot certify a parsed source identity.
    assert not require_disjoint([old], [new])["world_separation_established"]
    with pytest.raises(ValueError, match="missing world"):
        require_disjoint([old], [new], level="world")


def test_custom_probes_and_revised_defaults_are_supported(bind):
    tables = (bind(), bind("next", offset=100))
    custom = select_stage("two-table-ablation", tables, question="Does this mechanism fit?",
                          fit_mode="independent")
    assert custom.expected_count is None
    assert custom.manifest()["status"] == "planned"
    revised = reference_stage("fit8", tables, name="fit2", expected_count=2,
                              fit_mode="independent")
    assert revised.name == "fit2" and revised.expected_count == 2
    with pytest.raises(ValueError, match="count"):
        reference_stage("fit8", tables)
    with pytest.raises(ValueError, match="duplicate content"):
        select_stage("duplicates", (tables[0], replace(tables[0], table_id="alias")),
                     question="Can aliases inflate evidence?")


def test_continual_default_requires_previous_but_no_fixed_prior_stage(bind):
    old, new = bind(), bind("new", offset=100)
    with pytest.raises(ValueError, match="nonempty"):
        reference_stage("continual120", [new], expected_count=1)
    selection = reference_stage("continual120", [new], name="continual1", expected_count=1,
                                previous=[old])
    assert selection.previous == (old,)
    assert selection.separation_level == "table"


def test_frozen_corpus_manifest_is_reused_in_place(bind, tmp_path):
    original = bind()
    data = tmp_path / "data"
    data.mkdir()
    (data / "example.json").write_bytes(original.path.read_bytes())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": "tabu.tar.diverse-fit-corpus.1", "split_seed": 23,
        "records": [{"dataset": "example", "data_sha256": original.sha256,
                     "request": {"seed": 17}, "realized": {"world_hash": "world-A"}}],
    }))
    tables = bind_legacy_corpus(manifest, cohort="old120")
    assert tables[0].table_id == "old120/example"
    assert tables[0].provenance.world_id == "world-A"
    assert tables[0].provenance.verify(expected_data_sha256=original.sha256)
    assert tables[0].path == data / "example.json"
    with pytest.raises(ValueError, match="source manifest record"):
        replace(tables[0].provenance, world_id="invented").verify()
    with pytest.raises(ValueError, match="source record"):
        tables[0].provenance.verify(expected_data_sha256="0" * 64)
    with pytest.raises(ValueError, match="resolve exactly once"):
        replace(tables[0].provenance, record_id="missing").verify()
    (data / "example.json").write_text("{}")
    with pytest.raises(ValueError, match="dataset digest"):
        bind_legacy_corpus(manifest, cohort="old120")


def test_world_level_passes_for_distinct_verified_source_records(bind, tmp_path):
    originals = [bind("one"), bind("two", offset=100)]
    data = tmp_path / "data"
    data.mkdir()
    records = []
    for index, original in enumerate(originals):
        (data / original.path.name).write_bytes(original.path.read_bytes())
        records.append({"dataset": original.table_id, "data_sha256": original.sha256,
                        "request": {"seed": index},
                        "realized": {"world_hash": f"world-{index}"}})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": "tabu.tar.diverse-fit-corpus.1",
                                    "split_seed": 23, "records": records}))
    tables = bind_legacy_corpus(manifest, cohort="fixture")
    result = require_disjoint(tables[:1], tables[1:], level="world")
    assert result["world_separation_established"]
    assert not result["unknown_world_ids"]


@pytest.mark.parametrize("kwargs", [
    {"partition": "test"}, {"partition": "validation", "purpose": "training"},
])
def test_reserved_requests_fail_closed(kwargs):
    with pytest.raises(ValueError):
        EpisodeRequest(EpisodeSeeds(1, 2, 3, 4), **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"kind": "structured_damage", "fraction": None},
    {"partition": "test", "purpose": "final_test", "kind": "random_cell"},
    {"partition": "test", "purpose": "final_test", "kind": "supervised_row", "window_rows": 4},
])
def test_recipe_capabilities_belong_to_the_adapter(bind, kwargs):
    from tabu_lab.curriculum.v53_adapter import V53EpisodeFactory

    # These are legal requests for other adapters, not supported V5.3 recipes.
    request = EpisodeRequest(EpisodeSeeds(1, 2, 3, 4), **kwargs)
    with pytest.raises(ValueError, match=r"V5\.3"):
        V53EpisodeFactory()(bind(), request)


def test_adapter_cannot_silently_ignore_extended_recipe_parameters(bind):
    from tabu_lab.curriculum.v53_adapter import V53EpisodeFactory

    @dataclass(frozen=True)
    class GuardedRequest(EpisodeRequest):
        numeric_query_guard: str = "custom-guard"

    with pytest.raises(ValueError, match="extended request"):
        V53EpisodeFactory()(bind(), GuardedRequest(EpisodeSeeds(1, 2, 3, 4)))


def test_adapter_reuses_episode_builder_and_keeps_truth_out_of_inputs(bind):
    import torch

    from tabu_lab.curriculum.v53_adapter import V53EpisodeFactory

    table = bind()
    factory = V53EpisodeFactory()
    request = EpisodeRequest(EpisodeSeeds(1, 2, 3, 4))
    inputs, targets, truth, audit = factory(table, request)
    assert audit["row_ids"] == list(range(6))
    assert "curriculum/v53_adapter.py" in audit["episode_source"]["files"]
    assert audit["episode_request"]["purpose"] == "training"
    assert all(r < 6 for r, _ in audit["query_addresses"])
    for a, values in enumerate(inputs.values):
        assert torch.equal(values[inputs.query[:, a]], torch.zeros_like(values[inputs.query[:, a]]))
    again = factory(table, request)
    assert again[3] == audit
    assert torch.equal(again[1].targets, targets.targets)
    assert all(torch.equal(a, b) for a, b in zip(again[0].values, inputs.values, strict=True))
    assert truth is not inputs


def test_reserved_label_intervention_changes_only_scorer_truth(bind):
    import torch

    from tabu_lab.curriculum.v53_adapter import V53EpisodeFactory

    table = bind()
    payload = json.loads(table.path.read_text())
    changed = copy.deepcopy(payload)
    for row in changed["splits"]["test"]:
        changed["values"][row][-1] += 1e9
    # Keep the table ID fixed so this intervenes on truth, not episode RNG.
    replacement = replace(bind("changed", payload=changed), table_id=table.table_id,
                          cohort=table.cohort)
    request = EpisodeRequest(EpisodeSeeds(1, 2, 3, 4), kind="supervised_row",
                             partition="test", purpose="retrospective")
    factory = V53EpisodeFactory()
    left, right = factory(table, request), factory(replacement, request)
    assert all(torch.equal(a, b) for a, b in zip(left[0].values, right[0].values, strict=True))
    assert torch.equal(left[0].visible, right[0].visible)
    assert torch.equal(left[0].query, right[0].query)
    assert torch.equal(left[1].targets, right[1].targets)
    assert not torch.equal(left[2].values[-1], right[2].values[-1])
    assert left[3]["transductive"]
    assert left[3]["query_addresses"] == [[8, 2], [9, 2]]


def test_training_episode_ignores_reserved_label_intervention(bind):
    import torch

    from tabu_lab.curriculum.v53_adapter import V53EpisodeFactory

    old = bind()
    payload = json.loads(old.path.read_text())
    for row in [6, 7, 8, 9]:
        payload["values"][row][-1] += 1e9
    changed = replace(bind("changed", payload=payload), table_id=old.table_id, cohort=old.cohort)
    request = EpisodeRequest(EpisodeSeeds(1, 2, 3, 4))
    factory = V53EpisodeFactory()
    left, right = factory(old, request), factory(changed, request)
    for slot in (0, 2):
        assert all(torch.equal(a, b) for a, b in zip(left[slot].values, right[slot].values,
                                                   strict=True))
