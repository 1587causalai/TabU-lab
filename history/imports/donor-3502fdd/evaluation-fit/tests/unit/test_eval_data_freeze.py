from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

import tabu_lab.adapters.eval_data_freeze as freeze
from tabu_lab.adapters.eval_data_freeze import (
    ADULT_ROWID_SEMANTICS,
    EvalDataFreezeError,
    build_adult_freeze,
    build_diabetes_freeze,
    build_karate_freeze,
    build_movielens_freeze,
    load_freeze_bundle,
    main,
    verify_freeze_bundle,
    write_freeze_bundle,
)
from tabu_lab.adapters.eval_data_workflow import (
    EvalDataPreparationRequest,
    prepare_eval_data_bundle,
)
from tabu_lab.adapters.real_eval_data import DelimitedTableAuthority, MovieLensAuthority


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _pin(monkeypatch: pytest.MonkeyPatch, role: str, name: str, content: bytes) -> None:
    original = freeze._PINS[role]
    monkeypatch.setitem(
        freeze._PINS,
        role,
        freeze._PinnedSource(
            retained_name=name,
            sha256=_sha256(content),
            size_bytes=len(content),
            media_type=original.media_type,
        ),
    )


def _diabetes_sources() -> tuple[bytes, bytes]:
    rows: list[str] = []
    targets: list[str] = []
    for row_id in range(442):
        rows.append(
            " ".join(
                str(((row_id + 1) * (feature + 3)) % 101 + 0.125)
                for feature in range(10)
            )
        )
        targets.append(str((row_id * 13) % 311 + 1.0))
    data = gzip.compress(("\n".join(rows) + "\n").encode("ascii"), mtime=0)
    target = gzip.compress(("\n".join(targets) + "\n").encode("ascii"), mtime=0)
    return data, target


def _build_diabetes_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    data, target = _diabetes_sources()
    _pin(monkeypatch, "diabetes_data", "diabetes-data-test.csv.gz", data)
    _pin(monkeypatch, "diabetes_target", "diabetes-target-test.csv.gz", target)
    data_path = tmp_path / "diabetes-data.csv.gz"
    target_path = tmp_path / "diabetes-target.csv.gz"
    data_path.write_bytes(data)
    target_path.write_bytes(target)
    return build_diabetes_freeze(
        data_source=data_path,
        target_source=target_path,
        split_seed=1729,
        mask_seed=2718,
    )


def test_diabetes_freeze_is_deterministic_exhaustive_and_explicitly_unreviewed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _build_diabetes_candidate(tmp_path, monkeypatch)
    second = _build_diabetes_candidate(tmp_path, monkeypatch)

    assert first == second
    manifest = verify_freeze_bundle(first)
    assert manifest.authority_status == "self_consistent_unreviewed"
    assert manifest.publication_eligible is False
    assert manifest.review_ids == ()
    assert manifest.network_access is False
    assert manifest.decisions["representation"] == (
        "raw_unscaled_10_feature_csv_with_zero_based_row_id_v1"
    )
    assert {item.role for item in manifest.outputs} == {
        "canonical_retained_source",
        "completion_request",
        "supervised_request",
    }

    retained = first.files["retained/sklearn-diabetes-1.9.0-raw.csv"]
    assert retained.startswith(b"row_id,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,outcome\n")
    assert len(retained.decode("utf-8").splitlines()) == 443
    for path in (
        "requests/diabetes-supervised.request.json",
        "requests/diabetes-completion.request.json",
    ):
        request = EvalDataPreparationRequest.model_validate_json(first.files[path])
        assert isinstance(request.authority, DelimitedTableAuthority)
        partitions = request.authority.split.partitions
        assigned = [row_id for values in partitions.values() for row_id in values]
        assert {key: len(value) for key, value in partitions.items()} == {
            "train": 256,
            "validation": 64,
            "test": 122,
        }
        assert len(assigned) == len(set(assigned)) == 442
        assert set(assigned) == {str(value) for value in range(442)}
        assert request.source_sha256 == _sha256(retained)

    serialized = json.dumps(manifest.model_dump(mode="json"), sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "self_consistent_unreviewed" in serialized
    assert '"publication_eligible": false' in serialized


def test_base_suite_version_emits_distinct_v1_scenario_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, target = _diabetes_sources()
    _pin(monkeypatch, "diabetes_data", "diabetes-data-test.csv.gz", data)
    _pin(monkeypatch, "diabetes_target", "diabetes-target-test.csv.gz", target)
    data_path = tmp_path / "diabetes-data.csv.gz"
    target_path = tmp_path / "diabetes-target.csv.gz"
    data_path.write_bytes(data)
    target_path.write_bytes(target)

    bundle = build_diabetes_freeze(
        data_source=data_path,
        target_source=target_path,
        split_seed=1729,
        mask_seed=2718,
        suite_version="v1",
    )
    supervised = EvalDataPreparationRequest.model_validate_json(
        bundle.files["requests/diabetes-supervised.request.json"]
    )
    completion = EvalDataPreparationRequest.model_validate_json(
        bundle.files["requests/diabetes-completion.request.json"]
    )
    assert supervised.suite_id == "table-supervised-micro-v1"
    assert supervised.scenario_id == "sklearn-diabetes-regression-micro-base"
    assert completion.suite_id == "table-completion-micro-v1"
    assert completion.scenario_id == "sklearn-diabetes-feature-completion-micro-base"


def test_tabubase_v1_scenario_aliases_use_the_same_closed_authority_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_diabetes_candidate(tmp_path, monkeypatch)
    supervised = EvalDataPreparationRequest.model_validate_json(
        bundle.files["requests/diabetes-supervised.request.json"]
    )
    completion = EvalDataPreparationRequest.model_validate_json(
        bundle.files["requests/diabetes-completion.request.json"]
    )

    supervised_payload = supervised.model_dump(mode="python")
    supervised_payload.update(
        suite_id="table-supervised-micro-v1",
        scenario_id="sklearn-diabetes-regression-micro-base",
    )
    completion_payload = completion.model_dump(mode="python")
    completion_payload.update(
        suite_id="table-completion-micro-v1",
        scenario_id="sklearn-diabetes-feature-completion-micro-base",
    )

    supervised_v1 = EvalDataPreparationRequest.model_validate(supervised_payload)
    completion_v1 = EvalDataPreparationRequest.model_validate(completion_payload)
    assert supervised_v1.suite_id == "table-supervised-micro-v1"
    assert completion_v1.suite_id == "table-completion-micro-v1"

    retained = tmp_path / "diabetes-raw.csv"
    retained.write_bytes(bundle.files["retained/sklearn-diabetes-1.9.0-raw.csv"])
    supervised_bundle = prepare_eval_data_bundle(
        request=supervised_v1,
        source=retained,
    )
    completion_bundle = prepare_eval_data_bundle(
        request=completion_v1,
        source=retained,
    )
    assert supervised_bundle.request.scenario_id == "sklearn-diabetes-regression-micro-base"
    assert completion_bundle.request.scenario_id == "sklearn-diabetes-feature-completion-micro-base"


def test_freeze_write_is_create_once_hash_bound_and_allows_follow_on_private_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_diabetes_candidate(tmp_path, monkeypatch)
    root = tmp_path / "candidate"
    manifest_path = write_freeze_bundle(bundle, root)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert write_freeze_bundle(bundle, root) == manifest_path
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert load_freeze_bundle(root) == bundle

    follow_on = root / "private" / "prepared.json"
    follow_on.parent.mkdir()
    follow_on.write_text("not-part-of-freeze\n", encoding="utf-8")
    assert load_freeze_bundle(root) == bundle

    request_path = root / "requests" / "diabetes-supervised.request.json"
    request_path.write_bytes(request_path.read_bytes() + b"tamper")
    with pytest.raises(EvalDataFreezeError, match="drifted"):
        load_freeze_bundle(root)
    with pytest.raises(FileExistsError, match="already differs"):
        write_freeze_bundle(bundle, root)


def test_freeze_check_cli_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bundle = _build_diabetes_candidate(tmp_path, monkeypatch)
    root = tmp_path / "candidate"
    write_freeze_bundle(bundle, root)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert main(["check", "--output-root", str(root)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["freeze_id"] == bundle.manifest.freeze_id
    assert payload["authority_status"] == "self_consistent_unreviewed"
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_freeze_bundle_must_be_git_ignored_inside_a_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_diabetes_candidate(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    (repo / ".gitignore").write_text(".private/\n", encoding="utf-8")

    with pytest.raises(EvalDataFreezeError, match="Git-ignored"):
        write_freeze_bundle(bundle, repo / "unignored")
    assert not (repo / "unignored").exists()

    destination = repo / ".private" / "diabetes"
    write_freeze_bundle(bundle, destination)
    assert load_freeze_bundle(destination) == bundle


def test_freeze_output_rejects_root_and_ancestor_symlink_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_diabetes_candidate(tmp_path, monkeypatch)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "--quiet", str(repo)], check=True)
    unignored = repo / "unignored"
    unignored.mkdir()

    outside = tmp_path / "outside"
    outside.mkdir()
    redirected_parent = outside / "redirected-parent"
    redirected_parent.symlink_to(unignored, target_is_directory=True)
    redirected_output = redirected_parent / "candidate"
    with pytest.raises(EvalDataFreezeError, match="symlink component"):
        write_freeze_bundle(bundle, redirected_output)
    assert not (unignored / "candidate").exists()

    target = tmp_path / "target"
    target.mkdir()
    root_link = tmp_path / "root-link"
    root_link.symlink_to(target, target_is_directory=True)
    with pytest.raises(EvalDataFreezeError, match="symlink component"):
        write_freeze_bundle(bundle, root_link)
    assert not (target / "freeze-manifest.json").exists()


def _movielens_zip() -> bytes:
    pairs: set[tuple[int, int]] = set()
    pairs.update((user, item) for user in range(1, 65) for item in range(1, 1001))
    pairs.update((user, item) for user in range(65, 944) for item in range(1, 21))
    pairs.update((1 + (item % 64), item) for item in range(1001, 1683))
    for item in range(21, 129):
        for user in range(65, 944):
            if len(pairs) == 100_000:
                break
            pairs.add((user, item))
        if len(pairs) == 100_000:
            break
    assert len(pairs) == 100_000
    grid = sorted((user, item) for user, item in pairs if user <= 64 and item <= 128)
    test_grid = set(grid[:200])
    remainder = sorted(pairs - test_grid)
    test_pairs = test_grid | set(remainder[-19_800:])
    base_pairs = pairs - test_pairs
    assert len(base_pairs) == 80_000 and len(test_pairs) == 20_000

    def encoded(rows: set[tuple[int, int]]) -> bytes:
        return "".join(
            f"{user}\t{item}\t{((user + item) % 5) + 1}\t{1_000_000_000 + index}\n"
            for index, (user, item) in enumerate(sorted(rows))
        ).encode("ascii")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("ml-100k/u1.base", encoded(base_pairs))
        archive.writestr("ml-100k/u1.test", encoded(test_pairs))
    return buffer.getvalue()


def test_movielens_freeze_binds_train_side_validation_carve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = _movielens_zip()
    _pin(monkeypatch, "movielens", "ml-100k-test.zip", content)
    source = tmp_path / "ml-100k.zip"
    source.write_bytes(content)

    bundle = build_movielens_freeze(
        zip_source=source,
        validation_seed=31415,
        validation_count=8_000,
    )
    request = EvalDataPreparationRequest.model_validate_json(
        bundle.files["requests/movielens.request.json"]
    )
    assert isinstance(request.authority, MovieLensAuthority)
    assert len(request.authority.validation_interaction_ids) == 8_000
    assert len(set(request.authority.validation_interaction_ids)) == 8_000
    assert request.authority.base_member == "ml-100k/u1.base"
    assert request.authority.test_member == "ml-100k/u1.test"
    assert bundle.manifest.decisions["validation_recipe"] == (
        "tabu.eval-movielens-validation-rank.v1"
    )
    assert bundle.manifest.publication_eligible is False


def test_karate_freeze_is_lock_pinned_and_materializer_validated() -> None:
    first = build_karate_freeze(split_seed=1729)
    second = build_karate_freeze(split_seed=1729)

    assert first == second
    request = EvalDataPreparationRequest.model_validate_json(
        first.files["requests/karate.request.json"]
    )
    partitions = request.authority.split.partitions
    assert {key: len(value) for key, value in partitions.items()} == {
        "train": 20,
        "validation": 7,
        "test": 7,
    }
    retained = json.loads(first.files["retained/zachary-karate-networkx-3.6.1.json"])
    assert len(retained["nodes"]) == 34
    assert len(retained["edges"]) == 78
    assert all(set(node["features"]) == {"degree"} for node in retained["nodes"])
    assert first.manifest.decisions["node_feature_contract"] == "unweighted_degree_v1"


def _task_split_arff(row_count: int = 20) -> bytes:
    output = [
        "@relation adult_splits",
        "@attribute type {TRAIN,TEST}",
        "@attribute rowid numeric",
        "@attribute repeat numeric",
        "@attribute fold numeric",
        "@data",
    ]
    for fold in range(10):
        test = {fold * 2 % row_count, (fold * 2 + 1) % row_count}
        for row_id in range(row_count):
            split_type = "TEST" if row_id in test else "TRAIN"
            output.append(f"{split_type},{row_id},0,{fold}")
    return ("\n".join(output) + "\n").encode("utf-8")


def test_openml_task_fold_parser_is_exhaustive_and_validation_is_train_side() -> None:
    row_ids = tuple(str(value) for value in range(20))
    partitions = freeze._adult_task_fold(
        _task_split_arff(),
        fold=3,
        row_ids=row_ids,
        validation_seed=1729,
    )
    assert {key: len(value) for key, value in partitions.items()} == {
        "train": 16,
        "validation": 2,
        "test": 2,
    }
    assigned = [row_id for values in partitions.values() for row_id in values]
    assert len(assigned) == len(set(assigned)) == 20
    assert set(partitions["test"]) == {"6", "7"}
    assert not set(partitions["validation"]) & set(partitions["test"])


def test_adult_freeze_fails_closed_before_reading_without_fold_semantics_and_license(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(EvalDataFreezeError) as error:
        build_adult_freeze(
            data_source=tmp_path / "missing-data.arff",
            task_split_source=tmp_path / "missing-splits.arff",
            license_evidence=None,
            fold=None,
            rowid_semantics=None,
            validation_seed=1729,
            mask_seed=1729,
        )
    message = str(error.value)
    assert "explicit OpenML task fold" in message
    assert ADULT_ROWID_SEMANTICS in message
    assert "license evidence" in message

    output = tmp_path / "must-not-exist"
    assert (
        main(
            [
                "adult",
                "--data-arff",
                str(tmp_path / "missing-data.arff"),
                "--task-splits-arff",
                str(tmp_path / "missing-splits.arff"),
                "--validation-seed",
                "1729",
                "--mask-seed",
                "1729",
                "--output-root",
                str(output),
            ]
        )
        == 2
    )
    stderr = capsys.readouterr().err
    assert "Adult authority freeze is blocked" in stderr
    assert not output.exists()


def test_task_split_parser_rejects_non_exhaustive_fold() -> None:
    text = _task_split_arff().decode("utf-8")
    corrupted = text.replace("TEST,0,0,0\n", "", 1).encode("utf-8")
    with pytest.raises(EvalDataFreezeError, match="not an exhaustive disjoint assignment"):
        freeze._adult_task_fold(
            corrupted,
            fold=0,
            row_ids=tuple(str(value) for value in range(20)),
            validation_seed=1729,
        )


def test_manifest_rejects_review_or_publication_promotion_by_edit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _build_diabetes_candidate(tmp_path, monkeypatch)
    payload = bundle.manifest.model_dump(mode="json")
    payload["review_ids"] = ["self-review"]
    with pytest.raises(ValueError, match="candidate freeze cannot carry review ids"):
        freeze.EvalDataAuthorityFreezeManifest.model_validate(payload)
    payload = bundle.manifest.model_dump(mode="json")
    payload["publication_eligible"] = True
    with pytest.raises(ValueError):
        freeze.EvalDataAuthorityFreezeManifest.model_validate(payload)


def test_source_pin_rejects_one_byte_drift_before_any_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, target = _diabetes_sources()
    _pin(monkeypatch, "diabetes_data", "diabetes-data-test.csv.gz", data)
    _pin(monkeypatch, "diabetes_target", "diabetes-target-test.csv.gz", target)
    data_path = tmp_path / "data.csv.gz"
    target_path = tmp_path / "target.csv.gz"
    data_path.write_bytes(data + b"tamper")
    target_path.write_bytes(target)
    with pytest.raises(EvalDataFreezeError, match="size/SHA-256 pin"):
        build_diabetes_freeze(
            data_source=data_path,
            target_source=target_path,
            split_seed=1729,
            mask_seed=1729,
        )


def test_karate_locality_pair_never_touches_base() -> None:
    bundle = build_karate_freeze(split_seed=2718)
    request = EvalDataPreparationRequest.model_validate_json(
        bundle.files["requests/karate.request.json"]
    )
    perturbations = request.authority.perturbations
    assert perturbations.base_node_id in request.authority.split.partitions["test"]
    assert perturbations.base_node_id in perturbations.topology_toggle_edge
    assert perturbations.base_node_id not in perturbations.locality_toggle_edge


def test_split_rank_changes_with_seed_without_changing_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data, target = _diabetes_sources()
    _pin(monkeypatch, "diabetes_data", "diabetes-data-test.csv.gz", data)
    _pin(monkeypatch, "diabetes_target", "diabetes-target-test.csv.gz", target)
    data_path = tmp_path / "data.csv.gz"
    target_path = tmp_path / "target.csv.gz"
    data_path.write_bytes(data)
    target_path.write_bytes(target)
    first = build_diabetes_freeze(
        data_source=data_path,
        target_source=target_path,
        split_seed=11,
        mask_seed=17,
    )
    second = build_diabetes_freeze(
        data_source=data_path,
        target_source=target_path,
        split_seed=12,
        mask_seed=17,
    )
    source_path = "retained/sklearn-diabetes-1.9.0-raw.csv"
    assert first.files[source_path] == second.files[source_path]
    first_request = EvalDataPreparationRequest.model_validate_json(
        first.files["requests/diabetes-supervised.request.json"]
    )
    second_request = EvalDataPreparationRequest.model_validate_json(
        second.files["requests/diabetes-supervised.request.json"]
    )
    assert first_request.authority.split.partitions != second_request.authority.split.partitions
    assert first.manifest.freeze_id != second.manifest.freeze_id


def test_movielens_validation_ids_are_stably_sorted() -> None:
    ids = ("10:2", "2:10", "2:3")
    ordered = tuple(sorted(ids, key=lambda value: tuple(int(item) for item in value.split(":"))))
    assert ordered == ("2:3", "2:10", "10:2")


def test_candidate_freeze_contains_no_cross_task_composite_claim() -> None:
    bundle = build_karate_freeze(split_seed=31415)
    text = json.dumps(bundle.manifest.model_dump(mode="json"), sort_keys=True)
    for forbidden in ("composite_score", "foundation_model", "supported_model"):
        assert forbidden not in text


def test_task_split_fixture_has_ten_exhaustive_folds() -> None:
    _, rows = freeze._parse_arff(_task_split_arff(), role="test task split")
    assert len(rows) == 200
    counts = {
        fold: sum(int(row[3]) == fold for row in rows)
        for fold in range(10)
    }
    assert set(counts.values()) == {20}


def test_movielens_fixture_has_full_official_cardinalities() -> None:
    content = _movielens_zip()
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        base = archive.read("ml-100k/u1.base").splitlines()
        test = archive.read("ml-100k/u1.test").splitlines()
    assert len(base) == 80_000
    assert len(test) == 20_000
    pairs = {
        tuple(line.decode("ascii").split("\t")[:2])
        for line in itertools.chain(base, test)
    }
    assert len(pairs) == 100_000
