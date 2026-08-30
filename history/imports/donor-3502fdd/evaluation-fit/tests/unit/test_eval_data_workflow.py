from __future__ import annotations

import hashlib
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import tabu_lab.cli as cli_module
import tabu_lab.evaluation.foundry as foundry_module
from tabu_lab.adapters.eval_data_workflow import (
    EvalDataPreparationRequest,
    EvalDataWorkflowError,
    PreparedEvalDataBundle,
    check_prepared_eval_bundle,
    load_prepared_eval_bundle,
    prepare_and_write_eval_data,
    prepare_eval_data_bundle,
    register_prepared_eval_bundle,
)
from tabu_lab.adapters.real_eval_data import (
    ColumnAuthority,
    CompletionMaskAuthority,
    DelimitedTableAuthority,
    GraphPerturbationAuthority,
    KarateAuthority,
    MovieLensAuthority,
    SplitAuthority,
)
from tabu_lab.catalog import DatasetAuthorityStatus
from tabu_lab.cli import _load_prepared_scenario_for_cli, main
from tabu_lab.contracts import canonical_hash
from tabu_lab.evaluation.foundry import (
    DatasetSnapshotBinding,
    PreparationContract,
    PreparedExample,
    PreparedScenario,
    SourceMaterial,
    TargetKind,
    load_suite,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _diabetes_bytes() -> bytes:
    output = io.StringIO(newline="")
    features = tuple(f"x{index}" for index in range(10))
    output.write("row_id," + ",".join(features) + ",outcome\n")
    for row_id in range(442):
        values = [str(((row_id * (index + 3)) % 97) / 7.0) for index in range(10)]
        outcome = str(((row_id * 13) % 101) + 0.25)
        output.write(f"{row_id}," + ",".join(values) + f",{outcome}\n")
    return output.getvalue().encode("utf-8")


def _split(
    *,
    dataset_id: str,
    source_version: str,
    content: bytes,
    partitions: dict[str, tuple[str, ...]],
) -> SplitAuthority:
    return SplitAuthority(
        authority_id=f"{dataset_id}-offline-test-split",
        dataset_id=dataset_id,
        source_version=source_version,
        source_sha256=_sha256(content),
        stable_id_kind="decimal_integer",
        partitions=partitions,
    )


def _diabetes_request(content: bytes) -> EvalDataPreparationRequest:
    scenario = load_suite("table-supervised-micro-v0").scenarios[1]
    features = tuple(f"x{index}" for index in range(10))
    authority = DelimitedTableAuthority(
        delimiter=",",
        field_whitespace="preserve",
        header=("row_id", *features, "outcome"),
        row_id_column="row_id",
        feature_columns=tuple(
            ColumnAuthority(source_name=name, family_id=name, kind="numeric") for name in features
        ),
        response_column=ColumnAuthority(
            source_name="outcome",
            family_id="outcome",
            kind="numeric",
        ),
        split=_split(
            dataset_id=scenario.dataset.dataset_id,
            source_version=scenario.dataset.source_version,
            content=content,
            partitions={
                "train": tuple(str(value) for value in range(0, 256)),
                "validation": tuple(str(value) for value in range(256, 320)),
                "test": tuple(str(value) for value in range(320, 442)),
            },
        ),
    )
    return EvalDataPreparationRequest(
        suite_id="table-supervised-micro-v0",
        scenario_id=scenario.scenario_id,
        source_sha256=_sha256(content),
        source_size_bytes=len(content),
        source_media_type="text/csv",
        authority=authority,
    )


def _write_request(path: Path, request: EvalDataPreparationRequest) -> None:
    path.write_text(
        json.dumps(request.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_diabetes(tmp_path: Path) -> tuple[Path, Path, PreparedEvalDataBundle]:
    content = _diabetes_bytes()
    source = tmp_path / "diabetes-retained.csv"
    source.write_bytes(content)
    request = _diabetes_request(content)
    request_path = tmp_path / "request.json"
    _write_request(request_path, request)
    bundle_path = tmp_path / "private" / "prepared.json"
    bundle, written = prepare_and_write_eval_data(
        request_path=request_path,
        source=source,
        destination=bundle_path,
    )
    assert written == bundle_path
    return source, bundle_path, bundle


def test_prepare_bundle_is_direct_evaluator_cli_input(tmp_path: Path) -> None:
    _, bundle_path, bundle = _prepare_diabetes(tmp_path)
    suite = load_suite(bundle.request.suite_id)

    prepared = _load_prepared_scenario_for_cli(
        str(bundle_path),
        suite=suite,
        expected_scenario_id=bundle.request.scenario_id,
    )

    assert prepared == bundle.prepared


def test_cli_dry_run_passes_verified_prepared_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, bundle_path, bundle = _prepare_diabetes(tmp_path)
    suite = load_suite(bundle.request.suite_id)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        cli_module,
        "_load_eval_suite_for_cli",
        lambda *args, **kwargs: suite,
    )

    def fake_dry_run(resolved_suite, *, prepared):
        observed["suite"] = resolved_suite
        observed["prepared"] = prepared
        return SimpleNamespace(
            ready=True,
            model_dump=lambda mode: {"ready": True},
        )

    monkeypatch.setattr(foundry_module, "dry_run_suite", fake_dry_run)

    assert (
        cli_module._eval_dry_run(
            suite.suite_id,
            prepared_paths=(str(bundle_path),),
            directory=None,
            catalog_path="unused-by-fixture.json",
            as_json=True,
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == {"ready": True}
    assert observed["suite"] == suite
    assert observed["prepared"] == {bundle.prepared.scenario_id: bundle.prepared}


def _init_git_worktree(path: Path) -> Path:
    path.mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    (path / ".gitignore").write_text(
        ".local-runs/\nevaluations/data/private/\nevaluations/data/retained/\n",
        encoding="utf-8",
    )
    return path


def test_prepare_rejects_unignored_retained_source_inside_active_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    content = _diabetes_bytes()
    source = repo / "retained-source.csv"
    source.write_bytes(content)
    request_path = repo / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = repo / ".local-runs" / "eval-data" / "prepared.json"

    with pytest.raises(EvalDataWorkflowError, match=r"retained source.*Git-ignored"):
        prepare_and_write_eval_data(
            request_path=request_path,
            source=source,
            destination=bundle_path,
        )
    assert not bundle_path.exists()


@pytest.mark.parametrize("target_location", ("outside_repo", "ignored_in_repo"))
def test_prepare_rejects_repo_symlinked_retained_source_without_following_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_location: str,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    content = _diabetes_bytes()
    if target_location == "outside_repo":
        target = tmp_path / "outside-retained-source.csv"
    else:
        target = repo / "evaluations" / "data" / "retained" / "source.csv"
        target.parent.mkdir(parents=True)
    target.write_bytes(content)
    source_link = repo / "retained-source-link.csv"
    source_link.symlink_to(target)
    request_path = repo / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = repo / ".local-runs" / "eval-data" / "prepared.json"

    with pytest.raises(EvalDataWorkflowError, match=r"retained source.*symlink"):
        prepare_and_write_eval_data(
            request_path=request_path,
            source=source_link,
            destination=bundle_path,
        )

    assert target.read_bytes() == content
    assert not bundle_path.exists()


def test_prepare_rejects_repo_symlinked_source_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    content = _diabetes_bytes()
    outside = tmp_path / "outside-retained"
    outside.mkdir()
    target = outside / "source.csv"
    target.write_bytes(content)
    linked_directory = repo / "evaluations" / "data" / "retained"
    linked_directory.parent.mkdir(parents=True)
    linked_directory.symlink_to(outside, target_is_directory=True)
    source = linked_directory / "source.csv"
    request_path = repo / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = repo / ".local-runs" / "eval-data" / "prepared.json"

    with pytest.raises(EvalDataWorkflowError, match=r"retained source.*symlink"):
        prepare_and_write_eval_data(
            request_path=request_path,
            source=source,
            destination=bundle_path,
        )

    assert target.read_bytes() == content
    assert not bundle_path.exists()


def test_prepare_rejects_unignored_private_bundle_inside_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    content = _diabetes_bytes()
    source = tmp_path / "external-retained-source.csv"
    source.write_bytes(content)
    request_path = repo / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = repo / "private" / "prepared.json"

    with pytest.raises(
        EvalDataWorkflowError,
        match=r"private prepared bundle.*Git-ignored",
    ):
        prepare_and_write_eval_data(
            request_path=request_path,
            source=source,
            destination=bundle_path,
        )
    assert not bundle_path.exists()


def test_register_rejects_unignored_private_bundle_inside_git_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    external = tmp_path / "external"
    external.mkdir()
    _, safe_bundle_path, _ = _prepare_diabetes(external)
    bundle_path = repo / "prepared.json"
    bundle_path.write_bytes(safe_bundle_path.read_bytes())
    snapshot_path = repo / "datasets" / "diabetes.json"

    with pytest.raises(
        EvalDataWorkflowError,
        match=r"private prepared bundle.*Git-ignored",
    ):
        register_prepared_eval_bundle(
            bundle_path=bundle_path,
            destination=snapshot_path,
        )
    assert not snapshot_path.exists()


@pytest.mark.parametrize("target_location", ("outside_repo", "ignored_in_repo"))
def test_register_rejects_repo_symlinked_private_bundle_without_following_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_location: str,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    external = tmp_path / "external"
    external.mkdir()
    _, safe_bundle_path, _ = _prepare_diabetes(external)
    if target_location == "outside_repo":
        target = safe_bundle_path
    else:
        target = repo / "evaluations" / "data" / "private" / "target.json"
        target.parent.mkdir(parents=True)
        target.write_bytes(safe_bundle_path.read_bytes())
    bundle_link = repo / "evaluations" / "data" / "private" / "prepared-link.json"
    bundle_link.parent.mkdir(parents=True, exist_ok=True)
    bundle_link.symlink_to(target)
    snapshot_path = repo / "datasets" / "diabetes.json"

    with pytest.raises(EvalDataWorkflowError, match=r"private prepared bundle.*symlink"):
        register_prepared_eval_bundle(
            bundle_path=bundle_link,
            destination=snapshot_path,
        )

    assert target.is_file()
    assert not snapshot_path.exists()


def test_ignored_repo_inputs_are_allowed_and_public_snapshot_stays_trackable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    content = _diabetes_bytes()
    source = repo / "evaluations" / "data" / "retained" / "source.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(content)
    request_path = repo / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = repo / "evaluations" / "data" / "private" / "prepared.json"

    _, written = prepare_and_write_eval_data(
        request_path=request_path,
        source=source,
        destination=bundle_path,
    )
    assert written == bundle_path

    snapshot_path = repo / "datasets" / "diabetes.json"
    _, _, registered = register_prepared_eval_bundle(
        bundle_path=bundle_path,
        destination=snapshot_path,
    )
    assert registered == snapshot_path
    assert snapshot_path.is_file()
    ignored = subprocess.run(
        ["git", "-C", str(repo), "check-ignore", "--quiet", "--", "datasets/diabetes.json"],
        check=False,
    )
    assert ignored.returncode == 1


def test_repo_external_source_and_private_output_are_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _init_git_worktree(tmp_path / "repo")
    monkeypatch.chdir(repo)
    external = tmp_path / "external-eval-data"
    external.mkdir()
    content = _diabetes_bytes()
    source = external / "source.csv"
    source.write_bytes(content)
    request_path = repo / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = external / "private" / "prepared.json"

    _, written = prepare_and_write_eval_data(
        request_path=request_path,
        source=source,
        destination=bundle_path,
    )
    assert written == bundle_path


def test_repository_gitignore_protects_private_eval_paths_and_wandb() -> None:
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "wandb/offline-run-test/files/config.yaml",
        "evaluations/data/private/example.json",
        "evaluations/data/retained/example.csv",
    ):
        result = subprocess.run(
            ["git", "-C", str(root), "check-ignore", "--quiet", "--", relative],
            check=False,
        )
        assert result.returncode == 0


def test_diabetes_prepare_register_check_is_offline_hash_bound_and_public_safe(
    tmp_path: Path,
) -> None:
    source, bundle_path, bundle = _prepare_diabetes(tmp_path)

    loaded = load_prepared_eval_bundle(bundle_path)
    assert loaded == bundle
    assert bundle.request_sha256 == bundle.request.content_hash
    assert bundle.authority_sha256 == bundle.request.authority_sha256
    assert bundle.source_sha256 == _sha256(source.read_bytes())
    assert bundle.prepared_sha256 == bundle.prepared.content_hash
    assert bundle.prepared.source_material.content_bytes == source.read_bytes()
    bundle_text = bundle_path.read_text(encoding="utf-8")
    assert "private_evaluator_input" in bundle_text
    assert "content_base64" in bundle_text
    assert str(tmp_path) not in bundle_text

    snapshot_path = tmp_path / "datasets" / "diabetes.json"
    _, snapshot, written = register_prepared_eval_bundle(
        bundle_path=bundle_path,
        destination=snapshot_path,
    )
    assert written == snapshot_path
    assert snapshot.source_sha256 == bundle.source_sha256
    assert snapshot.content_sha256 == bundle.prepared_sha256
    assert snapshot.split_manifest_sha256 == bundle.prepared.binding.split_sha256
    assert snapshot.truth_sidecar_sha256 == bundle.prepared.binding.truth_sidecar_sha256
    assert snapshot.schema_version == "tabu.dataset-snapshot.v3"
    assert snapshot.request_sha256 == bundle.request_sha256
    assert snapshot.authority_sha256 == bundle.authority_sha256
    assert snapshot.authority_status is DatasetAuthorityStatus.SELF_CONSISTENT_UNREVIEWED
    assert snapshot.review_ids == ()
    assert not snapshot.publication_eligible
    public_text = snapshot_path.read_text(encoding="utf-8")
    public_payload = json.loads(public_text)
    assert public_payload["request_sha256"] == bundle.request_sha256
    assert public_payload["authority_sha256"] == bundle.authority_sha256
    assert public_payload["authority_status"] == "self_consistent_unreviewed"
    assert "content_base64" not in public_text
    assert "private_evaluator_input" not in public_text
    assert str(tmp_path) not in public_text

    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (bundle_path, snapshot_path)
    }
    report = check_prepared_eval_bundle(
        bundle_path=bundle_path,
        snapshot_path=snapshot_path,
    )
    assert report.valid
    assert report.snapshot_matches is True
    assert report.dataset_snapshot_id == snapshot.dataset_snapshot_id
    after = {
        path: (path.read_bytes(), path.stat().st_mtime_ns) for path in (bundle_path, snapshot_path)
    }
    assert after == before


def test_workflow_rejects_source_bundle_and_snapshot_tamper(tmp_path: Path) -> None:
    source, bundle_path, bundle = _prepare_diabetes(tmp_path)
    tampered_source = tmp_path / "tampered.csv"
    tampered_source.write_bytes(source.read_bytes() + b"tamper")
    with pytest.raises(EvalDataWorkflowError, match="source_sha256"):
        prepare_eval_data_bundle(request=bundle.request, source=tampered_source)

    tampered_bundle_path = tmp_path / "tampered-bundle.json"
    tampered_bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    tampered_bundle["prepared_sha256"] = "0" * 64
    tampered_bundle_path.write_text(json.dumps(tampered_bundle), encoding="utf-8")
    with pytest.raises(ValueError, match="prepared_sha256"):
        load_prepared_eval_bundle(tampered_bundle_path)

    snapshot_path = tmp_path / "snapshot.json"
    register_prepared_eval_bundle(
        bundle_path=bundle_path,
        destination=snapshot_path,
    )
    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_payload["license_id"] = "tampered-license"
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    with pytest.raises(EvalDataWorkflowError, match="differs"):
        check_prepared_eval_bundle(
            bundle_path=bundle_path,
            snapshot_path=snapshot_path,
        )


def test_cli_eval_data_rejects_tamper_and_overwrite(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    content = _diabetes_bytes()
    source = tmp_path / "source.csv"
    source.write_bytes(content)
    request_path = tmp_path / "request.json"
    _write_request(request_path, _diabetes_request(content))
    bundle_path = tmp_path / "prepared.json"
    snapshot_path = tmp_path / "snapshot.json"

    prepare_args = [
        "eval",
        "data",
        "prepare",
        str(request_path),
        "--source",
        str(source),
        "--output",
        str(bundle_path),
        "--json",
    ]
    assert main(prepare_args) == 0
    prepare_payload = json.loads(capsys.readouterr().out)
    assert prepare_payload["prepared"] is True
    assert main(prepare_args) == 0  # byte-identical retry is idempotent
    capsys.readouterr()
    clean_bundle_path = tmp_path / "prepared-clean.json"
    clean_bundle_path.write_bytes(bundle_path.read_bytes())

    assert (
        main(
            [
                "eval",
                "data",
                "register",
                str(bundle_path),
                "--output",
                str(snapshot_path),
                "--json",
            ]
        )
        == 0
    )
    register_payload = json.loads(capsys.readouterr().out)
    assert (
        register_payload["authority_review_subject_sha256"]
        == register_payload["dataset_snapshot_sha256"]
    )
    assert register_payload["request_sha256"] == prepare_payload["request_sha256"]
    assert register_payload["authority_sha256"] == prepare_payload["authority_sha256"]
    assert register_payload["authority_status"] == "self_consistent_unreviewed"
    assert register_payload["review_ids"] == []
    assert register_payload["publication_eligible"] is False
    assert (
        main(
            [
                "eval",
                "data",
                "check",
                str(bundle_path),
                "--snapshot",
                str(snapshot_path),
                "--json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["snapshot_matches"] is True

    tampered_source = tmp_path / "source-tampered.csv"
    tampered_source.write_bytes(content + b"tamper")
    assert (
        main(
            [
                "eval",
                "data",
                "prepare",
                str(request_path),
                "--source",
                str(tampered_source),
                "--output",
                str(tmp_path / "must-not-exist.json"),
                "--json",
            ]
        )
        == 2
    )
    assert "source_sha256" in capsys.readouterr().err
    assert not (tmp_path / "must-not-exist.json").exists()

    bundle_path.write_text(bundle_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert main(prepare_args) == 2
    assert "already exists with different content" in capsys.readouterr().err

    snapshot_payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot_payload["license_id"] = "overwrite-attempt"
    snapshot_path.write_text(json.dumps(snapshot_payload), encoding="utf-8")
    assert (
        main(
            [
                "eval",
                "data",
                "register",
                str(clean_bundle_path),
                "--output",
                str(snapshot_path),
                "--json",
            ]
        )
        == 2
    )
    assert "different content" in capsys.readouterr().err


def _minimal_prepared(request: EvalDataPreparationRequest, content: bytes) -> PreparedScenario:
    target_kind = (
        TargetKind.CATEGORICAL
        if request.scenario_id == "zachary-karate-club-label-completion"
        else TargetKind.NUMERIC
    )
    target = "Mr. Hi" if target_kind is TargetKind.CATEGORICAL else 1.0
    partitions = {
        name: (
            PreparedExample(
                example_id=f"{name}-example",
                target_kind=target_kind,
                target_family="target",
                features={"x": 1.0},
                target=target,
            ),
        )
        for name in ("train", "validation", "test")
    }
    preprocessing: dict[str, object] = {
        "fit_partition": "train",
        "implementation_sha256": "1" * 64,
        "fitted_state_sha256": "2" * 64,
        "source_authority_sha256": request.authority.content_hash,
    }
    if request.completion_mask_authority is not None:
        preprocessing["execution"] = {
            "mask_authority_sha256": request.completion_mask_authority.content_hash
        }
    preparation = PreparationContract(
        preprocessing=preprocessing,
        selection={"kind": "dispatch-test"},
        mask={"kind": "dispatch-test"},
    )
    source = SourceMaterial.from_bytes(
        dataset_id=(
            request.authority.dataset_id
            if isinstance(request.authority, MovieLensAuthority)
            else request.authority.split.dataset_id
        ),
        content=content,
        media_type=request.source_media_type,
    )
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    binding = DatasetSnapshotBinding(
        dataset_id=source.dataset_id,
        source_sha256=source.raw_sha256,
        split_sha256=PreparedScenario.split_sha256_for(
            train=train,
            validation=validation,
            test=test,
        ),
        recipe_sha256=PreparedScenario.recipe_sha256_for(preparation=preparation),
        truth_sidecar_sha256=PreparedScenario.truth_sidecar_sha256_for(test=test),
        partition_counts={"train": 1, "validation": 1, "test": 1},
    )
    return PreparedScenario(
        scenario_id=request.scenario_id,
        binding=binding,
        source_material=source,
        preparation=preparation,
        train=train,
        validation=validation,
        test=test,
    )


@pytest.mark.parametrize(
    ("suite_id", "scenario_id", "authority_kind", "expected_function"),
    [
        (
            "table-completion-micro-v0",
            "sklearn-diabetes-feature-completion-micro",
            "table",
            "materialize_table_completion",
        ),
        (
            "graph-completion-micro-v0",
            "zachary-karate-club-label-completion",
            "karate",
            "materialize_karate",
        ),
        (
            "recsys-completion-micro-v0",
            "movielens-100k-interaction-completion",
            "movielens",
            "materialize_movielens",
        ),
    ],
)
def test_frozen_authority_dispatch_is_explicit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    suite_id: str,
    scenario_id: str,
    authority_kind: str,
    expected_function: str,
) -> None:
    import tabu_lab.adapters.eval_data_workflow as workflow

    content = b"retained-dispatch-test"
    scenario = load_suite(suite_id).scenarios[0 if authority_kind != "table" else 1]
    split = _split(
        dataset_id=scenario.dataset.dataset_id,
        source_version=scenario.dataset.source_version,
        content=content,
        partitions={"train": ("1",), "validation": ("2",), "test": ("3",)},
    )
    if authority_kind == "table":
        authority = DelimitedTableAuthority(
            delimiter=",",
            field_whitespace="preserve",
            header=("row_id", "x", "outcome"),
            row_id_column="row_id",
            feature_columns=(ColumnAuthority(source_name="x", family_id="x", kind="numeric"),),
            response_column=ColumnAuthority(
                source_name="outcome",
                family_id="outcome",
                kind="numeric",
            ),
            split=split,
        )
        mask = CompletionMaskAuthority(mask_seed=1729)
        media_type = "text/csv"
    elif authority_kind == "karate":
        authority = KarateAuthority(
            split=split,
            feature_columns=(ColumnAuthority(source_name="x", family_id="x", kind="numeric"),),
            club_domain=("Mr. Hi", "Officer"),
            topology_sha256=canonical_hash({"topology": "dispatch-test"}),
            perturbations=GraphPerturbationAuthority(
                base_node_id="3",
                topology_toggle_edge=("1", "3"),
                locality_toggle_edge=("1", "2"),
            ),
        )
        mask = None
        media_type = "application/json"
    else:
        authority = MovieLensAuthority(
            source_sha256=_sha256(content),
            base_member="ml-100k/u1.base",
            test_member="ml-100k/u1.test",
            validation_interaction_ids=("1:1",),
        )
        mask = None
        media_type = "application/zip"
    request = EvalDataPreparationRequest(
        suite_id=suite_id,
        scenario_id=scenario_id,
        source_sha256=_sha256(content),
        source_size_bytes=len(content),
        source_media_type=media_type,
        authority=authority,
        completion_mask_authority=mask,
    )
    source = tmp_path / "retained.bin"
    source.write_bytes(content)
    called: list[str] = []

    def fake_materializer(**_: object) -> PreparedScenario:
        called.append(expected_function)
        return _minimal_prepared(request, content)

    monkeypatch.setattr(workflow, expected_function, fake_materializer)
    bundle = prepare_eval_data_bundle(request=request, source=source)
    assert bundle.request.scenario_id == scenario_id
    assert called == [expected_function]


def test_templates_are_non_evidence_placeholders() -> None:
    root = Path(__file__).resolve().parents[2] / "evaluations" / "data"
    templates = tuple(sorted((root / "templates").glob("*.template.yaml")))
    assert len(templates) == 6
    for template in templates:
        text = template.read_text(encoding="utf-8")
        assert "REPLACE_WITH_ACTUAL_SHA256" in text
        assert "tabu.dataset-snapshot.v2" not in text
        assert "tabu.eval-result" not in text
        assert not any(
            len(token) == 64 and set(token) <= set("0123456789abcdef") for token in text.split()
        )
