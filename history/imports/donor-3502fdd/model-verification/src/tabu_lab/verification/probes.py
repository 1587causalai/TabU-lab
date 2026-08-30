"""Shared executable probes used by pytest and the structured MVE runner."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import torch

from tabu_lab.contracts import FeatureKind, FeatureRole, FeatureSpec
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.models.types import DenseModelInput
from tabu_lab.registry import get_model_spec

from .composition import describe_model
from .contracts import AssessmentOutcome, VerificationCheckResult
from .registry import register_check

_SMOKE_CONFIG = ReferenceConfig(
    d_model=8,
    n_heads=2,
    d_ff=16,
    n_blocks=1,
    inducing_slots=2,
    matched_slots=2,
    max_features=256,
)


def _fixture(contract_id: str) -> Any:
    from tabu_lab.experiments.fixtures import build_f0_fixture

    return build_f0_fixture(contract_id)


def _model(contract_id: str) -> tuple[Any, Any]:
    if contract_id.startswith("tabu.cell.") and contract_id != "tabu.cell.rec":
        # The new cell-as-Unit family has builders but no F0 fixtures yet. A
        # tiny truth-free dense input keeps component probes executable
        # without pretending that it has a synthetic-fit experiment.
        values = torch.tensor(
            [[[1.0, 2.0, 3.0, 4.0], [5.0, 0.0, 7.0, 8.0]]],
            dtype=torch.float32,
        )
        visible = torch.tensor(
            [[[True, True, True, True], [True, False, True, True]]],
            dtype=torch.bool,
        )
        target = ~visible
        fixture = SimpleNamespace(
            builder_options={},
            evidence=DenseModelInput(
                values=values,
                visible_mask=visible,
                target_mask=target,
                natural_missing_mask=torch.zeros_like(target),
                artificial_target_mask=target,
                query_target_mask=torch.zeros_like(target),
                unsupported_target_mask=torch.zeros_like(target),
            ),
        )
        model_options = {"config": _SMOKE_CONFIG}
        if contract_id == "tabu.cell.base":
            model_options["profile"] = "completion.artificial_mask.v1"
        return build_model(contract_id, **model_options), fixture
    fixture = _fixture(contract_id)
    model = build_model(contract_id, config=_SMOKE_CONFIG, **fixture.builder_options)
    return model, fixture


def _forward(model: Any, fixture: Any) -> Any:
    """Use the public episode boundary, with the cell.base dense test seam."""

    if isinstance(fixture.evidence, DenseModelInput):
        return model._forward_dense(fixture.evidence)
    return model(fixture.evidence)


def _result(
    check_id: str, outcome: AssessmentOutcome, detail: str, **metrics: Any
) -> VerificationCheckResult:
    return VerificationCheckResult(
        check_id=check_id,
        outcome=outcome,
        detail=detail,
        metrics=metrics,
    )


def component_build(contract_id: str, context: Mapping[str, Any]) -> VerificationCheckResult:
    check_id = "component.build_interface"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        model, _ = _model(contract_id)
        return _result(
            check_id,
            AssessmentOutcome.PASSED,
            "reference builder constructed the declared model",
            composition_hash=describe_model(
                model,
                contract_id=contract_id,
                contract_version=get_model_spec(contract_id).contract_version,
            ).composition_hash,
        )
    except Exception as exc:  # pragma: no cover - defensive boundary
        return _result(
            check_id, AssessmentOutcome.FAILED, f"builder failed: {type(exc).__name__}: {exc}"
        )


def component_forward(contract_id: str, context: Mapping[str, Any]) -> VerificationCheckResult:
    check_id = "component.finite_forward"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        model, fixture = _model(contract_id)
        with torch.no_grad():
            prediction = _forward(model, fixture)
        tensors = [
            value for value in prediction.outputs.values() if isinstance(value, torch.Tensor)
        ]
        finite = all(bool(torch.isfinite(value).all()) for value in tensors)
        return _result(
            check_id,
            AssessmentOutcome.PASSED if finite else AssessmentOutcome.FAILED,
            "forward outputs are finite" if finite else "forward produced a non-finite tensor",
            output_tensors=len(tensors),
        )
    except Exception as exc:
        return _result(
            check_id, AssessmentOutcome.FAILED, f"forward failed: {type(exc).__name__}: {exc}"
        )


def component_gradients(contract_id: str, context: Mapping[str, Any]) -> VerificationCheckResult:
    check_id = "component.gradient_reachability"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        model, fixture = _model(contract_id)
        model.zero_grad(set_to_none=True)
        prediction = _forward(model, fixture)
        differentiable = [
            value
            for value in prediction.outputs.values()
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.requires_grad
            )
        ]
        if not differentiable:
            return _result(
                check_id,
                AssessmentOutcome.FAILED,
                "forward produced no differentiable output",
            )
        torch.stack([value.reshape(-1).sum() for value in differentiable]).sum().backward()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        reachable = [
            parameter
            for parameter in trainable
            if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        ]
        passed = bool(trainable) and bool(reachable)
        return _result(
            check_id,
            AssessmentOutcome.PASSED if passed else AssessmentOutcome.FAILED,
            "finite gradients reach declared trainable parameters"
            if passed
            else "no finite gradient reaches a declared trainable parameter",
            trainable_parameters=len(trainable),
            reachable_parameters=len(reachable),
        )
    except Exception as exc:
        return _result(
            check_id,
            AssessmentOutcome.FAILED,
            f"gradient probe failed: {type(exc).__name__}: {exc}",
        )


def component_determinism(contract_id: str, context: Mapping[str, Any]) -> VerificationCheckResult:
    check_id = "component.seed_dtype_checkpoint"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        torch.manual_seed(1729)
        first, fixture = _model(contract_id)
        torch.manual_seed(1729)
        second, _ = _model(contract_id)
        first.eval()
        second.eval()
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            left = _forward(first, fixture)
            right = _forward(second, fixture)
        equal = all(
            torch.equal(a, b)
            for a, b in zip(
                (v for v in left.outputs.values() if isinstance(v, torch.Tensor)),
                (v for v in right.outputs.values() if isinstance(v, torch.Tensor)),
                strict=False,
            )
        )
        return _result(
            check_id,
            AssessmentOutcome.PASSED if equal else AssessmentOutcome.FAILED,
            "fixed seed save/load produced equivalent forward outputs"
            if equal
            else "fixed seed save/load outputs differ",
        )
    except Exception as exc:
        return _result(
            check_id,
            AssessmentOutcome.FAILED,
            f"determinism/checkpoint probe failed: {type(exc).__name__}: {exc}",
        )


def component_invalid_config(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    check_id = "component.invalid_config_fail_closed"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        ReferenceConfig(d_model=7, n_heads=2)
    except ValueError:
        return _result(check_id, AssessmentOutcome.PASSED, "invalid configuration was rejected")
    return _result(check_id, AssessmentOutcome.FAILED, "invalid configuration was accepted")


def component_truth_boundary(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    check_id = "component.truth_sidecar_abstention"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        model, fixture = _model(contract_id)
        with torch.no_grad():
            prediction = _forward(model, fixture)
        has_truth = "target_values" in prediction.metadata or "truth" in prediction.metadata
        return _result(
            check_id,
            AssessmentOutcome.FAILED if has_truth else AssessmentOutcome.PASSED,
            "forward boundary carries no TruthSidecar payload"
            if not has_truth
            else "forward metadata exposed target truth",
        )
    except Exception as exc:
        return _result(
            check_id,
            AssessmentOutcome.FAILED,
            f"truth-boundary probe failed: {type(exc).__name__}: {exc}",
        )


def component_profile_matrix(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    """Exercise both public TabUBase 0.2 profiles through the forward boundary."""

    check_id = "component.profile_matrix"
    if contract_id != "tabu.cell.base":
        return _result(
            check_id,
            AssessmentOutcome.NOT_APPLICABLE,
            "the v0.2 profile matrix is declared only for tabu.cell.base",
        )
    try:
        completion_model, completion_fixture = _model(contract_id)
        supervised_model = build_model(
            contract_id,
            config=_SMOKE_CONFIG,
            profile="supervised.label_broadcast.v1",
        )
        values = torch.tensor(
            [[[1.0, 0.0, 1.0, 0.0], [2.0, 1.0, 2.0, 3.0], [4.0, 2.0, 0.0, 0.0]]],
            dtype=torch.float32,
        )
        visible = torch.tensor(
            [[[True, True, True, False], [True, True, True, True], [True, True, True, False]]],
            dtype=torch.bool,
        )
        query = ~visible
        supervised_fixture = SimpleNamespace(
            evidence=DenseModelInput(
                values=values,
                visible_mask=visible,
                target_mask=query,
                natural_missing_mask=torch.zeros_like(query),
                artificial_target_mask=torch.zeros_like(query),
                query_target_mask=query,
                unsupported_target_mask=torch.zeros_like(query),
                feature_specs=(
                    FeatureSpec(name="x_numeric", kind=FeatureKind.NUMERIC),
                    FeatureSpec(
                        name="x_ordinal",
                        kind=FeatureKind.ORDINAL,
                        domain=("low", "mid", "high"),
                        codebook_id="ordinal-v1",
                    ),
                    FeatureSpec(
                        name="x_nominal",
                        kind=FeatureKind.CATEGORICAL,
                        domain=("a", "b", "c"),
                        codebook_id="nominal-v1",
                    ),
                    FeatureSpec(name="y", role=FeatureRole.RESPONSE),
                ),
                episode_id="tabubase-profile-matrix",
            )
        )
        reports: dict[str, dict[str, Any]] = {}
        for profile, model, fixture in (
            ("completion.artificial_mask.v1", completion_model, completion_fixture),
            ("supervised.label_broadcast.v1", supervised_model, supervised_fixture),
        ):
            with torch.no_grad():
                prediction = _forward(model, fixture)
            tensors = [
                value for value in prediction.outputs.values() if isinstance(value, torch.Tensor)
            ]
            finite = bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)
            metadata = prediction.metadata
            reports[profile] = {
                "finite": finite,
                "profile_id": metadata.get("profile_id"),
                "label_broadcast": metadata.get("label_broadcast"),
                "query_marker": metadata.get("query_marker"),
            }
            if not finite or metadata.get("profile_id") != profile:
                return _result(
                    check_id,
                    AssessmentOutcome.FAILED,
                    f"profile forward failed for {profile}",
                    profiles=reports,
                )
            if bool(metadata.get("label_broadcast")) != profile.startswith("supervised."):
                return _result(
                    check_id,
                    AssessmentOutcome.FAILED,
                    f"profile broadcast semantics drifted for {profile}",
                    profiles=reports,
                )
        return _result(
            check_id,
            AssessmentOutcome.PASSED,
            "completion and supervised profiles pass the same finite forward boundary",
            profiles=reports,
        )
    except Exception as exc:
        return _result(
            check_id,
            AssessmentOutcome.FAILED,
            f"profile matrix probe failed: {type(exc).__name__}: {exc}",
        )


def architecture_substitution(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    check_id = "architecture.controlled_substitution"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    if contract_id not in {
        "tabuf",
        "tabu.unit_row",
        "tabu.unit_pair",
        "tabul",
        "tabufl",
        "tabu4graph",
        "tabu4rec",
        "tabu.cell.base",
    }:
        return _result(
            check_id, AssessmentOutcome.BLOCKED, "no substitution contract is registered"
        )
    try:
        if contract_id == "tabu.cell.base":
            fixture = _model(contract_id)[1]
            options = {"profile": "completion.artificial_mask.v1"}
        else:
            fixture = _fixture(contract_id)
            options = dict(fixture.builder_options)
        if contract_id == "tabu.cell.base":
            options.setdefault("profile", "completion.artificial_mask.v1")
        base = build_model(contract_id, config=_SMOKE_CONFIG, **options)
        base_hash = describe_model(
            base,
            contract_id=contract_id,
            contract_version=get_model_spec(contract_id).contract_version,
        ).composition_hash
        omab = build_model(
            contract_id,
            config=replace(_SMOKE_CONFIG, block_kind="omab"),
            **options,
        )
        mab = build_model(
            contract_id,
            config=replace(_SMOKE_CONFIG, block_kind="mab"),
            **options,
        )
        omab_hash = describe_model(
            omab,
            contract_id=contract_id,
            contract_version=get_model_spec(contract_id).contract_version,
        ).composition_hash
        mab_hash = describe_model(
            mab,
            contract_id=contract_id,
            contract_version=get_model_spec(contract_id).contract_version,
        ).composition_hash
        nw = build_model(
            contract_id,
            config=_SMOKE_CONFIG,
            numeric_terminal="nadaraya_watson",
            **options,
        )
        local_linear = build_model(
            contract_id,
            config=_SMOKE_CONFIG,
            numeric_terminal="local_linear",
            **options,
        )
        nw_hash = describe_model(
            nw,
            contract_id=contract_id,
            contract_version=get_model_spec(contract_id).contract_version,
        ).composition_hash
        local_linear_hash = describe_model(
            local_linear,
            contract_id=contract_id,
            contract_version=get_model_spec(contract_id).contract_version,
        ).composition_hash
        if nw_hash == local_linear_hash:
            return _result(
                check_id,
                AssessmentOutcome.BLOCKED,
                "numeric terminal substitution is not exposed by this composition",
                substitutions=["omab->mab", "nadaraya_watson->local_linear"],
                composition_hashes={
                    "base": base_hash,
                    "omab": omab_hash,
                    "mab": mab_hash,
                    "nadaraya_watson": nw_hash,
                    "local_linear": local_linear_hash,
                },
            )
        # MAB is retained only as a named non-O-closed ablation.  It is
        # reported, but never counted as a same-contract OMAB replacement for
        # the frozen Base profile.
        comparable_hashes = {base_hash, nw_hash, local_linear_hash}
        if contract_id != "tabu.cell.base":
            comparable_hashes.add(mab_hash)
        distinct = (
            len(comparable_hashes) >= 3
            if contract_id != "tabu.cell.base"
            else nw_hash != local_linear_hash
        )
        return _result(
            check_id,
            AssessmentOutcome.PASSED if distinct else AssessmentOutcome.FAILED,
            "controlled substitutions changed only the declared composition seams"
            if distinct
            else "controlled substitutions did not produce distinct semantic hashes",
            substitutions=["omab->mab", "nadaraya_watson->local_linear"],
            composition_hashes={
                "base": base_hash,
                "omab": omab_hash,
                "mab": mab_hash,
                "nadaraya_watson": nw_hash,
                "local_linear": local_linear_hash,
            },
        )
    except (TypeError, ValueError, KeyError) as exc:
        return _result(
            check_id,
            AssessmentOutcome.BLOCKED,
            f"no complete substitution seam is registered: {type(exc).__name__}: {exc}",
        )


def _oracle_outcomes(build: Callable[[], tuple[Any, Any]]) -> dict[str, str]:
    """Re-run the Link 1 component oracle against one supplied variant build.

    The five checks deliberately mirror the registered component probes
    without delegating to them, so a substitution can never change what the
    regression oracle looks at.  Every outcome is a plain ``passed`` /
    ``failed[:ExceptionName]`` token so the report stays JSON-typed.
    """

    outcomes: dict[str, str] = {}
    try:
        model, fixture = build()
    except Exception as exc:  # pragma: no cover - defensive boundary
        return {"build": f"failed:{type(exc).__name__}"}
    outcomes["build"] = "passed"

    try:
        with torch.no_grad():
            prediction = _forward(model, fixture)
        tensors = [
            value for value in prediction.outputs.values() if isinstance(value, torch.Tensor)
        ]
        finite = bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors)
        outcomes["forward_finite"] = "passed" if finite else "failed"
    except Exception as exc:
        outcomes["forward_finite"] = f"failed:{type(exc).__name__}"

    try:
        model.zero_grad(set_to_none=True)
        prediction = _forward(model, fixture)
        differentiable = [
            value
            for value in prediction.outputs.values()
            if (
                isinstance(value, torch.Tensor)
                and value.is_floating_point()
                and value.requires_grad
            )
        ]
        if differentiable:
            torch.stack([value.reshape(-1).sum() for value in differentiable]).sum().backward()
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
            reachable = [
                parameter
                for parameter in trainable
                if parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            ]
            outcomes["gradient_reachability"] = (
                "passed" if trainable and reachable else "failed"
            )
        else:
            outcomes["gradient_reachability"] = "failed"
    except Exception as exc:
        outcomes["gradient_reachability"] = f"failed:{type(exc).__name__}"

    try:
        torch.manual_seed(1729)
        first, first_fixture = build()
        torch.manual_seed(1729)
        second, _ = build()
        first.eval()
        second.eval()
        second.load_state_dict(first.state_dict())
        with torch.no_grad():
            left = _forward(first, first_fixture)
            right = _forward(second, first_fixture)
        equal = all(
            torch.equal(a, b)
            for a, b in zip(
                (v for v in left.outputs.values() if isinstance(v, torch.Tensor)),
                (v for v in right.outputs.values() if isinstance(v, torch.Tensor)),
                strict=False,
            )
        )
        outcomes["seed_dtype_checkpoint"] = "passed" if equal else "failed"
    except Exception as exc:
        outcomes["seed_dtype_checkpoint"] = f"failed:{type(exc).__name__}"

    try:
        with torch.no_grad():
            prediction = _forward(model, fixture)
        has_truth = "target_values" in prediction.metadata or "truth" in prediction.metadata
        outcomes["truth_sidecar_abstention"] = "passed" if not has_truth else "failed"
    except Exception as exc:
        outcomes["truth_sidecar_abstention"] = f"failed:{type(exc).__name__}"

    return outcomes


_SUBSTITUTION_SEAM_CONTRACTS = {
    "tabuf",
    "tabu.unit_row",
    "tabu.unit_pair",
    "tabul",
    "tabufl",
    "tabu4graph",
    "tabu4rec",
    "tabu.cell.base",
}

_SUBSTITUTION_SEAMS = {
    "numeric_terminal:nadaraya_watson": {"numeric_terminal": "nadaraya_watson"},
    "numeric_terminal:local_linear": {"numeric_terminal": "local_linear"},
}


def _substitution_seams(contract_id: str) -> dict[str, dict[str, Any]]:
    """Declared swap seams for one contract, verified against the builders.

    The terminal and config seams exist on every contract; the label and
    recommendation address plans exist only where the builders declare them.
    A seam that is declared here but rejected by a builder surfaces as a
    probe ``BLOCKED`` result rather than a silent skip.
    """

    seams: dict[str, dict[str, Any]] = dict(_SUBSTITUTION_SEAMS)
    if contract_id != "tabu.cell.base":
        seams["geometry_normalization:rms_unit"] = {
            "config": replace(_SMOKE_CONFIG, geometry_normalization="rms_unit")
        }
        seams["block_kind:mab"] = {"config": replace(_SMOKE_CONFIG, block_kind="mab")}
    if contract_id in {"tabul", "tabufl"}:
        seams["label_address_plan:predictor_only_per_label_v1"] = {
            "label_address_plan": "predictor_only_per_label_v1"
        }
        seams["label_address_plan:predictor_unit_linked_per_label_v2"] = {
            "label_address_plan": "predictor_unit_linked_per_label_v2"
        }
    if contract_id == "tabu4rec":
        seams["recommendation_address_plan:cell_global_support_v1"] = {
            "recommendation_address_plan": "cell_global_support_v1"
        }
    return seams


def architecture_substitution_nonregression(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    """Swap each declared seam and require the Link 1 oracle to stay green.

    ``architecture.controlled_substitution`` proves that a substitution changes
    the composition identity; this check proves it changes nothing else.  The
    two together are the executable form of evaluation-chain Link 2.
    """

    check_id = "architecture.substitution_nonregression"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    if contract_id not in _SUBSTITUTION_SEAM_CONTRACTS:
        return _result(
            check_id, AssessmentOutcome.BLOCKED, "no substitution contract is registered"
        )
    try:
        if contract_id == "tabu.cell.base":
            options = {"profile": "completion.artificial_mask.v1"}
            base_fixture = _model(contract_id)[1]
        else:
            base_fixture = _fixture(contract_id)
            options = dict(base_fixture.builder_options)

        def build_for(seam_kwargs: dict[str, Any]) -> Callable[[], tuple[Any, Any]]:
            def build() -> tuple[Any, Any]:
                model = build_model(
                    contract_id,
                    config=seam_kwargs.get("config", _SMOKE_CONFIG),
                    **options,
                    **{
                        name: value
                        for name, value in seam_kwargs.items()
                        if name != "config"
                    },
                )
                fixture = base_fixture if contract_id == "tabu.cell.base" else _fixture(contract_id)
                return model, fixture

            return build

        seam_reports = {
            name: _oracle_outcomes(build_for(seam_kwargs))
            for name, seam_kwargs in _substitution_seams(contract_id).items()
        }
        regressed = sorted(
            name
            for name, outcomes in seam_reports.items()
            if any(not value.startswith("passed") for value in outcomes.values())
        )
        return _result(
            check_id,
            AssessmentOutcome.PASSED if not regressed else AssessmentOutcome.FAILED,
            "substituted variants pass the component oracle unchanged"
            if not regressed
            else f"substituted variants regressed the component oracle: {regressed}",
            seam_reports=seam_reports,
        )
    except (TypeError, ValueError, KeyError) as exc:
        return _result(
            check_id,
            AssessmentOutcome.BLOCKED,
            f"no complete substitution seam is registered: {type(exc).__name__}: {exc}",
        )


def architecture_composition_identity(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    check_id = "architecture.composition_checkpoint_identity"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        model, _ = _model(contract_id)
        descriptor = describe_model(
            model,
            contract_id=contract_id,
            contract_version=get_model_spec(contract_id).contract_version,
        )
        return _result(
            check_id,
            AssessmentOutcome.PASSED,
            "composition has a stable semantic hash",
            composition_hash=descriptor.composition_hash,
        )
    except Exception as exc:
        return _result(
            check_id,
            AssessmentOutcome.FAILED,
            f"composition probe failed: {type(exc).__name__}: {exc}",
        )


def architecture_builder_registry(
    contract_id: str, context: Mapping[str, Any]
) -> VerificationCheckResult:
    check_id = "architecture.builder_registry_extension"
    if contract_id == "tabu4do":
        return _result(check_id, AssessmentOutcome.NOT_APPLICABLE, "TabU4Do remains design_open")
    try:
        from tabu_lab.models import MODEL_BUILDERS, BuilderRegistry

        baseline_ids = MODEL_BUILDERS.ids()
        if not baseline_ids:
            return _result(check_id, AssessmentOutcome.FAILED, "global builder registry is empty")
        local = BuilderRegistry({"probe.base": lambda **_: "base"})
        local.register("probe.extension", lambda **_: "extended")
        resolved_base = local.get("probe.base")()
        resolved_extension = local.get("probe.extension")()
        duplicate_rejected = False
        try:
            local.register("probe.extension", lambda **_: "again")
        except ValueError:
            duplicate_rejected = True
        global_unchanged = MODEL_BUILDERS.ids() == baseline_ids
        passed = (
            resolved_base == "base"
            and resolved_extension == "extended"
            and duplicate_rejected
            and global_unchanged
        )
        return _result(
            check_id,
            AssessmentOutcome.PASSED if passed else AssessmentOutcome.FAILED,
            "registry extension resolves new builders, rejects duplicates, and leaves "
            "built-in registrations unchanged"
            if passed
            else "builder registry extension contract violated",
            baseline_builder_count=len(baseline_ids),
            probe_extension_id="probe.extension",
        )
    except Exception as exc:
        return _result(
            check_id,
            AssessmentOutcome.FAILED,
            f"builder registry probe failed: {type(exc).__name__}: {exc}",
        )


for _id, _fn in {
    "component.build_interface": component_build,
    "component.finite_forward": component_forward,
    "component.gradient_reachability": component_gradients,
    "component.seed_dtype_checkpoint": component_determinism,
    "component.invalid_config_fail_closed": component_invalid_config,
    "component.truth_sidecar_abstention": component_truth_boundary,
    "component.profile_matrix": component_profile_matrix,
    "architecture.controlled_substitution": architecture_substitution,
    "architecture.substitution_nonregression": architecture_substitution_nonregression,
    "architecture.composition_checkpoint_identity": architecture_composition_identity,
    "architecture.builder_registry_extension": architecture_builder_registry,
}.items():
    register_check(_id, _fn)


__all__ = [
    "architecture_builder_registry",
    "architecture_composition_identity",
    "architecture_substitution",
    "architecture_substitution_nonregression",
    "component_build",
    "component_determinism",
    "component_forward",
    "component_gradients",
    "component_invalid_config",
    "component_profile_matrix",
    "component_truth_boundary",
]
