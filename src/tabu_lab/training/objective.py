"""Loss-side objectives for truth-free TabU model predictions."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from tabu_lab.contracts import LossBundle, PredictionBundle, TruthSidecar

NumericTargetCoordinate = Literal["raw", "context_standardized"]


def _required_auxiliary(prediction: PredictionBundle, name: str) -> Tensor:
    value = prediction.auxiliaries.get(name)
    if value is None:
        raise ValueError(f"prediction requires {name} auxiliary")
    return value


def _zero_path(prediction: PredictionBundle, shape: torch.Size) -> Tensor:
    coordinates = prediction.auxiliaries.get("coordinates")
    if coordinates is None or coordinates.shape[:-1] != shape:
        raise ValueError("abstaining prediction requires coordinate provenance")
    return coordinates.sum(dim=-1) * 0.0


def _family_tensor(
    prediction: PredictionBundle,
    name: str,
    *,
    shape: torch.Size,
) -> Tensor:
    entry = prediction.entries.get(name)
    if entry is None or entry.values is None:
        return _zero_path(prediction, shape)
    if entry.values.shape != shape:
        raise ValueError(f"{name} prediction and target masks must have identical shapes")
    return entry.values


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    """Reduce a dense tensor without autograd's indexed-scatter backward.

    Boolean indexing and ``masked_scatter`` lower to an accumulating index-put
    during backward on MPS.  PyTorch does not provide a deterministic MPS
    implementation for that kernel.  The fit contract requires deterministic
    algorithms, so loss reductions stay dense and use ``where`` instead.
    """

    if values.shape != mask.shape or mask.dtype is not torch.bool:
        raise ValueError("masked mean requires matching values and bool mask")
    count = mask.sum().to(dtype=values.dtype)
    return torch.where(mask, values, torch.zeros_like(values)).sum() / count.clamp_min(1)


def _numeric_truth_in_prediction_coordinates(
    prediction: PredictionBundle,
    truth_values: Tensor,
    *,
    coordinate: NumericTargetCoordinate,
) -> Tensor:
    """Project raw sidecar truth into the numeric prediction coordinate system."""

    if coordinate == "raw":
        return truth_values

    numeric_entry = prediction.entries.get("numeric")
    value_space = None if numeric_entry is None else numeric_entry.metadata.get("value_space")
    if value_space != "context_standardized":
        raise ValueError(
            "context-standardized objective requires a numeric prediction entry "
            "declared in context_standardized value space"
        )
    mean = _required_auxiliary(prediction, "numeric_context_mean").to(
        device=truth_values.device,
        dtype=truth_values.dtype,
    )
    scale = _required_auxiliary(prediction, "numeric_context_scale").to(
        device=truth_values.device,
        dtype=truth_values.dtype,
    )
    try:
        mean = torch.broadcast_to(mean, truth_values.shape)
        scale = torch.broadcast_to(scale, truth_values.shape)
    except RuntimeError as exc:
        raise ValueError(
            "numeric context mean/scale must broadcast to TruthSidecar target values"
        ) from exc
    if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(scale).all()):
        raise ValueError("numeric context mean/scale must be finite")
    if bool((scale <= 0).any()):
        raise ValueError("numeric context scale must be strictly positive")
    return (truth_values - mean) / scale


class MixedObjective(nn.Module):
    """Equal-weight active-family objective for numeric and categorical cells.

    Families are active only when at least one factual sidecar cell also has
    model support. Thus an absent family never changes another family's scale,
    and abstentions are counted rather than converted into zero predictions.
    """

    def __init__(
        self,
        *,
        mse_weight: float = 1.0,
        mae_weight: float = 0.0,
        categorical_nll_weight: float = 1.0,
        categorical_epsilon: float = 1.0e-8,
        include_categorical: bool = True,
        numeric_target_coordinate: NumericTargetCoordinate = "raw",
    ) -> None:
        super().__init__()
        if mse_weight < 0.0 or mae_weight < 0.0:
            raise ValueError("numeric objective weights must be non-negative")
        if mse_weight == 0.0 and mae_weight == 0.0:
            raise ValueError("at least one numeric objective weight must be positive")
        if categorical_nll_weight <= 0.0:
            raise ValueError("categorical_nll_weight must be positive")
        if not 0.0 < categorical_epsilon < 1.0:
            raise ValueError("categorical_epsilon must be in (0, 1)")
        if numeric_target_coordinate not in {"raw", "context_standardized"}:
            raise ValueError(
                "numeric_target_coordinate must be raw or context_standardized"
            )
        self.mse_weight = float(mse_weight)
        self.mae_weight = float(mae_weight)
        self.categorical_nll_weight = float(categorical_nll_weight)
        self.categorical_epsilon = float(categorical_epsilon)
        self.include_categorical = bool(include_categorical)
        self.numeric_target_coordinate = numeric_target_coordinate

    @property
    def resume_config(self) -> dict[str, float | bool | str]:
        """Immutable scalar configuration bound into exact-resume checkpoints."""

        return {
            "categorical_epsilon": self.categorical_epsilon,
            "categorical_nll_weight": self.categorical_nll_weight,
            "include_categorical": self.include_categorical,
            "mae_weight": self.mae_weight,
            "mse_weight": self.mse_weight,
            "numeric_target_coordinate": self.numeric_target_coordinate,
        }

    def forward(
        self,
        prediction: PredictionBundle,
        truth: TruthSidecar,
    ) -> LossBundle:
        if prediction.episode_id != truth.episode_id:
            raise ValueError("prediction and TruthSidecar episode ids must match")
        model_targets = _required_auxiliary(prediction, "target_mask").to(torch.bool)
        model_support = _required_auxiliary(prediction, "support_available").to(torch.bool)
        if model_targets.shape != model_support.shape:
            raise ValueError("target_mask and support_available must have identical shapes")
        truth_values = truth.target_values.to(device=model_targets.device)
        truth_targets = truth.target_mask.to(device=model_targets.device)
        if truth_values.shape != model_targets.shape or truth_targets.shape != model_targets.shape:
            raise ValueError("prediction and TruthSidecar shapes must match")
        if bool((truth_targets & ~model_targets).any()):
            raise ValueError("TruthSidecar target_mask must be a subset of model targets")

        numeric_targets = prediction.auxiliaries.get("numeric_target_mask", model_targets)
        numeric_support = prediction.auxiliaries.get("numeric_support_available", model_support)
        categorical_targets = prediction.auxiliaries.get(
            "categorical_target_mask", torch.zeros_like(model_targets)
        )
        categorical_support = prediction.auxiliaries.get(
            "categorical_support_available", torch.zeros_like(model_targets)
        )
        for name, mask in (
            ("numeric_target_mask", numeric_targets),
            ("numeric_support_available", numeric_support),
            ("categorical_target_mask", categorical_targets),
            ("categorical_support_available", categorical_support),
        ):
            if mask.shape != model_targets.shape:
                raise ValueError(f"{name} must match target_mask")
        numeric_targets = numeric_targets.to(device=model_targets.device, dtype=torch.bool)
        numeric_support = numeric_support.to(device=model_targets.device, dtype=torch.bool)
        categorical_targets = categorical_targets.to(device=model_targets.device, dtype=torch.bool)
        categorical_support = categorical_support.to(device=model_targets.device, dtype=torch.bool)
        if bool((numeric_targets & categorical_targets).any()) or not torch.equal(
            numeric_targets | categorical_targets, model_targets
        ):
            raise ValueError("numeric/categorical target masks must partition model targets")

        numeric = _family_tensor(prediction, "numeric", shape=model_targets.shape)
        numeric_scored = truth_targets & numeric_targets & numeric_support
        numeric_truth = truth_values.to(device=numeric.device, dtype=numeric.dtype)
        if bool(numeric_scored.any()):
            numeric_truth = _numeric_truth_in_prediction_coordinates(
                prediction,
                numeric_truth,
                coordinate=self.numeric_target_coordinate,
            )
        numeric_error_cells = numeric - numeric_truth
        if bool(numeric_scored.any()):
            mse = _masked_mean(numeric_error_cells.square(), numeric_scored)
            mae = _masked_mean(numeric_error_cells.abs(), numeric_scored)
        else:
            zero = numeric.sum() * 0.0
            mse = zero
            mae = zero
        numeric_loss = self.mse_weight * mse + self.mae_weight * mae

        distribution_entry = prediction.entries.get("distribution")
        domain_mask = prediction.auxiliaries.get("categorical_domain_mask")
        categorical_log_probabilities = prediction.auxiliaries.get("categorical_log_probabilities")
        categorical_class_support_available = prediction.auxiliaries.get(
            "categorical_class_support_available"
        )
        categorical_scored = truth_targets & categorical_targets & categorical_support
        categorical_accuracy = numeric.sum() * 0.0
        categorical_nll = numeric.sum() * 0.0
        categorical_nll_cells = torch.zeros_like(numeric)
        if bool(categorical_scored.any()):
            if not self.include_categorical:
                categorical_scored = torch.zeros_like(categorical_scored)
            else:
                if distribution_entry is None or distribution_entry.values is None:
                    raise ValueError("supported categorical targets require distribution values")
                probabilities = distribution_entry.values
                if probabilities.shape[:-1] != model_targets.shape:
                    raise ValueError("categorical distribution must be [...,M,C]")
                if categorical_log_probabilities is not None and (
                    categorical_log_probabilities.shape != probabilities.shape
                    or not categorical_log_probabilities.is_floating_point()
                    or not bool(torch.isfinite(categorical_log_probabilities).all())
                ):
                    raise ValueError(
                        "categorical log probabilities must be finite and match "
                        "the public distribution"
                    )
                if categorical_class_support_available is not None and (
                    categorical_class_support_available.shape != probabilities.shape
                    or categorical_class_support_available.dtype is not torch.bool
                ):
                    raise ValueError(
                        "categorical class support availability must be bool and match "
                        "the public distribution"
                    )
                if domain_mask is None or domain_mask.ndim != 2:
                    raise ValueError("categorical distribution requires declared domain mask")
                if domain_mask.shape != probabilities.shape[-2:]:
                    raise ValueError("categorical domain mask must be [M,C]")
                raw_codes = truth_values.to(device=probabilities.device)
                rounded_codes = raw_codes.round()
                if bool((categorical_scored & ~torch.isclose(raw_codes, rounded_codes)).any()):
                    raise ValueError("categorical truth values must be integer domain codes")
                codes = rounded_codes.to(torch.int64)
                outside_domain = (codes < 0) | (codes >= probabilities.shape[-1])
                if bool((categorical_scored & outside_domain).any()):
                    raise ValueError("categorical truth code is outside the declared domain")
                safe_codes = codes.clamp(0, probabilities.shape[-1] - 1)
                selected_classes = F.one_hot(
                    safe_codes,
                    num_classes=probabilities.shape[-1],
                ).to(torch.bool)
                domain_view = domain_mask.to(device=probabilities.device).view(
                    *((1,) * (model_targets.ndim - 1)), *domain_mask.shape
                )
                expanded_domain = domain_view.expand(*model_targets.shape, -1)
                valid_codes = (expanded_domain & selected_classes).any(dim=-1)
                if bool((categorical_scored & ~valid_codes).any()):
                    raise ValueError("categorical truth code is outside the declared domain")
                if categorical_log_probabilities is None:
                    selected = (
                        probabilities * selected_classes.to(probabilities.dtype)
                    ).sum(dim=-1)
                    selected_nll = -selected.clamp_min(self.categorical_epsilon).log()
                else:
                    dense_log_probabilities = categorical_log_probabilities.to(
                        device=probabilities.device
                    )
                    selected_log_probabilities = (
                        dense_log_probabilities
                        * selected_classes.to(dense_log_probabilities.dtype)
                    ).sum(dim=-1)
                    selected_nll = (-selected_log_probabilities).clamp_min(0.0)
                    if categorical_class_support_available is not None:
                        dense_class_support = categorical_class_support_available.to(
                            device=probabilities.device
                        )
                        selected_class_support = (
                            dense_class_support & selected_classes
                        ).any(dim=-1)
                        epsilon_nll = -selected_nll.new_full(
                            selected_nll.shape,
                            self.categorical_epsilon,
                        ).log()
                        selected_nll = torch.where(
                            selected_class_support,
                            selected_nll,
                            epsilon_nll,
                        )
                categorical_nll = _masked_mean(selected_nll, categorical_scored)
                categorical_nll_cells = torch.where(
                    categorical_scored,
                    selected_nll.to(numeric.dtype),
                    torch.zeros_like(numeric),
                )
                categorical_accuracy = _masked_mean(
                    (probabilities.argmax(dim=-1) == safe_codes).to(probabilities.dtype),
                    categorical_scored,
                )

        completion_targets = prediction.auxiliaries.get("completion_target_mask", model_targets).to(
            device=model_targets.device, dtype=torch.bool
        )
        label_targets = prediction.auxiliaries.get(
            "label_target_mask", torch.zeros_like(model_targets)
        ).to(device=model_targets.device, dtype=torch.bool)
        if (
            completion_targets.shape != model_targets.shape
            or label_targets.shape != model_targets.shape
        ):
            raise ValueError("completion/label family masks must match target_mask")
        if bool((completion_targets & label_targets).any()):
            raise ValueError("completion and label target families must be disjoint")
        if bool((truth_targets & ~(completion_targets | label_targets)).any()):
            raise ValueError("factual truth must belong to completion F or label L family")

        zero = numeric.sum() * 0.0

        def family_loss(family_mask: Tensor) -> tuple[Tensor, tuple[str, ...]]:
            typed_losses: list[Tensor] = []
            active_types: list[str] = []
            family_numeric = numeric_scored & family_mask
            if bool(family_numeric.any()):
                family_numeric_loss = (
                    self.mse_weight * _masked_mean(numeric_error_cells.square(), family_numeric)
                    + self.mae_weight * _masked_mean(numeric_error_cells.abs(), family_numeric)
                )
                typed_losses.append(family_numeric_loss)
                active_types.append("numeric")
            family_categorical = categorical_scored & family_mask
            if self.include_categorical and bool(family_categorical.any()):
                typed_losses.append(
                    self.categorical_nll_weight
                    * _masked_mean(categorical_nll_cells, family_categorical)
                )
                active_types.append("categorical")
            loss = torch.stack(typed_losses).mean() if typed_losses else zero
            return loss, tuple(active_types)

        completion_loss, completion_types = family_loss(completion_targets)
        label_loss, label_types = family_loss(label_targets)
        active_losses: list[Tensor] = []
        active_families: list[str] = []
        if completion_types:
            active_losses.append(completion_loss)
            active_families.append("F")
        if label_types:
            active_losses.append(label_loss)
            active_families.append("L")
        total = torch.stack(active_losses).mean() if active_losses else numeric.sum() * 0.0

        target_count = int(truth_targets.sum().item())
        declared_count = int(model_targets.sum().item())
        numeric_scored_count = int(numeric_scored.sum().item())
        categorical_scored_count = int(categorical_scored.sum().item())
        scored_count = numeric_scored_count + categorical_scored_count
        return LossBundle(
            episode_id=prediction.episode_id,
            total=total,
            components={
                "categorical_accuracy": categorical_accuracy,
                "categorical_nll": categorical_nll,
                "completion_loss": completion_loss,
                "label_loss": label_loss,
                "mae": mae,
                "mse": mse,
                "numeric_loss": numeric_loss,
            },
            counts={
                "abstained_targets": target_count - scored_count,
                "active_families": len(active_families),
                "categorical_scored_targets": categorical_scored_count,
                "categorical_targets": int((truth_targets & categorical_targets).sum().item()),
                "completion_scored_targets": int(
                    ((numeric_scored | categorical_scored) & completion_targets).sum().item()
                ),
                "completion_targets": int((truth_targets & completion_targets).sum().item()),
                "declared_targets": declared_count,
                "no_truth_targets": declared_count - target_count,
                "numeric_scored_targets": numeric_scored_count,
                "numeric_targets": int((truth_targets & numeric_targets).sum().item()),
                "label_scored_targets": int(
                    ((numeric_scored | categorical_scored) & label_targets).sum().item()
                ),
                "label_targets": int((truth_targets & label_targets).sum().item()),
                "scored_targets": scored_count,
                "targets": target_count,
            },
            metadata={
                "objective": (
                    "mixed_active_family_equal" if self.include_categorical else "numeric"
                ),
                "active_families": tuple(active_families),
                "completion_active_types": completion_types,
                "categorical_nll_weight": self.categorical_nll_weight,
                "family_reduction": "equal_over_active_families",
                "label_active_types": label_types,
                "type_reduction": "equal_within_each_active_family",
                "mae_weight": self.mae_weight,
                "mse_weight": self.mse_weight,
                "numeric_target_coordinate": self.numeric_target_coordinate,
                "status": (
                    "no_truth" if target_count == 0 else "no_support" if scored_count == 0 else "ok"
                ),
                "unsupported_policy": "exclude_and_count",
            },
        )


class NumericObjective(MixedObjective):
    """Compatibility specialization that scores only the numeric family."""

    def __init__(
        self,
        *,
        mse_weight: float = 1.0,
        mae_weight: float = 0.0,
        numeric_target_coordinate: NumericTargetCoordinate = "raw",
    ) -> None:
        super().__init__(
            mse_weight=mse_weight,
            mae_weight=mae_weight,
            include_categorical=False,
            numeric_target_coordinate=numeric_target_coordinate,
        )


Objective = MixedObjective


__all__ = ["MixedObjective", "NumericObjective", "NumericTargetCoordinate", "Objective"]
