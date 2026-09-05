"""TabU-v2 cell-as-query reference implementation.

This module is intentionally independent from the historical ``TabUR``
classes.  It implements the current structural contract directly:

``visible values -> cell queries -> (N+K)x(M+K) carrier -> column OMAB
-> row OMAB -> hierarchical response field -> same-column terminal``.

The model never accepts or imports a :class:`TruthSidecar` during forward.
Loss and evaluation remain outside this module at the typed objective boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import Tensor, nn

from tabu_lab.primitives import SameColumnNumericLocalLinear, SameColumnNumericNW

from .components import CellTokenizer, SymbolTable, TokenTable
from .dynamics import WholeTableDynamics
from .reference import DenseReferenceModel, _shape_event
from .types import DenseModelInput, ReferenceConfig


class TabUV2CellAsQueryModel(DenseReferenceModel):
    """Canonical TabU-v2 model for whole-table masked parallel completion."""

    model_id = "tabu.v2.tabur"

    def __init__(
        self,
        config: ReferenceConfig | None = None,
        *,
        k: int | None = None,
        lambda_F: float = 0.0,
        lambda_U: float = 0.0,
        numeric_terminal: str = "local_linear",
        nominal_tokenizer: str = CellTokenizer.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
        feature_address: bool = False,
        context_terminal: str | None = None,
    ) -> None:
        config = config or ReferenceConfig()
        # DenseReferenceModel owns the public evidence boundary.  TabU-v2
        # keeps its query identity in this contract-specific class instead of
        # changing the shared base constructor.
        super().__init__(config)

        resolved_k = config.matched_slots if k is None else k
        if (
            isinstance(resolved_k, bool)
            or not isinstance(resolved_k, int)
            or resolved_k <= 0
        ):
            raise ValueError("k must be a positive integer")
        if lambda_F < 0.0 or lambda_U < 0.0:
            raise ValueError("lambda_F and lambda_U must be non-negative")
        if lambda_F == 0.0 and lambda_U != 0.0:
            raise ValueError("lambda_U is only active when lambda_F is non-zero")
        resolved_terminal = getattr(numeric_terminal, "value", numeric_terminal)
        if resolved_terminal not in {"local_linear", "nadaraya_watson"}:
            raise ValueError("numeric_terminal must be local_linear or nadaraya_watson")

        self.k = resolved_k
        self.lambda_F = float(lambda_F)
        self.lambda_U = float(lambda_U)
        self.numeric_terminal = resolved_terminal
        self.nominal_tokenizer = nominal_tokenizer
        self.nominal_codebook_size = int(nominal_codebook_size)
        self.nominal_codebook_seed = int(nominal_codebook_seed)
        self.feature_address = bool(feature_address)
        if context_terminal not in {None, "linear", "mlp", "xgboost"}:
            raise ValueError("context_terminal must be None, linear, mlp, or xgboost")
        self.context_terminal = context_terminal
        self.tokenizer = CellTokenizer(
            config,
            marker="query",
            nominal_tokenizer=nominal_tokenizer,
            nominal_codebook_size=nominal_codebook_size,
            nominal_codebook_seed=nominal_codebook_seed,
            feature_address=feature_address,
        )
        self.dynamics = WholeTableDynamics(config)
        self.unit_query = nn.Parameter(torch.empty(resolved_k, config.d_model))
        self.feature_query = nn.Parameter(torch.empty(resolved_k, config.d_model))
        self.response_base = nn.Parameter(torch.empty(resolved_k, config.d_model))
        for parameter in (self.unit_query, self.feature_query, self.response_base):
            nn.init.normal_(parameter, std=0.02)
        self.terminal = (
            SameColumnNumericNW(config.routing_bandwidth)
            if resolved_terminal == "nadaraya_watson"
            else SameColumnNumericLocalLinear(config.routing_bandwidth)
        )

    def _context_terminal_outputs(
        self,
        inputs: DenseModelInput,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        """Fit an explicit context-only reference terminal, when requested.

        This optional diagnostic variant consumes only labeled SOURCE cells in
        the current episode.  It is deliberately separate from the learned
        TabU-v2 terminal: the canonical model remains optimizer-free and
        permutation-aware, while this adapter makes a like-for-like ICL
        comparison against the repository's fixed classical estimators.
        """

        if self.context_terminal is None:
            return None, None, None
        import warnings

        import numpy as np
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import StandardScaler

        response_index = len(inputs.feature_specs) - 1
        for index, spec in enumerate(inputs.feature_specs):
            if (
                getattr(getattr(spec, "role", None), "value", getattr(spec, "role", None))
                == "response"
            ):
                response_index = index
                break
        if response_index < 0 or response_index >= inputs.values.shape[-1]:
            raise ValueError("context terminal requires a declared response feature")
        predictors = [index for index in range(inputs.values.shape[-1]) if index != response_index]
        if not predictors:
            raise ValueError("context terminal requires at least one predictor")
        response_spec = inputs.feature_specs[response_index]
        response_kind = getattr(
            getattr(response_spec, "kind", None),
            "value",
            getattr(response_spec, "kind", "numeric"),
        )
        is_categorical = response_kind == "categorical"
        class_count = len(getattr(response_spec, "domain", ())) if is_categorical else 0
        if is_categorical and class_count < 2:
            raise ValueError("categorical context terminal requires a declared response domain")
        seed = int(inputs.metadata.get("split_seed", 1729)) if inputs.metadata else 1729
        batch, n_rows, n_features = inputs.values.shape
        numeric_override = torch.zeros_like(inputs.values) if not is_categorical else None
        categorical_override = (
            torch.zeros(
                batch,
                n_rows,
                n_features,
                class_count,
                dtype=inputs.values.dtype,
                device=inputs.values.device,
            )
            if is_categorical
            else None
        )
        override_mask = torch.zeros_like(inputs.values, dtype=torch.bool)

        for batch_index in range(batch):
            context_rows = inputs.visible_mask[batch_index, :, response_index]
            target_rows = (
                inputs.target_mask[batch_index, :, response_index]
                & ~inputs.unsupported_target_mask[batch_index, :, response_index]
            )
            if not bool(context_rows.any()) or not bool(target_rows.any()):
                continue
            context_positions = context_rows.nonzero(as_tuple=False).flatten().tolist()
            # The fixed baseline receipts fit on ``train_indices`` in source
            # row order.  Evidence may use a class-balanced context order for
            # ICL, so restore the provenance order inside this explicit
            # classical adapter; the canonical learned terminal is unaffected.
            if inputs.row_ids:
                def row_number(position: int) -> int:
                    value = inputs.row_ids[position].rsplit("-", 1)[-1]
                    return int(value) if value.isdigit() else position

                fit_order = (
                    inputs.metadata.get("context_fit_row_order")
                    if inputs.metadata
                    else None
                )
                if isinstance(fit_order, list | tuple):
                    fit_rank = {int(row): rank for rank, row in enumerate(fit_order)}
                    context_positions.sort(
                        key=lambda position: fit_rank.get(
                            row_number(position), len(fit_rank) + position
                        )
                    )
                else:
                    context_positions.sort(key=row_number)
            context_index = torch.as_tensor(
                context_positions, dtype=torch.long, device=inputs.values.device
            )
            x_train = (
                inputs.values[batch_index, context_index][:, predictors]
                .detach()
                .cpu()
                .numpy()
            )
            x_query = inputs.values[batch_index, target_rows][:, predictors].detach().cpu().numpy()
            y_train = (
                inputs.values[batch_index, context_index, response_index]
                .detach()
                .cpu()
                .numpy()
            )
            estimator_name = self.context_terminal
            if is_categorical:
                y_train = y_train.astype(np.int64)
                if estimator_name == "linear":
                    scaler = StandardScaler().fit(x_train)
                    estimator = LogisticRegression(
                        C=1.0, max_iter=2000, solver="lbfgs", random_state=seed
                    )
                    train_features = scaler.transform(x_train)
                    query_features = scaler.transform(x_query)
                elif estimator_name == "mlp":
                    scaler = StandardScaler().fit(x_train)
                    estimator = MLPClassifier(
                        hidden_layer_sizes=(64, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1.0e-4,
                        batch_size=min(64, len(y_train)),
                        learning_rate_init=1.0e-3,
                        max_iter=500,
                        early_stopping=False,
                        shuffle=True,
                        tol=1.0e-4,
                        random_state=seed,
                    )
                    train_features = scaler.transform(x_train)
                    query_features = scaler.transform(x_query)
                else:
                    estimator = __import__("xgboost").XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=8,
                        tree_method="hist",
                        random_state=seed,
                        eval_metric="logloss",
                    )
                    train_features, query_features = x_train, x_query
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(train_features, y_train)
                probabilities = np.asarray(
                    estimator.predict_proba(query_features), dtype=np.float64
                )
                aligned = np.zeros((len(x_query), class_count), dtype=np.float64)
                for column, label in enumerate(
                    np.asarray(estimator.classes_, dtype=np.int64).tolist()
                ):
                    aligned[:, label] = probabilities[:, column]
                aligned = np.clip(aligned, 1.0e-12, None)
                aligned /= aligned.sum(axis=1, keepdims=True)
                categorical_override[batch_index, target_rows, response_index] = torch.as_tensor(
                    aligned, device=inputs.values.device, dtype=inputs.values.dtype
                )
            else:
                if estimator_name == "linear":
                    scaler = StandardScaler().fit(x_train)
                    estimator = Ridge(alpha=1.0)
                    train_features = scaler.transform(x_train)
                    query_features = scaler.transform(x_query)
                elif estimator_name == "mlp":
                    scaler = StandardScaler().fit(x_train)
                    target_mean = float(y_train.mean())
                    target_scale = max(float(y_train.std()), 1.0e-6)
                    estimator = MLPRegressor(
                        hidden_layer_sizes=(64, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1.0e-4,
                        batch_size=min(64, len(y_train)),
                        learning_rate_init=1.0e-3,
                        max_iter=500,
                        early_stopping=False,
                        shuffle=True,
                        tol=1.0e-4,
                        random_state=seed,
                    )
                    train_features = scaler.transform(x_train)
                    query_features = scaler.transform(x_query)
                    y_train = (y_train - target_mean) / target_scale
                else:
                    estimator = __import__("xgboost").XGBRegressor(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        n_jobs=8,
                        tree_method="hist",
                        random_state=seed,
                        objective="reg:squarederror",
                        eval_metric="rmse",
                    )
                    train_features, query_features = x_train, x_query
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    estimator.fit(train_features, y_train)
                predicted = np.asarray(estimator.predict(query_features), dtype=np.float64)
                if estimator_name == "mlp":
                    predicted = predicted * target_scale + target_mean
                numeric_override[batch_index, target_rows, response_index] = torch.as_tensor(
                    predicted, device=inputs.values.device, dtype=inputs.values.dtype
                )
            override_mask[batch_index, target_rows, response_index] = True
        return numeric_override, categorical_override, override_mask

    def _compile_extended_carrier(
        self,
        inputs: DenseModelInput,
    ) -> tuple[SymbolTable, TokenTable, Tensor, Tensor]:
        """Compile Step 1--3 without allowing target truth into the carrier."""

        symbols = self.symbolizer(inputs)
        # TabU-v2 has one shared Cell Query seed.  Artificial masks and QUERY
        # origins are both receiver queries; natural missing cells remain exact
        # zero and unsupported targets are never made into factual sources.
        eligible_query = inputs.target_mask & ~inputs.unsupported_target_mask
        query_symbols = replace(
            symbols,
            artificial_target_mask=torch.zeros_like(inputs.target_mask),
            query_target_mask=eligible_query,
        )
        tokens = self.tokenizer(query_symbols)
        cells = tokens.cells
        batch, n_rows, n_features, d_model = cells.shape
        carrier = cells.new_zeros(
            batch,
            n_rows + self.k,
            n_features + self.k,
            d_model,
        )
        carrier[:, :n_rows, :n_features] = cells
        unit_queries = self.unit_query.to(device=cells.device, dtype=cells.dtype).view(
            1, 1, self.k, d_model
        )
        feature_queries = self.feature_query.to(
            device=cells.device,
            dtype=cells.dtype,
        ).view(1, self.k, 1, d_model)
        carrier[:, :n_rows, n_features:] = unit_queries.expand(batch, n_rows, -1, -1)
        carrier[:, n_rows:, :n_features] = feature_queries.expand(
            batch, -1, n_features, -1
        )

        # This is the only factual source gate.  Unit Query, Feature Query,
        # Cell Query and Null positions are receiver-only structural states.
        source_mask = torch.zeros(
            batch,
            n_rows + self.k,
            n_features + self.k,
            dtype=torch.bool,
            device=cells.device,
        )
        source_mask[:, :n_rows, :n_features] = inputs.visible_mask
        return symbols, tokens, carrier, source_mask

    def _response_field(
        self,
        carrier: Tensor,
        *,
        n_rows: int,
        n_features: int,
    ) -> Tensor:
        """Apply the hierarchical response law ``z[r,a] = A[r,a] c[r,a]``."""

        if carrier.ndim != 4:
            raise ValueError("carrier must be [B,N+K,M+K,D]")
        if carrier.shape[1] < n_rows or carrier.shape[2] < n_features:
            raise ValueError("carrier is smaller than the ordinary table")
        cells = carrier[:, :n_rows, :n_features]
        unit_states = carrier[:, :n_rows, n_features : n_features + self.k]
        feature_states = carrier[:, n_rows : n_rows + self.k, :n_features].permute(
            0, 2, 1, 3
        )
        base = self.response_base.to(device=carrier.device, dtype=carrier.dtype).view(
            1, 1, 1, self.k, carrier.shape[-1]
        )
        address = base + self.lambda_F * (
            feature_states.unsqueeze(1) + self.lambda_U * unit_states.unsqueeze(2)
        )
        return torch.einsum("bnmd,bnmkd->bnmk", cells, address)

    @staticmethod
    def _null_mask(
        source_mask: Tensor,
        *,
        n_rows: int,
        n_features: int,
    ) -> Tensor:
        null_mask = torch.zeros_like(source_mask)
        null_mask[:, n_rows:, n_features:] = True
        return null_mask

    def _forward_dense(self, inputs: Any, **kwargs: Any) -> Any:
        emit_trace = bool(kwargs.get("emit_trace", True))
        resolved = self._resolve_inputs(
            inputs,
            visible_mask=kwargs.get("visible_mask"),
            target_mask=kwargs.get("target_mask"),
            natural_missing_mask=kwargs.get("natural_missing_mask"),
            graph=kwargs.get("graph"),
            target_feature=kwargs.get("target_feature"),
            episode_id=kwargs.get("episode_id"),
        )
        symbols, tokens, carrier_input, source_mask = self._compile_extended_carrier(
            resolved
        )
        carrier = self.dynamics(carrier_input, source_mask=source_mask)
        n_rows, n_features = resolved.values.shape[1:]
        response_field = self._response_field(
            carrier,
            n_rows=n_rows,
            n_features=n_features,
        )
        numeric_scale_state = tokens.numeric_scale_state
        if numeric_scale_state is None:
            raise RuntimeError("TabU-v2 requires the tokenizer numeric scale state")
        readout = self.terminal(
            response_field,
            numeric_scale_state.standardized_values,
            resolved.visible_mask,
        )
        numeric_override, categorical_override, override_mask = self._context_terminal_outputs(
            resolved
        )
        if numeric_override is not None and override_mask is not None:
            numeric_override = (
                numeric_override - numeric_scale_state.mean
            ) / numeric_scale_state.scale.clamp_min(torch.finfo(numeric_override.dtype).tiny)
            readout = replace(
                readout,
                values=torch.where(override_mask, numeric_override, readout.values),
            )
        numeric_raw_prediction = (
            readout.values * numeric_scale_state.scale + numeric_scale_state.mean
        )
        null_mask = self._null_mask(
            source_mask,
            n_rows=n_rows,
            n_features=n_features,
        )
        tokenizer_metadata = {
            "tokenizer_version": "cell-query-tokenizer.v2",
            "nominal_tokenizer": self.nominal_tokenizer,
            "nominal_codebook_size": self.nominal_codebook_size,
            "nominal_codebook_seed": self.nominal_codebook_seed,
            "cell_query_seed": "shared",
            "feature_address": self.feature_address,
        }
        common_metadata: Mapping[str, Any] = {
            "dynamics_plan": self._dynamics_plan_name(self.dynamics),
            "unit": "cell_query",
            "family_id": "tabu.v2.cell_as_query",
            "carrier_role": "value_query_null_extended",
            "numeric_terminal": self.numeric_terminal,
            "response_field": "hierarchical_Ac",
            "query_subtokens": self.k,
            "lambda_F": self.lambda_F,
            "lambda_U": self.lambda_U,
            "carrier_shape": tuple(carrier.shape),
            "source_gate": "visible_value_cells_only",
            "truth_boundary": "objective_sidecar_only",
            "context_terminal": self.context_terminal,
            **tokenizer_metadata,
        }
        events = (
            _shape_event(
                "symbolizer",
                symbols.values,
                input_tensor=resolved.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                supervision_boundary="absent",
            ),
            _shape_event(
                "tokenizer",
                tokens.cells,
                input_tensor=symbols.values,
                source_mask=resolved.visible_mask,
                null_mask=resolved.natural_missing_mask,
                **tokenizer_metadata,
            ),
            _shape_event(
                "extended_carrier",
                carrier_input,
                input_tensor=tokens.cells,
                source_mask=source_mask,
                null_mask=null_mask,
                operation_trace=(
                    "compile_value_cells",
                    "append_unit_queries",
                    "append_feature_queries",
                    "append_null_corner",
                ),
                carrier_role="value_query_null_extended",
            ),
            _shape_event(
                "dynamics_plan",
                carrier,
                input_tensor=carrier_input,
                source_mask=source_mask,
                null_mask=null_mask,
                operation_trace=self.dynamics.plan.stages,
                plan=self._dynamics_plan_name(self.dynamics),
            ),
            _shape_event(
                "response_field",
                response_field,
                input_tensor=carrier,
                source_mask=source_mask[:, :n_rows, :n_features],
                operation_trace=(
                    "hierarchical_response_address",
                    "response_state_field",
                ),
                response_law="A=W+lambda_F(F+lambda_U U); z=A c",
            ),
            _shape_event(
                "readout",
                readout.values,
                input_tensor=response_field,
                source_mask=resolved.visible_mask,
                operation_trace=(
                    "same_column_visible_support",
                    f"numeric_{self.numeric_terminal}",
                ),
                terminal=f"numeric_{self.numeric_terminal}",
                geometry="same_column_response_field",
                numeric_prediction_scale="context_standardized",
            ),
            _shape_event(
                "prediction_boundary",
                resolved.target_mask,
                input_tensor=readout.values,
                source_mask=resolved.visible_mask,
                operation_trace=("model_forward_complete",),
                supervision_boundary="sidecar_only",
                truth_not_available=True,
                model_forward_complete=True,
            ),
        )
        return self._bundle(
            inputs=resolved,
            values=readout.values,
            support_available=readout.support_available,
            coordinates=response_field,
            routing_weights=readout.routing.weights,
            routing_log_weights=readout.routing.log_weights,
            routing_support_mask=readout.routing.support_mask,
            categorical_probabilities_override=categorical_override,
            categorical_override_mask=override_mask,
            extra_auxiliaries={
                "numeric_raw_prediction": numeric_raw_prediction,
                "numeric_context_mean": numeric_scale_state.mean,
                "numeric_context_std": numeric_scale_state.std,
                "numeric_context_scale": numeric_scale_state.scale,
                "numeric_context_count": numeric_scale_state.context_count,
            },
            events=events,
            metadata={
                **common_metadata,
                "numeric_prediction_scale": "context_standardized",
            },
            emit_trace=emit_trace,
        )

    def checkpoint_identity(self) -> dict[str, Any]:
        """Return the semantic identity needed before loading v2 weights."""

        return {
            "model_id": self.model_id,
            "contract_version": self.contract_version,
            "model_spec_hash": self.model_spec_hash,
            "query_subtokens": self.k,
            "lambda_F": self.lambda_F,
            "lambda_U": self.lambda_U,
            "numeric_terminal": self.numeric_terminal,
            "nominal_tokenizer": self.nominal_tokenizer,
            "nominal_codebook_size": self.nominal_codebook_size,
            "nominal_codebook_seed": self.nominal_codebook_seed,
            "feature_address": self.feature_address,
            "context_terminal": self.context_terminal,
            "reference_config": self.config.semantic_hash,
        }

    def validate_checkpoint_identity(self, identity: Mapping[str, Any]) -> None:
        expected = self.checkpoint_identity()
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(
                    f"checkpoint identity mismatch at {key}: expected {value!r}"
                )


__all__ = ["TabUV2CellAsQueryModel"]
