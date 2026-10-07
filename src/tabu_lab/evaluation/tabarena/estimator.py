"""Frozen ICL adapter with replaceable support, query and probability strategies.

No gradient updates, target validation split, or query-fitted moments occur.
Visible Query features may choose the versioned whole-episode softlog switch.
The defaults are an integration baseline, not a selected competition recipe.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

from tabu_lab.models.restoration.contracts import ColumnSchema, RestorationInput
from tabu_lab.models.restoration_v7.codec import build_value_codec, select_softlog
from tabu_lab.models.restoration_v7.model import V7Episode

from .network import execution_dtype, load_network


def _implementation(method):
    return f"{method.__module__}.{method.__qualname__}"


def _row_indices(indices, size, name):
    indices = np.asarray(indices)
    if (
        indices.ndim != 1
        or not len(indices)
        or indices.dtype.kind not in "iu"
        or np.any(indices < 0)
        or np.any(indices >= size)
        or len(np.unique(indices)) != len(indices)
    ):
        raise ValueError(f"{name} must contain unique, in-range integer row indices")
    return indices.astype(np.intp)


class _TabUV7Base(BaseEstimator):
    def __init__(
        self,
        checkpoint_path=None,
        checkpoint_sha256=None,
        device="cpu",
        random_state=0,
        max_support_rows=256,
        query_mode="singleton",
        temperature=1.0,
        network=None,
        constant_support_numeric="null",
        legacy_overlay=None,
    ):
        self.checkpoint_path = checkpoint_path
        self.checkpoint_sha256 = checkpoint_sha256
        self.device = device
        self.random_state = random_state
        self.max_support_rows = max_support_rows
        self.query_mode = query_mode
        self.temperature = temperature
        self.network = network
        self.constant_support_numeric = constant_support_numeric
        self.legacy_overlay = legacy_overlay

    def _frame(self, X, fitting=False):
        frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        if not frame.columns.is_unique:
            raise ValueError("Feature names must be unique")
        if fitting:
            self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
            self.n_features_in_ = frame.shape[1]
        elif list(frame.columns) != list(self.feature_names_in_):
            raise ValueError("Prediction columns must match fitted columns in order")
        return frame.reset_index(drop=True)

    def fit(self, X, y):
        if self.constant_support_numeric not in ("iqr_null", "null", "keep"):
            raise ValueError("constant_support_numeric must be iqr_null, null or keep")
        if type(self)._query_groups is _TabUV7Base._query_groups and self.query_mode not in (
            "singleton",
            "joint",
        ):
            raise ValueError("query_mode must be singleton or joint for the built-in grouping")
        if type(self.random_state) is not int or not 0 <= self.random_state < 2**32:
            raise ValueError("random_state must be an integer in [0, 2**32)")
        if self.max_support_rows is not None and (
            type(self.max_support_rows) is not int or self.max_support_rows < 1
        ):
            raise ValueError("max_support_rows must be a positive integer or None")
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        frame = self._frame(X, fitting=True)
        labels = np.asarray(y)
        if (
            labels.ndim != 1
            or len(labels) != len(frame)
            or not len(labels)
            or pd.isna(labels).any()
        ):
            raise ValueError("y must be nonempty, complete, one dimensional, and row-aligned")
        if self._classification:
            self.label_encoder_ = LabelEncoder().fit(labels)
            self.classes_ = self.label_encoder_.classes_
            labels = self.label_encoder_.transform(labels)
        else:
            labels = labels.astype(np.float64)
            if not np.isfinite(labels).all():
                raise ValueError("Regression targets must be finite")
        self.support_indices_ = _row_indices(
            self._select_support_indices(frame.copy(), labels.copy()),
            len(labels),
            "Support selection",
        )
        limit = len(self.support_indices_)
        if self.max_support_rows is not None and limit > self.max_support_rows:
            raise ValueError("Support selection exceeds max_support_rows")
        if self._classification and len(np.unique(labels[self.support_indices_])) != len(
            self.classes_
        ):
            raise ValueError("Support selection cannot discard target classes")
        support = frame.iloc[self.support_indices_]
        self.columns_ = []
        self.schema_ = []
        values, masks = [], []
        for position, name in enumerate(frame.columns):
            series = support[name]
            numeric = pd.api.types.is_numeric_dtype(
                series.dtype
            ) and not pd.api.types.is_bool_dtype(series.dtype)
            if numeric:
                raw = pd.to_numeric(series).to_numpy(dtype=np.float64, na_value=np.nan)
                visible = np.isfinite(raw)
                spec = {"name": name, "kind": "numeric"}
                encoded = np.where(visible, raw, 0.0)
                schema = ColumnSchema(f"feature_{position}", "numeric")
                # Fit the low-variation decision on finite selected Support only.
                # Quantile equality avoids subtracting extreme endpoints; the
                # midpoint interpolation below avoids overflow in b-a.
                observed = raw[visible]
                low_variation = False
                if len(observed) and self.constant_support_numeric != "keep":
                    low_variation = bool(np.all(observed == observed[0]))
                    if self.constant_support_numeric == "iqr_null":
                        ordered = np.sort(observed)

                        def quantile(q, ordered=ordered):
                            index = (len(ordered) - 1) * q
                            lo, hi = int(np.floor(index)), int(np.ceil(index))
                            fraction = index - lo
                            return (1 - fraction) * ordered[lo] + fraction * ordered[hi]

                        low_variation = bool(quantile(0.25) == quantile(0.75))
                if low_variation:
                    spec["support_constant_null"] = True
                    encoded = np.zeros_like(raw)
                    visible = np.zeros_like(visible)
            else:
                if pd.api.types.is_datetime64_any_dtype(series.dtype):
                    raise ValueError("Datetime columns need an explicit upstream representation")
                categories = list(pd.unique(series.dropna()))
                mapping = {value: i for i, value in enumerate(categories)}
                mapped = series.astype(object).map(mapping)
                visible = mapped.notna().to_numpy()
                encoded = mapped.fillna(0).to_numpy(dtype=np.int64)
                spec = {"name": name, "kind": "nominal", "mapping": mapping}
                schema = ColumnSchema(f"feature_{position}", "nominal", max(1, len(categories)))
            if not visible.any() and not spec.get("support_constant_null", False):
                spec["support_empty_null"] = True  # Keep column address; no fitted payload.
            self.columns_.append(spec)
            self.schema_.append(schema)
            values.append(encoded)
            masks.append(visible)
        self.schema_.append(
            ColumnSchema("target", "nominal", len(self.classes_))
            if self._classification
            else ColumnSchema("target", "numeric")
        )
        values.append(labels[self.support_indices_])
        masks.append(np.ones(limit, dtype=bool))
        self.schema_ = tuple(self.schema_)
        self.support_values_ = tuple(np.asarray(value).copy() for value in values)
        self.support_visible_ = np.column_stack(masks)
        self.network_ = (
            self.network
            if self.network is not None
            else load_network(
                self.checkpoint_path,
                self.checkpoint_sha256,
                self.device,
                legacy_overlay=self.legacy_overlay,
            )
        )
        if (
            getattr(self.network_, "checkpoint_sha256", None) != self.checkpoint_sha256
            or not self.checkpoint_sha256
        ):
            raise ValueError("Injected network must carry the requested checkpoint SHA256")
        inference_protocol = getattr(self.network_, "inference_protocol", None)
        selected_overlay = (
            inference_protocol.get("legacy_overlay") if inference_protocol is not None else None
        )
        if selected_overlay != self.legacy_overlay:
            raise ValueError("Injected network does not carry the requested legacy_overlay receipt")
        self.protocol_ = {
            "adapter_version": "0.2.0",
            "query_mode": self.query_mode
            if type(self)._query_groups is _TabUV7Base._query_groups
            else "custom; see strategies",
            "query_mode_parameter": self.query_mode,
            "estimator": f"{type(self).__module__}.{type(self).__qualname__}",
            "strategies": {
                "support_selection": _implementation(self._select_support_indices),
                "query_grouping": _implementation(self._query_groups),
                "probability_readout": _implementation(self._readout_probabilities)
                if self._classification
                else None,
            },
            "support_rows": limit,
            "support_policy": (
                "seeded subset retaining every target class"
                if self._classification
                else "seeded subset"
            )
            if type(self)._select_support_indices is _TabUV7Base._select_support_indices
            else "custom; see strategies",
            "codec_policy": (
                "S: selected Support; V: current episode Visible; whole-episode softlog"
            ),
            "numeric_preprocessing": self.network_.config.numeric_preprocessing,
            "constant_support_numeric": self.constant_support_numeric,
            "numeric_null_criterion": "support_q25_equals_q75_linear"
            if self.constant_support_numeric == "iqr_null"
            else self.constant_support_numeric,
            "constant_numeric_null_columns": [
                s["name"] for s in self.columns_ if s.get("support_constant_null", False)
            ],
            "missing_and_unseen_features": (
                "Null cells; empty and constant column addresses retained"
            ),
            "donor_policy": (
                "SHA256 of encoded visible query features and random_state, modulo support count"
            ),
            "probabilities": (
                "softmax_negative_squared_code_distance"
                if type(self)._readout_probabilities is TabUV7Classifier._readout_probabilities
                else "custom; see strategies"
            )
            if self._classification
            else None,
            "temperature": self.temperature,
            "checkpoint_sha256": self.checkpoint_sha256,
            "legacy_overlay_parameter": self.legacy_overlay,
            "network_inference_protocol": (
                dict(inference_protocol) if inference_protocol is not None else None
            ),
            "gradient_updates": 0,
        }
        return self.to(self.device)

    def _select_support_indices(self, X, y):
        """Override to select unique training rows; y is encoded for classification.

        Respect max_support_rows and retain every target class. This hook receives
        training data only. Keep any learned strategy state on the fitted estimator.
        """
        order = np.random.default_rng(self.random_state).permutation(len(y))
        limit = len(y) if self.max_support_rows is None else min(self.max_support_rows, len(y))
        if self._classification:
            if limit < len(self.classes_):
                raise ValueError("max_support_rows cannot discard target classes")
            mandatory = [
                next(int(i) for i in order if y[i] == c) for c in range(len(self.classes_))
            ]
            used = set(mandatory)
            order = np.asarray(mandatory + [int(i) for i in order if int(i) not in used])
        return np.sort(order[:limit])

    def _query_groups(self, n_queries):
        """Override to partition query row indices; output order is restored below.

        Every input row must occur exactly once. Grouping changes model semantics;
        custom grouping must undergo its own quality and official contract checks.
        """
        if self.query_mode == "joint":
            return [np.arange(n_queries)]
        if self.query_mode == "singleton":
            return (np.asarray([i]) for i in range(n_queries))
        raise ValueError("query_mode must be singleton or joint for the built-in grouping")

    def _tensor_values(self, values):
        return tuple(
            torch.as_tensor(
                v,
                device=self.device,
                dtype=execution_dtype(self.device) if s.kind == "numeric" else torch.long,
            )
            for s, v in zip(self.schema_, values, strict=True)
        )

    def to(self, device):
        target = torch.device(device)
        if hasattr(self, "network_"):
            current = next(self.network_.parameters()).device
            dtype = execution_dtype(target)
            # MPS cannot allocate FP64, even as an intermediate in a combined .to.
            if current.type == "mps" and target.type != "mps":
                self.network_.to(device=target).to(dtype=dtype)
            elif target.type == "mps" and current.type != "mps":
                self.network_.to(dtype=dtype).to(device=target)
            else:
                self.network_.to(device=target, dtype=dtype)
            self.network_.eval()
            self.device = str(target)
            visible = torch.as_tensor(self.support_visible_, device=self.device)
            inputs = RestorationInput(
                self.schema_,
                self._tensor_values(self.support_values_),
                visible,
                torch.zeros_like(visible),
                self.random_state,
            )
            self.codec_ = build_value_codec(
                inputs,
                codec=self.network_.config.codec,
                dim=self.network_.config.code_dim,
                epsilon=self.network_.config.epsilon,
                numeric_preprocessing=self.network_.config.numeric_preprocessing,
            )
        else:
            self.device = str(target)
        return self

    def _encode_query(self, X):
        frame = self._frame(X)
        values, masks = [], []
        for spec in self.columns_:
            series = frame[spec["name"]]
            if spec.get("support_constant_null", False) or spec.get("support_empty_null", False):
                # Null payloads also stay out of donor hashing and encoding.
                values.append(np.zeros(len(frame), dtype=np.float64))
                visible = np.zeros(len(frame), dtype=bool)
            elif spec["kind"] == "numeric":
                raw = pd.to_numeric(series, errors="raise").to_numpy(
                    dtype=np.float64, na_value=np.nan
                )
                visible = np.isfinite(raw)
                values.append(np.where(visible, raw, 0.0))
            else:
                mapped = series.astype(object).map(spec["mapping"])
                visible = mapped.notna().to_numpy()
                values.append(mapped.fillna(0).to_numpy(dtype=np.int64))
            masks.append(visible)
        values.append(np.zeros(len(frame), dtype=np.int64 if self._classification else np.float64))
        masks.append(np.zeros(len(frame), dtype=bool))
        return values, np.column_stack(masks)

    def _states(self, X):
        check_is_fitted(self, ["codec_", "network_", "support_values_"])
        values, visible = self._encode_query(X)
        nq = len(visible)
        if not nq:
            return torch.empty(
                0,
                self.network_.config.code_dim,
                device=self.device,
                dtype=execution_dtype(self.device),
            )
        ns = len(self.support_visible_)
        donors = []
        for i in range(nq):
            h = hashlib.sha256(self.random_state.to_bytes(4, "little"))
            h.update(visible[i].tobytes())
            for value in values[:-1]:
                # Explicit endian and zero normalization keep hashes portable.
                h.update(np.asarray([0 if value[i] == 0 else value[i]], dtype="<f8").tobytes())
            donors.append(int.from_bytes(h.digest()[:8], "little") % ns)
        outputs = torch.empty(
            nq,
            self.network_.config.code_dim,
            device=self.device,
            dtype=execution_dtype(self.device),
        )
        seen = np.zeros(nq, dtype=bool)
        with torch.inference_mode():
            for group in self._query_groups(nq):
                ids = _row_indices(group, nq, "Query group")
                if seen[ids].any():
                    raise ValueError("Query groups must cover each input row exactly once")
                seen[ids] = True
                combined = tuple(
                    np.concatenate((s, q[ids]))
                    for s, q in zip(self.support_values_, values, strict=True)
                )
                vis = torch.as_tensor(
                    np.concatenate((self.support_visible_, visible[ids])), device=self.device
                )
                qry = torch.zeros_like(vis)
                qry[ns:, -1] = True
                inputs = RestorationInput(
                    self.schema_, self._tensor_values(combined), vis, qry, self.random_state
                )
                codec = select_softlog(self.codec_, inputs)
                episode = V7Episode(
                    inputs,
                    codec,
                    len(self.schema_) - 1,
                    torch.arange(ns, device=self.device),
                    torch.arange(ns, ns + len(ids), device=self.device),
                    torch.as_tensor([donors[i] for i in ids], device=self.device),
                    codec.encode_observed(inputs),
                )
                state = self.network_(episode, decode=False).states[-1]
                if not bool(torch.isfinite(state).all()):
                    raise FloatingPointError("Nonfinite V7 output")
                outputs[torch.as_tensor(ids, device=self.device)] = state
        if not seen.all():
            raise ValueError("Query groups must cover each input row exactly once")
        return outputs

    def save(self, path):
        """Save a fitted estimator, including weights and support; trusted files only."""
        check_is_fitted(self, "network_")
        with Path(path).open("wb") as handle:
            pickle.dump(self, handle, protocol=5)

    @classmethod
    def load(cls, path):
        """Load a trusted local pickle. Never use this on untrusted files."""
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("Saved estimator type mismatch")
        return model


class TabUV7Classifier(ClassifierMixin, _TabUV7Base):
    _classification = True

    def predict_proba(self, X):
        states = self._states(X)
        probabilities = self._readout_probabilities(states)
        if (
            not isinstance(probabilities, torch.Tensor)
            or probabilities.shape != (len(states), len(self.classes_))
            or not bool(torch.isfinite(probabilities).all())
            or bool((probabilities < 0).any())
            or not torch.allclose(
                probabilities.sum(-1), probabilities.new_ones(len(states)), rtol=1e-5, atol=1e-6
            )
        ):
            raise ValueError(
                "Probability readout must return a finite nonnegative [N, C] tensor "
                "with row sums 1 in classes_ order"
            )
        return probabilities.detach().cpu().numpy()

    def _readout_probabilities(self, states):
        """Override the readout; return probabilities in classes_ order as a tensor.

        This default is a distance-based adapter extension, not a calibrated
        likelihood. Query targets or evaluation labels must not fit the readout.
        """
        column = self.codec_.columns[-1]
        # Every target class is retained; codec category order is mapped explicitly.
        probabilities = column.probabilities(states, temperature=self.temperature)
        result = torch.zeros_like(probabilities)
        result[:, column.categories] = probabilities
        return result

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return self.classes_[probabilities.argmax(axis=1)]


class TabUV7Regressor(RegressorMixin, _TabUV7Base):
    _classification = False

    def predict(self, X):
        states = self._states(X)
        return self.codec_.columns[-1].decode(states).detach().cpu().numpy()
