"""Explicit Step 1/2 components for dense reference models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE

from .types import DenseModelInput, ReferenceConfig


@dataclass(frozen=True)
class SymbolTable:
    values: Tensor
    visible_mask: Tensor
    target_mask: Tensor
    natural_missing_mask: Tensor
    graph: Tensor | None
    target_feature: int | None
    feature_kinds: tuple[str, ...]
    feature_domains: tuple[tuple[str, ...], ...] = ()
    feature_codebooks: tuple[str | None, ...] = ()
    artificial_target_mask: Tensor | None = None
    query_target_mask: Tensor | None = None
    episode_id: str = "dense-episode"
    # Rows/cells allowed to contribute train/context statistics.  Completion
    # uses visible evidence; supervised episodes exclude query rows so labels
    # can never affect their own normalization.
    context_mask: Tensor | None = None
    profile_id: str | None = None


class Symbolizer(nn.Module):
    """Step 1: preserve raw roles while exposing no target truth."""

    def forward(self, inputs: DenseModelInput) -> SymbolTable:
        if bool((inputs.values[~inputs.visible_mask] != 0).any()):
            raise ValueError("non-visible forward values must be physical zero")
        return SymbolTable(
            values=inputs.values,
            visible_mask=inputs.visible_mask,
            target_mask=inputs.target_mask,
            natural_missing_mask=inputs.natural_missing_mask,
            graph=inputs.graph,
            target_feature=inputs.target_feature,
            feature_kinds=tuple(
                getattr(getattr(spec, "kind", None), "value", "numeric")
                for spec in inputs.feature_specs
            )
            if inputs.feature_specs
            else tuple("numeric" for _ in range(inputs.values.shape[-1])),
            feature_domains=tuple(
                tuple(getattr(spec, "domain", ())) for spec in inputs.feature_specs
            )
            if inputs.feature_specs
            else tuple(() for _ in range(inputs.values.shape[-1])),
            feature_codebooks=tuple(
                getattr(spec, "codebook_id", None) for spec in inputs.feature_specs
            )
            if inputs.feature_specs
            else tuple(None for _ in range(inputs.values.shape[-1])),
            artificial_target_mask=inputs.artificial_target_mask,
            query_target_mask=inputs.query_target_mask,
            episode_id=inputs.episode_id,
            context_mask=inputs.metadata.get("context_mask") if inputs.metadata else None,
            profile_id=inputs.metadata.get("profile_id") if inputs.metadata else None,
        )


@dataclass(frozen=True)
class NumericScaleState:
    standardized_values: Tensor
    mean: Tensor
    std: Tensor
    scale: Tensor
    context_count: Tensor


@dataclass(frozen=True)
class TokenTable:
    cells: Tensor
    visible_mask: Tensor
    target_mask: Tensor
    natural_missing_mask: Tensor
    # TabUBase numeric terminals operate on the same context-standardized
    # scale as Step 1.  This stays optional so the legacy tokenizer carrier
    # remains source-compatible.
    numeric_scale_state: NumericScaleState | None = None


class CellTokenizer(nn.Module):
    """TabUBase Step 1--2 tokenizer from the current factory contract.

    Continuous values use context-only standardization followed by one shared
    learnable multi-scale Fourier lift.  Nominal values use an episode-seeded
    random sphere table, so no static category or feature identity is learned.
    Only the table-cell-as-Unit carrier is included in this model anchor.
    """

    _scale_epsilon = 1.0e-6

    EPISODE_RANDOM_SPHERE_V1 = "episode_random_sphere"
    SOURCE_SCOPED_FROZEN_CODEBOOK_V2 = "source_scoped_frozen_codebook.v2"
    DEFAULT_NOMINAL_CODEBOOK_SIZE = 100
    DEFAULT_NOMINAL_CODEBOOK_SEED = 1729

    def __init__(
        self,
        config: ReferenceConfig,
        *,
        marker: str = "mask",
        nominal_tokenizer: str = EPISODE_RANDOM_SPHERE_V1,
        nominal_codebook_size: int = 100,
        nominal_codebook_seed: int = 1729,
    ) -> None:
        super().__init__()
        if marker not in {"mask", "query"}:
            raise ValueError("marker must be 'mask' or 'query'")
        if nominal_tokenizer not in {
            self.EPISODE_RANDOM_SPHERE_V1,
            self.SOURCE_SCOPED_FROZEN_CODEBOOK_V2,
        }:
            raise ValueError("unknown nominal tokenizer plan")
        if (
            isinstance(nominal_codebook_size, bool)
            or not isinstance(nominal_codebook_size, int)
            or isinstance(nominal_codebook_seed, bool)
            or not isinstance(nominal_codebook_seed, int)
            or nominal_codebook_size < 2
            or nominal_codebook_seed < 0
        ):
            raise ValueError("nominal codebook size must exceed one and seed be non-negative")
        if nominal_tokenizer == self.EPISODE_RANDOM_SPHERE_V1 and (
            nominal_codebook_size != self.DEFAULT_NOMINAL_CODEBOOK_SIZE
            or nominal_codebook_seed != self.DEFAULT_NOMINAL_CODEBOOK_SEED
        ):
            raise ValueError(
                "episode_random_sphere v1 requires the canonical codebook size and seed; "
                "these controls are only configurable for source_scoped_frozen_codebook.v2"
            )
        self.config = config
        self.marker = marker
        self.nominal_tokenizer = nominal_tokenizer
        self.nominal_codebook_size = int(nominal_codebook_size)
        self.nominal_codebook_seed = int(nominal_codebook_seed)
        self.n_frequencies = max(1, config.d_model // 4)
        frequencies = torch.arange(
            1,
            self.n_frequencies + 1,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        self.log_frequencies = nn.Parameter(frequencies.log())
        self.continuous_value_encoder = nn.Linear(
            2 * self.n_frequencies,
            config.d_model,
            bias=False,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        self.mask_token = nn.Parameter(torch.empty(config.d_model, dtype=DEFAULT_FLOAT_DTYPE))
        self.query_token = nn.Parameter(torch.empty(config.d_model, dtype=DEFAULT_FLOAT_DTYPE))
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.query_token, std=0.02)
        if nominal_tokenizer == self.SOURCE_SCOPED_FROZEN_CODEBOOK_V2:
            generator = torch.Generator(device="cpu").manual_seed(self.nominal_codebook_seed)
            frozen_codebook = torch.nn.functional.normalize(
                torch.randn(
                    self.nominal_codebook_size,
                    config.d_model,
                    generator=generator,
                    dtype=DEFAULT_FLOAT_DTYPE,
                ),
                p=2.0,
                dim=-1,
                eps=1.0e-12,
            )
        else:
            frozen_codebook = torch.empty(0, config.d_model, dtype=DEFAULT_FLOAT_DTYPE)
        # The pool is deterministic from semantic config and is intentionally
        # absent from checkpoints.  This preserves v1 checkpoint parameter
        # compatibility while the codebook hash remains bound in variant identity.
        self.register_buffer("frozen_nominal_codebook", frozen_codebook, persistent=False)

    def _episode_seed(self, episode_id: str, feature_key: str) -> int:
        digest = hashlib.sha256(f"{episode_id}:nominal:{feature_key}".encode()).digest()
        return int.from_bytes(digest[:8], "little") % (2**63 - 1)

    def _random_sphere(
        self,
        *,
        episode_id: str,
        feature_key: str,
        count: int,
        device: torch.device,
    ) -> Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._episode_seed(episode_id, feature_key))
        samples = torch.randn(
            count,
            self.config.d_model,
            generator=generator,
            dtype=DEFAULT_FLOAT_DTYPE,
        )
        return torch.nn.functional.normalize(samples, p=2.0, dim=-1, eps=1.0e-12).to(device)

    @property
    def nominal_codebook_hash(self) -> str:
        value = self.frozen_nominal_codebook.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _source_scoped_domain_indices(
        self,
        *,
        codebook_id: str | None,
        domain: tuple[str, ...],
        device: torch.device,
    ) -> Tensor:
        if not codebook_id or not codebook_id.strip():
            raise ValueError("source-scoped nominal tokenizer requires codebook_id")
        if len(domain) > self.nominal_codebook_size:
            raise ValueError("nominal domain exceeds frozen codebook capacity")
        if len(set(domain)) != len(domain):
            raise ValueError("nominal domain labels must be unique")
        assigned: dict[str, int] = {}
        used: set[int] = set()
        # Sorted labels make collision resolution independent of declared
        # domain order.  Reordering the wire-code domain and its payload codes
        # therefore preserves semantic category tokens.
        for label in sorted(domain):
            payload = (
                f"tabubase-nominal-codebook-v2|{self.nominal_codebook_seed}|{codebook_id}|{label}"
            ).encode()
            index = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            index %= self.nominal_codebook_size
            while index in used:
                index = (index + 1) % self.nominal_codebook_size
            assigned[label] = index
            used.add(index)
        return torch.tensor(
            [assigned[label] for label in domain],
            dtype=torch.long,
            device=device,
        )

    def _continuous_tokens(self, values: Tensor) -> Tensor:
        frequencies = self.log_frequencies.exp().to(dtype=values.dtype, device=values.device)
        lifted = values.unsqueeze(-1) * frequencies.view(1, 1, 1, -1)
        lifted = torch.cat((lifted.sin(), lifted.cos()), dim=-1)
        return self.continuous_value_encoder(lifted)

    def context_standardize_numeric(
        self,
        symbols: SymbolTable,
    ) -> NumericScaleState:
        """Return Step-1 numeric values and their context-only affine state.

        Numeric supports and numeric predictions share this standardized
        scale.  Non-numeric wire codes pass through unchanged; their reported
        mean/scale sentinels are zero/one and are ignored by typed terminals.
        """

        values = symbols.values.to(dtype=DEFAULT_FLOAT_DTYPE)
        _, _, n_features = values.shape
        if len(symbols.feature_kinds) != n_features:
            raise ValueError("cell tokenizer requires one declared kind per feature")
        context = symbols.visible_mask
        if symbols.context_mask is not None:
            if (
                symbols.context_mask.shape != symbols.values.shape
                or symbols.context_mask.dtype is not torch.bool
            ):
                raise ValueError("context_mask must be bool and match values shape")
            context = context & symbols.context_mask
        numeric_features = torch.tensor(
            tuple(kind == "numeric" for kind in symbols.feature_kinds),
            dtype=torch.bool,
            device=values.device,
        ).view(1, 1, n_features)
        statistics_mask = context & numeric_features
        count = statistics_mask.sum(dim=1, keepdim=True).to(values.dtype)
        safe_count = count.clamp_min(1.0)
        masked_abs = torch.where(statistics_mask, values.abs(), torch.zeros_like(values))
        max_abs = masked_abs.amax(dim=1, keepdim=True)
        safe_max_abs = torch.where(max_abs > 0.0, max_abs, torch.ones_like(max_abs))
        scaled_values = values / safe_max_abs
        scaled_context = torch.where(
            statistics_mask,
            scaled_values,
            torch.zeros_like(scaled_values),
        )
        scaled_mean = scaled_context.sum(dim=1, keepdim=True) / safe_count
        centered_for_statistics = torch.where(
            statistics_mask,
            scaled_values - scaled_mean,
            torch.zeros_like(scaled_values),
        )
        scaled_variance = centered_for_statistics.square().sum(dim=1, keepdim=True) / safe_count
        scaled_std = scaled_variance.sqrt()
        mean = scaled_mean * safe_max_abs
        std = scaled_std * safe_max_abs
        scale = std + self._scale_epsilon
        scaled_denominator = (scaled_std + self._scale_epsilon / safe_max_abs).clamp_min(
            torch.finfo(values.dtype).tiny
        )
        standardized_numeric = (scaled_values - scaled_mean) / scaled_denominator
        standardized = torch.where(numeric_features, standardized_numeric, values)
        public_mean = torch.where(numeric_features, mean, torch.zeros_like(mean))
        public_std = torch.where(numeric_features, std, torch.zeros_like(std))
        public_scale = torch.where(numeric_features, scale, torch.ones_like(scale))
        public_count = torch.where(numeric_features, count, torch.zeros_like(count))
        return NumericScaleState(
            standardized_values=standardized,
            mean=public_mean,
            std=public_std,
            scale=public_scale,
            context_count=public_count,
        )

    def forward(self, symbols: SymbolTable) -> TokenTable:
        batch, n_rows, n_features = symbols.values.shape
        if n_features > self.config.max_features:
            raise ValueError(
                f"feature count {n_features} exceeds max_features={self.config.max_features}"
            )
        if len(symbols.feature_kinds) != n_features:
            raise ValueError("cell tokenizer requires one declared kind per feature")
        if symbols.feature_domains and len(symbols.feature_domains) != n_features:
            raise ValueError("cell tokenizer feature domains must match the feature axis")
        if symbols.feature_codebooks and len(symbols.feature_codebooks) != n_features:
            raise ValueError("cell tokenizer feature codebooks must match the feature axis")
        numeric_scale_state = self.context_standardize_numeric(symbols)
        values = numeric_scale_state.standardized_values
        visible = symbols.visible_mask
        cells = values.new_zeros(batch, n_rows, n_features, self.config.d_model)
        for feature, kind in enumerate(symbols.feature_kinds):
            normalized = values[:, :, feature]
            kind_name = getattr(kind, "value", kind)
            domain = symbols.feature_domains[feature] if symbols.feature_domains else ()
            if kind_name in {"numeric", "ordinal"}:
                if kind_name == "ordinal":
                    if not domain:
                        raise ValueError("ordinal cell features require a declared domain")
                    if bool((visible[:, :, feature] & (normalized != normalized.round())).any()):
                        raise ValueError("ordinal values must use integral declared-domain codes")
                    domain_max = len(domain) - 1
                    if bool(
                        (
                            visible[:, :, feature] & ((normalized < 0) | (normalized > domain_max))
                        ).any()
                    ):
                        raise ValueError("ordinal values must lie inside the declared domain")
                    if domain_max <= 0:
                        raise ValueError(
                            "ordinal cell features require at least two declared levels"
                        )
                    normalized = normalized / float(domain_max)
                encoded = self._continuous_tokens(normalized.unsqueeze(-1)).squeeze(-2)
            elif kind_name == "categorical":
                if not domain:
                    raise ValueError("categorical cell features require a declared domain")
                if bool((visible[:, :, feature] & (normalized != normalized.round())).any()):
                    raise ValueError("categorical values must use integral declared-domain codes")
                max_code = len(domain) - 1
                if bool(
                    (visible[:, :, feature] & ((normalized < 0) | (normalized > max_code))).any()
                ):
                    raise ValueError("categorical values must lie inside the declared domain")
                if self.nominal_tokenizer == self.SOURCE_SCOPED_FROZEN_CODEBOOK_V2:
                    domain_indices = self._source_scoped_domain_indices(
                        codebook_id=(
                            symbols.feature_codebooks[feature]
                            if symbols.feature_codebooks
                            else None
                        ),
                        domain=domain,
                        device=values.device,
                    )
                    encoded = self.frozen_nominal_codebook[
                        domain_indices[normalized.to(torch.long).clamp(0, max_code)]
                    ]
                else:
                    # v1 deliberately creates a local episode code in
                    # first-appearance order.  It remains frozen as the
                    # historical control even though v2 fixes its row-order
                    # sensitivity and cross-episode token drift.
                    local_codes = torch.zeros_like(normalized, dtype=torch.long)
                    visible_feature = visible[:, :, feature]
                    observed_codes = (
                        normalized.detach()[visible_feature]
                        .to(device="cpu", dtype=torch.long)
                        .tolist()
                    )
                    ordered_codes = list(dict.fromkeys(observed_codes))
                    for local_count, code in enumerate(ordered_codes):
                        code_mask = visible_feature & (normalized == float(code))
                        local_codes = torch.where(
                            code_mask,
                            torch.full_like(local_codes, local_count),
                            local_codes,
                        )
                    local_count = len(ordered_codes)
                    if local_count:
                        sphere = self._random_sphere(
                            episode_id=symbols.episode_id,
                            feature_key=f"cardinality:{len(domain)}",
                            count=local_count,
                            device=values.device,
                        )
                        encoded = sphere[local_codes.clamp_max(local_count - 1)]
                    else:
                        encoded = values.new_zeros(*normalized.shape, self.config.d_model)
            else:
                raise ValueError(f"unknown cell feature kind: {kind_name!r}")
            cells[:, :, feature] = encoded

        artificial = symbols.artificial_target_mask
        query = symbols.query_target_mask
        if artificial is None:
            artificial = symbols.target_mask
        if query is None:
            query = torch.zeros_like(symbols.target_mask)
        cells = torch.where(
            artificial.unsqueeze(-1),
            self.mask_token.view(1, 1, 1, -1).expand_as(cells),
            cells,
        )
        cells = torch.where(
            query.unsqueeze(-1),
            self.query_token.view(1, 1, 1, -1).expand_as(cells),
            cells,
        )
        cells = torch.where(symbols.natural_missing_mask.unsqueeze(-1), 0.0, cells)
        return TokenTable(
            cells=cells,
            visible_mask=symbols.visible_mask,
            target_mask=symbols.target_mask,
            natural_missing_mask=symbols.natural_missing_mask,
            numeric_scale_state=numeric_scale_state,
        )


__all__ = [
    "CellTokenizer",
    "NumericScaleState",
    "SymbolTable",
    "Symbolizer",
    "TokenTable",
]
