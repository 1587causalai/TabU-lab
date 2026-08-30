"""Explicit model-factory contract builders."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields
from typing import Any, TypeVar

from torch import nn

from tabu_lab.numerics import DEFAULT_FLOAT_DTYPE

from .reference import (
    TabU4GraphModel,
    TabU4RecModel,
    TabUFLModel,
    TabUFModel,
    TabULModel,
    TabUUnitPairModel,
    TabUUnitRowModel,
)
from .table_cell import (
    TabUCellBaseModel,
    TabUCellColumnModel,
    TabUCellRecModel,
    TabUCellRowColumnModel,
    TabUCellRowModel,
)
from .types import DesignOpenBuild, ReferenceConfig

_ModelT = TypeVar("_ModelT", bound=nn.Module)


def _build_float32(
    factory: Callable[..., _ModelT],
    /,
    *args: Any,
    **kwargs: Any,
) -> _ModelT:
    """Construct one executable model under the repository dtype policy."""

    return factory(*args, **kwargs).to(dtype=DEFAULT_FLOAT_DTYPE)


def _config_from_kwargs(kwargs: dict[str, Any]) -> ReferenceConfig:
    explicit = kwargs.pop("config", None)
    if explicit is not None:
        if not isinstance(explicit, ReferenceConfig):
            raise TypeError("config must be a ReferenceConfig")
        return explicit
    names = {field.name for field in fields(ReferenceConfig)}
    values = {name: kwargs.pop(name) for name in tuple(kwargs) if name in names}
    return ReferenceConfig(**values)


def build_tabufl(**kwargs: Any) -> TabUFLModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    label_columns = kwargs.pop("label_columns", (-1,))
    readout_geometry = kwargs.pop("readout_geometry", "matched_uf")
    label_address_plan = kwargs.pop("label_address_plan", "matched_uf")
    numeric_terminal = kwargs.pop("numeric_terminal", "nadaraya_watson")
    if kwargs:
        raise TypeError(f"unknown TabUFL builder options: {sorted(kwargs)}")
    return _build_float32(
        TabUFLModel,
        config,
        label_columns=label_columns,
        readout_geometry=readout_geometry,
        label_address_plan=label_address_plan,
        numeric_terminal=numeric_terminal,
    )


def build_tabul(**kwargs: Any) -> TabULModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    label_columns = kwargs.pop("label_columns", (-1,))
    readout_geometry = kwargs.pop("readout_geometry", "matched_uf")
    label_address_plan = kwargs.pop("label_address_plan", "matched_uf")
    numeric_terminal = kwargs.pop("numeric_terminal", "nadaraya_watson")
    if kwargs:
        raise TypeError(f"unknown TabUL builder options: {sorted(kwargs)}")
    return _build_float32(
        TabULModel,
        config,
        label_columns=label_columns,
        readout_geometry=readout_geometry,
        label_address_plan=label_address_plan,
        numeric_terminal=numeric_terminal,
    )


def build_tabuf(**kwargs: Any) -> TabUFModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    readout_geometry = kwargs.pop("readout_geometry", "matched_uf")
    numeric_terminal = kwargs.pop("numeric_terminal", "nadaraya_watson")
    if kwargs:
        raise TypeError(f"unknown TabUF builder options: {sorted(kwargs)}")
    return _build_float32(
        TabUFModel,
        config,
        readout_geometry=readout_geometry,
        numeric_terminal=numeric_terminal,
    )


def build_tabu4rec(**kwargs: Any) -> TabU4RecModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    readout_geometry = kwargs.pop("readout_geometry", "matched_uf")
    recommendation_address_plan = kwargs.pop("recommendation_address_plan", "matched_uf")
    numeric_terminal = kwargs.pop("numeric_terminal", "nadaraya_watson")
    rec_axis_summary_dim = int(kwargs.pop("rec_axis_summary_dim", 2))
    rec_matched_residual_scale = float(kwargs.pop("rec_matched_residual_scale", 0.1))
    if kwargs:
        raise TypeError(f"unknown TabU4Rec builder options: {sorted(kwargs)}")
    return _build_float32(
        TabU4RecModel,
        config,
        readout_geometry=readout_geometry,
        recommendation_address_plan=recommendation_address_plan,
        rec_axis_summary_dim=rec_axis_summary_dim,
        rec_matched_residual_scale=rec_matched_residual_scale,
        numeric_terminal=numeric_terminal,
    )


def build_tabu4graph(**kwargs: Any) -> TabU4GraphModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    target_feature = int(kwargs.pop("target_feature", 0))
    unit_receiver_plan = kwargs.pop("unit_receiver_plan", "same_row_visible_cells")
    numeric_terminal = kwargs.pop("numeric_terminal", "nadaraya_watson")
    if kwargs:
        raise TypeError(f"unknown TabU4Graph builder options: {sorted(kwargs)}")
    return _build_float32(
        TabU4GraphModel,
        config,
        target_feature=target_feature,
        unit_receiver_plan=unit_receiver_plan,
        numeric_terminal=numeric_terminal,
    )


def build_tabu4do(**kwargs: Any) -> DesignOpenBuild:
    if kwargs:
        raise TypeError(
            "TabU4Do is design-open and accepts no implementation options: " f"{sorted(kwargs)}"
        )
    return DesignOpenBuild(model_id="tabu4do")


def build_tabu_unit_row(**kwargs: Any) -> TabUUnitRowModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    numeric_terminal = kwargs.pop("numeric_terminal", "local_linear")
    if kwargs:
        raise TypeError(f"unknown Unit-row builder options: {sorted(kwargs)}")
    return _build_float32(TabUUnitRowModel, config, numeric_terminal=numeric_terminal)


def build_tabu_unit_pair(**kwargs: Any) -> TabUUnitPairModel:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    # Unit-as-cell.tex freezes Local Linear as the default numeric terminal;
    # NW remains an explicit nested baseline/variant.
    numeric_terminal = kwargs.pop("numeric_terminal", "local_linear")
    if kwargs:
        raise TypeError(f"unknown Unit-as-cell builder options: {sorted(kwargs)}")
    return _build_float32(
        TabUUnitPairModel,
        config,
        numeric_terminal=numeric_terminal,
    )


def build_tabu_cell_base(**kwargs: Any) -> TabUCellBaseModel:
    """Build the independent axis-B TabUBase identity.

    The implementation shares the dense cell primitives with the legacy
    Unit-as-cell model, while keeping a separate contract/trace ID.  The
    v0.2.0 profile is explicit; ``label_broadcast`` is retained only as a
    backwards-compatible adapter for older local probes.
    """

    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    numeric_terminal = kwargs.pop("numeric_terminal", "local_linear")
    profile = kwargs.pop("profile", None)
    label_broadcast = kwargs.pop("label_broadcast", None)
    label_broadcast_tau = float(kwargs.pop("label_broadcast_tau", 1.0e-6))
    nominal_tokenizer = kwargs.pop("nominal_tokenizer", "episode_random_sphere")
    nominal_codebook_size = int(kwargs.pop("nominal_codebook_size", 100))
    nominal_codebook_seed = int(kwargs.pop("nominal_codebook_seed", 1729))
    variant_ref = kwargs.pop("variant_ref", None)
    label_columns = kwargs.pop("label_columns", None)
    if label_columns is not None and len(tuple(label_columns)) != 1:
        raise ValueError("tabu.cell.base supervised profile requires exactly one label column")
    if kwargs:
        raise TypeError(f"unknown table-cell base builder options: {sorted(kwargs)}")
    return _build_float32(
        TabUCellBaseModel,
        config,
        numeric_terminal=numeric_terminal,
        profile=profile,
        label_broadcast=label_broadcast,
        label_broadcast_tau=label_broadcast_tau,
        nominal_tokenizer=nominal_tokenizer,
        nominal_codebook_size=nominal_codebook_size,
        nominal_codebook_seed=nominal_codebook_seed,
        variant_ref=variant_ref,
    )


def _build_tabu_cell_special(
    model_type: type[_ModelT],
    model_name: str,
    **kwargs: Any,
) -> _ModelT:
    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    numeric_terminal = kwargs.pop("numeric_terminal", "local_linear")
    label_broadcast = bool(kwargs.pop("label_broadcast", False))
    label_broadcast_tau = float(kwargs.pop("label_broadcast_tau", 1.0e-6))
    if kwargs:
        raise TypeError(f"unknown {model_name} builder options: {sorted(kwargs)}")
    return _build_float32(
        model_type,
        config,
        numeric_terminal=numeric_terminal,
        label_broadcast=label_broadcast,
        label_broadcast_tau=label_broadcast_tau,
    )


def build_tabu_cell_row(**kwargs: Any) -> TabUCellRowModel:
    """Build the experimental row-special cell contract."""

    return _build_tabu_cell_special(TabUCellRowModel, "tabu.cell.row", **kwargs)


def build_tabu_cell_column(**kwargs: Any) -> TabUCellColumnModel:
    """Build the experimental column-special cell contract."""

    return _build_tabu_cell_special(TabUCellColumnModel, "tabu.cell.column", **kwargs)


def build_tabu_cell_row_column(**kwargs: Any) -> TabUCellRowColumnModel:
    """Build the experimental concatenated row/column cell contract."""

    return _build_tabu_cell_special(
        TabUCellRowColumnModel,
        "tabu.cell.row_column",
        **kwargs,
    )


def build_tabu_cell_rec(**kwargs: Any) -> TabUCellRecModel:
    """Build a direct axis-B Rec profile for component conformance work.

    The contract registry intentionally remains ``design_open`` until the
    profile lineage is reviewed; callers must choose ``m``, ``w``, or ``rc``.
    """

    kwargs = dict(kwargs)
    config = _config_from_kwargs(kwargs)
    profile = kwargs.pop("profile", None)
    if profile is None:
        raise TypeError("tabu.cell.rec requires an explicit profile: m, w, or rc")
    numeric_terminal = kwargs.pop("numeric_terminal", None)
    if kwargs:
        raise TypeError(f"unknown tabu.cell.rec builder options: {sorted(kwargs)}")
    return _build_float32(
        TabUCellRecModel,
        config,
        profile=profile,
        numeric_terminal=numeric_terminal,
    )


_BUILDERS = {
    "tabufl": build_tabufl,
    "tabul": build_tabul,
    "tabuf": build_tabuf,
    "tabu4rec": build_tabu4rec,
    "tabu4graph": build_tabu4graph,
    "tabu4do": build_tabu4do,
    "tabu.unit_row": build_tabu_unit_row,
    "tabu.unit_pair": build_tabu_unit_pair,
    "tabu.cell.base": build_tabu_cell_base,
    "tabu.cell.row": build_tabu_cell_row,
    "tabu.cell.column": build_tabu_cell_column,
    "tabu.cell.row_column": build_tabu_cell_row_column,
    "tabu.cell.rec": build_tabu_cell_rec,
}


class BuilderRegistry:
    """Explicit extension point for model builders.

    Registration is intentionally local and duplicate-rejecting.  A new model
    does not require editing a central conditional branch, while the default
    built-ins remain exactly the same.
    """

    def __init__(self, builders: Mapping[str, Callable[..., Any]] | None = None) -> None:
        self._builders: dict[str, Callable[..., Any]] = dict(builders or {})

    def register(
        self,
        model_id: str,
        builder: Callable[..., Any],
        *,
        replace: bool = False,
    ) -> None:
        if not model_id or not model_id.strip():
            raise ValueError("model_id cannot be blank")
        if not callable(builder):
            raise TypeError("builder must be callable")
        if model_id in self._builders and not replace:
            raise ValueError(f"model builder already registered: {model_id}")
        self._builders[model_id] = builder

    def get(self, model_id: str) -> Callable[..., Any]:
        try:
            return self._builders[model_id]
        except KeyError as exc:
            raise KeyError(f"unknown TabU model id: {model_id}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def build(self, model_id: str, **kwargs: Any) -> Any:
        return self.get(model_id)(**kwargs)


MODEL_BUILDERS = BuilderRegistry(_BUILDERS)


def register_model_builder(
    model_id: str,
    builder: Callable[..., Any],
    *,
    replace: bool = False,
) -> None:
    """Register a builder through the public explicit extension seam."""

    MODEL_BUILDERS.register(model_id, builder, replace=replace)


def build_model(model_id: str, **kwargs: Any) -> Any:
    return MODEL_BUILDERS.build(model_id, **kwargs)


def build_from_spec(spec: Any, **kwargs: Any) -> Any:
    """Registry-facing delayed builder.

    The registry deliberately does not bind implementation types.  This seam
    accepts either a typed ``ModelSpec`` or a mapping with ``model_id``.
    """

    if isinstance(spec, Mapping):
        model_id = spec.get("model_id", spec.get("id"))
    else:
        model_id = getattr(spec, "model_id", getattr(spec, "id", None))
    if not isinstance(model_id, str) or not model_id:
        raise TypeError("spec must expose a non-empty model_id")
    return build_model(model_id, **kwargs)


__all__ = [
    "MODEL_BUILDERS",
    "BuilderRegistry",
    "build_from_spec",
    "build_model",
    "build_tabu4do",
    "build_tabu4graph",
    "build_tabu4rec",
    "build_tabu_cell_base",
    "build_tabu_cell_column",
    "build_tabu_cell_row",
    "build_tabu_cell_row_column",
    "build_tabu_cell_rec",
    "build_tabu_unit_pair",
    "build_tabu_unit_row",
    "build_tabuf",
    "build_tabufl",
    "build_tabul",
    "register_model_builder",
]
