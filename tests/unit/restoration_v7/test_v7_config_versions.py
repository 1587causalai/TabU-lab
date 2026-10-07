"""Version defaults must not silently redefine historical V7 constructor calls."""

import json

import pytest

from tabu_lab.models.restoration_v7 import V7Config


@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_direct_constructor_factory_and_partial_deserialization_agree(version):
    direct = V7Config(model_version=version)
    assert direct == V7Config.for_version(version)
    assert direct == V7Config.from_dict({"model_version": version})
    assert V7Config.from_dict(json.loads(json.dumps(direct.as_dict()))) == direct
    if version == "v7":
        assert direct == V7Config() == V7Config.legacy() == V7Config.from_dict({})
        assert direct.coupling_bias and direct.numeric_preprocessing == "legacy"
        assert direct.unit_source_policy == "legacy_cell_sources"
        assert direct.query_init == "seed" and not direct.query_source and direct.rounds == 1
    else:
        assert direct == V7Config.v73()
        assert not direct.coupling_bias
        assert direct.numeric_preprocessing == "standard_asinh_v1"
        assert direct.unit_source_policy == "observed"
        assert direct.query_init == "donor" and direct.query_source and direct.rounds == 4


def test_explicit_values_are_not_mistaken_for_missing_version_defaults():
    overrides = dict(
        query_init="seed",
        query_source=False,
        rounds=1,
        coupling_bias=True,
        numeric_preprocessing="legacy",
        unit_source_policy="legacy_cell_sources",
    )
    direct = V7Config(model_version="v7.3", **overrides)
    assert direct == V7Config.v73(**overrides)
    assert direct == V7Config.from_dict({"model_version": "v7.3", **overrides})
    for name, value in overrides.items():
        assert getattr(direct, name) == value
    # Pre-version checkpoints can already contain explicit experimental choices.
    recorded = V7Config.v73(rounds=8, unit_layers=1).as_dict()
    recorded.pop("model_version")
    restored = V7Config.from_dict(recorded)
    assert restored.as_dict() == recorded | {"model_version": "v7"}


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize(
    "field",
    [
        "query_init",
        "query_source",
        "rounds",
        "coupling_bias",
        "numeric_preprocessing",
        "unit_source_policy",
    ],
)
def test_explicit_none_is_invalid_rather_than_a_request_for_defaults(version, field):
    with pytest.raises(ValueError):
        V7Config(model_version=version, **{field: None})


def test_original_positional_constructor_still_binds_the_original_fields():
    fields = (
        "code_dim",
        "codec",
        "unit_layers",
        "value_map",
        "query_init",
        "query_source",
        "gradient_checkpointing",
        "backbone",
        "rounds",
        "share_rounds",
        "coupling_blocks",
        "coupling_hidden",
        "coupling_alpha",
        "coupling_scale",
        "bandwidth",
        "ridge",
        "epsilon",
        "center_chunk_size",
        "round_loss_rho",
        "chi_numeric",
        "chi_discrete",
    )
    arguments = (
        64,
        "C64/8",
        2,
        "identity",
        "donor",
        True,
        True,
        dict(width=64, layers=1, heads=2, ff_width=64, slots=3),
        3,
        False,
        2,
        (12,),
        0.2,
        False,
        0.7,
        0.002,
        1e-7,
        7,
        0.4,
        23.0,
        0.8,
    )
    assert V7Config(*arguments) == V7Config.legacy(**dict(zip(fields, arguments, strict=True)))


@pytest.mark.parametrize("mode", ["legacy", "standard_softlog_v1", "standard_asinh_v1"])
@pytest.mark.parametrize("version", ["v7", "v7.3"])
def test_explicit_numeric_protocol_survives_factory_and_checkpoint_config(version, mode):
    config = V7Config.for_version(version, numeric_preprocessing=mode)
    recorded = json.loads(json.dumps(config.as_dict()))
    assert V7Config.from_dict(recorded) == config
    assert V7Config(model_version=version, numeric_preprocessing=mode) == config
    assert config.numeric_preprocessing == mode
