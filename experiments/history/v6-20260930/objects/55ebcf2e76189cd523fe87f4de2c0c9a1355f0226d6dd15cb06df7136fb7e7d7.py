"""Audited V5.4-to-V5.5 model-only transfer for old120 Small/H4 or H8/Unit3."""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from tabu_lab.models.restoration._dtype import execution_dtype
from tabu_lab.models.restoration_v53 import V53LossConfig
from tabu_lab.models.restoration_v54 import V54Config, V54Model
from tabu_lab.restoration_optimizers import adamw

from .artifacts import (
    WARM_START_SCHEMA, atomic_json, finite_state, load_checkpoint, save_warm_start,
    sha256,
)
from .data import build_episode
from .evaluation import evaluate_probe
from .factory import make_model
from .loss_replay import V2
from .protocol import V54_SCHEMA, V55_SCHEMA, _digest
from .runner import _seed_model, _validate_resume, configure_runtime, train_step


def _validated_parent(plan, checkpoint, resolved_path):
    payload, checkpoint_digest = load_checkpoint(checkpoint)
    resolved_path = Path(resolved_path)
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    donor_spec, donor_identity = resolved["spec"], resolved["identity"]
    if payload.get("purpose") != "training" or donor_spec.get("schema") != V54_SCHEMA:
        raise ValueError("donor must be a V5.4 training checkpoint")
    if donor_identity != payload.get("identity") or donor_identity.get("schema") != V54_SCHEMA:
        raise ValueError("donor resolved identity differs from checkpoint")
    if _digest({key: value for key, value in donor_identity.items() if key != "sha256"}) != (
        donor_identity.get("sha256")
    ):
        raise ValueError("donor resolved identity digest mismatch")
    portable = dict(donor_spec)
    portable["tables"] = [
        {key: value for key, value in entry.items() if key != "path"}
        for entry in donor_spec["tables"]
    ]
    if _digest(portable) != donor_identity.get("manifest_sha256"):
        raise ValueError("donor resolved manifest digest mismatch")
    if payload.get("model_config") != donor_spec.get("model"):
        raise ValueError("donor model config differs from resolved manifest")
    donor_config = V54Config.from_dict(donor_spec["model"])
    if donor_config.as_dict() != payload["model_config"]:
        raise ValueError("donor model config is not fully resolved")
    # The donor's own scheduler/optimizer invariants are checked before any
    # conversion. The supplied runtime is its recorded one, not this host's.
    donor_plan = replace(plan, spec=donor_spec, config=donor_config,
                         identity=donor_identity)
    _validate_resume(payload, donor_plan, payload["runtime"])
    if plan.spec["schema"] != V55_SCHEMA or plan.config.codec_version != (
        "constant_weight_composition_v2"
    ):
        raise ValueError("target must be a V5.5 ordinal composition v2 plan")
    if donor_config.codec_version != "constant_weight_composition_v1":
        raise ValueError("donor must use V5.4 ordinal composition v1")
    source_config = donor_config.as_dict()
    target_config = plan.config.as_dict()
    if {**source_config, "codec_version": target_config["codec_version"]} != target_config:
        raise ValueError("V5.4 to V5.5 transfer permits only the ordinal codec change")
    if (target_config["size"] != "small" or target_config["unit_layers"] != 3
            or target_config["backbone"]["heads"] not in (4, 8)):
        raise ValueError("this warm-start entry is restricted to Small/H4 or H8/Unit3")
    if donor_identity.get("datasets") != plan.identity.get("datasets") or (
        donor_identity.get("data_sha256") != plan.identity.get("data_sha256")
    ):
        raise ValueError("donor and target old120 data identity differs")
    for key in ("seeds", "optimizer", "probes"):
        if donor_spec[key] != plan.spec[key]:
            raise ValueError(f"donor and target {key} differ")
    if len(donor_spec["stages"]) != 1 or len(plan.spec["stages"]) != 1:
        raise ValueError("old120 warm start requires one bounded stage")
    donor_stage, target_stage = donor_spec["stages"][0], plan.spec["stages"][0]
    for key in ("name", "sampling", "recipe", "loss", "optimizer", "probes"):
        if donor_stage[key] != target_stage[key]:
            raise ValueError(f"donor and target stage {key} differ")
    target_replay = target_stage.get("loss_replay", {})
    if target_replay != {
        "kind": V2,
        "normal_max_updates": 983_040,
        "start_normal_cursor": 0,
        "start_extra_updates": 0,
    } or target_stage["max_updates"] != 1_327_104:
        raise ValueError("target requires a fresh full old120 replay-v2 budget")
    return payload, checkpoint_digest, donor_config, sha256(resolved_path)


def _convert_state(plan, payload, donor_config, device):
    source = payload["model"]
    parent_runtime = payload["runtime"]
    source_dtype = {
        "mps": torch.float32, "cpu": torch.float64, "cuda:0": torch.float64,
    }.get(parent_runtime.get("device"))
    if source_dtype is None or parent_runtime.get("dtype") != str(source_dtype).removeprefix("torch."):
        raise ValueError("donor checkpoint runtime dtype is not qualified")
    donor_model = V54Model(donor_config).to(dtype=torch.float64)
    donor_reference = donor_model.state_dict()
    target_model = make_model(plan).to(device=device, dtype=execution_dtype(device))
    target_reference = target_model.state_dict()
    if set(source) != set(donor_reference) or set(source) != set(target_reference):
        raise ValueError("donor and target model parameter names differ")
    if len(source) != 116 or sum(p.numel() for p in target_model.parameters()) != 2_082_688:
        raise ValueError("Small/H4 or H8/Unit3 model parameter count differs")
    if source["_codec_signature"].tolist() != [4, 1] or (
        target_reference["_codec_signature"].tolist() != [6, 1]
    ):
        raise ValueError("unexpected ordinal codec signature")
    converted, mapping = {}, []
    for name, expected in target_reference.items():
        value = source[name]
        if value.shape != donor_reference[name].shape or value.shape != expected.shape:
            raise ValueError(f"warm-start tensor shape differs: {name}")
        if name == "_codec_signature":
            replacement = expected.detach().cpu().clone()
            action = "replace_codec_signature"
        else:
            if not value.is_floating_point() or not expected.is_floating_point():
                raise ValueError(f"unexpected nonfloating model tensor: {name}")
            if value.dtype != source_dtype or expected.dtype != execution_dtype(device):
                raise ValueError(f"unexpected source or target tensor dtype: {name}")
            replacement = value.to(dtype=expected.dtype, device="cpu").clone()
            action = "copy_cast"
        converted[name] = replacement
        mapping.append({
            "source": name, "target": name, "source_shape": list(value.shape),
            "target_shape": list(expected.shape), "source_dtype": str(value.dtype),
            "target_dtype": str(replacement.dtype), "action": action,
        })
    target_model.load_state_dict(converted, strict=True)
    if not finite_state(converted):
        raise FloatingPointError("nonfinite converted model")
    return converted, mapping, target_model


def validate_warm_start(payload, plan, model):
    """Reject stale or malformed transfers before the first checkpoint is saved."""
    if payload.get("schema") != WARM_START_SCHEMA or payload.get("purpose") != (
        "weights_only_initialization"
    ):
        raise ValueError("not a V5.5 model-only warm start")
    if plan.spec["schema"] != V55_SCHEMA or payload.get("target_identity") != plan.identity:
        raise ValueError("warm-start target identity drift")
    if payload.get("target_model_config") != plan.config.as_dict():
        raise ValueError("warm-start target model config drift")
    expected = model.state_dict()
    state = payload["model"]
    mapping = payload.get("conversion", {}).get("tensors")
    if set(state) != set(expected) or not isinstance(mapping, list) or len(mapping) != len(state):
        raise ValueError("warm-start parameter mapping is incomplete")
    if {item.get("target") for item in mapping} != set(state) or (
        len({item.get("target") for item in mapping}) != len(mapping)
    ):
        raise ValueError("warm-start parameter mapping has duplicate or unknown names")
    for item in mapping:
        name = item["target"]
        tensor = state[name]
        if (item.get("source") != name or item.get("source_shape") != list(tensor.shape)
                or item.get("target_shape") != list(expected[name].shape)
                or item.get("target_dtype") != str(tensor.dtype)
                or tensor.shape != expected[name].shape
                or tensor.dtype != expected[name].dtype
                or item.get("action") != (
                    "replace_codec_signature" if name == "_codec_signature" else "copy_cast"
                )):
            raise ValueError(f"warm-start parameter mapping differs: {name}")
    if state["_codec_signature"].tolist() != [6, 1] or not finite_state(state):
        raise ValueError("warm-start codec signature or parameter values differ")
    conversion = payload["conversion"]
    if conversion.get("codec_signature") != {"source": [4, 1], "target": [6, 1]} or (
        conversion.get("tensor_count") != len(state)
        or conversion.get("trainable_parameters") != sum(p.numel() for p in model.parameters())
    ):
        raise ValueError("warm-start conversion contract differs")
    target_runtime = payload.get("target_runtime", {})
    parameter = next(model.parameters())
    if (target_runtime.get("device") != str(parameter.device)
            or target_runtime.get("dtype") != str(parameter.dtype).removeprefix("torch.")):
        raise ValueError("warm-start device or dtype differs from prepared runtime")
    parent = payload.get("parent", {})
    if (parent.get("identity", {}).get("schema") != V54_SCHEMA
            or not isinstance(parent.get("checkpoint_sha256"), str)
            or len(parent["checkpoint_sha256"]) != 64):
        raise ValueError("warm-start parent identity is missing")


def _bank_identity(plan, donor_config, probe, device):
    """Verify that changing the codec has not changed the frozen Query addresses."""
    import hashlib

    seeds = dict(plan.spec["seeds"])
    seeds["evaluation"] = int.from_bytes(
        hashlib.sha256(f"{seeds['evaluation']}/{probe['name']}".encode()).digest()[:8],
        "little",
    )
    bank = []
    for table in plan.tables:
        if table.cohort not in probe["cohorts"]:
            continue
        for index in range(probe["masks"]):
            paired = []
            for config in (donor_config, plan.config):
                _, _, _, info = build_episode(
                    table, probe["recipe"], index, seeds, device,
                    evaluation=True, partition=probe["partition"],
                    epsilon=config.epsilon, codec_version=config.codec_version,
                )
                paired.append((info["row_ids"], info["query_addresses"]))
            if paired[0] != paired[1]:
                raise ValueError(f"fixed Query bank differs for {table.name} mask {index}")
            bank.append([table.name, index, *paired[0]])
    return {"sha256": _digest(bank), "episodes": len(bank)}


def _comparison(plan, donor_config, source_model, target_model, device):
    probes = [probe for probe in plan.spec["probes"] if probe["name"] in (
        plan.spec["stages"][0]["probes"]
    ) and probe["partition"] == "train"]
    if not probes:
        raise ValueError("target stage has no fixed train Query probe")
    donor_plan = replace(plan, config=donor_config)
    reports = {}
    for probe in probes:
        bank = _bank_identity(plan, donor_config, probe, device)
        old = evaluate_probe(source_model, donor_plan, probe, device)
        new = evaluate_probe(target_model, plan, probe, device)
        for a, b in zip(old["by_table"], new["by_table"], strict=True):
            for key in ("table", "query_exposures", "unique_query_cells", "query_rows"):
                if a[key] != b[key]:
                    raise ValueError(f"fixed Query coverage differs: {a['table']} {key}")
        reports[probe["name"]] = {"bank": bank, "v54": old, "v55": new}
    return reports


def _representative_tables(plan):
    numeric = [table for table in plan.tables if table.role == "train" and all(
        column.kind == "numeric" for column in table.schema
    )]
    mixed = [table for table in plan.tables if table.role == "train" and {
        "nominal", "ordinal"
    } <= {column.kind for column in table.schema}]
    if not numeric or not mixed:
        raise ValueError("warm-start preflight requires numeric and nominal/ordinal tables")
    def footprint(table):
        return min(table.train_rows, table.window_rows or table.train_rows) * table.width
    return [("numeric", max(numeric, key=footprint)),
            ("mixed_nominal_ordinal", max(mixed, key=footprint))]


def prepare_v54_warm_start(plan, donor_checkpoint, donor_resolved, output, *, device="cpu"):
    """Create a V5.5 weight transfer only after same-device numerical admission."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    receipt = {
        "schema": WARM_START_SCHEMA, "purpose": "weights_only_initialization",
        "outcome": "started", "target_identity": plan.identity,
    }
    atomic_json(output / "started.json", receipt)
    try:
        payload, donor_digest, donor_config, resolved_digest = _validated_parent(
            plan, donor_checkpoint, donor_resolved,
        )
        runtime = configure_runtime(device)
        receipt["runtime"] = runtime
        converted, mapping, target_model = _convert_state(plan, payload, donor_config, device)
        source_model = V54Model(donor_config).to(device=device, dtype=execution_dtype(device))
        source_model.load_state_dict(payload["model"], strict=True)
        comparison = _comparison(plan, donor_config, source_model, target_model, device)
        atomic_json(output / "step0-comparison.json", comparison)
        receipt["step0_comparison"] = {
            name: {
                "bank": item["bank"],
                "v54_macro": item["v54"]["macro"], "v55_macro": item["v55"]["macro"],
                "table_count": item["v55"]["tables"],
            } for name, item in comparison.items()
        }
        # This is a separate model and optimizer. The artifact tensors and the
        # evaluated step-zero model remain unchanged by the admission step.
        stage = plan.spec["stages"][0]
        one_step = []
        for selection, table in _representative_tables(plan):
            _seed_model(plan)
            clone = make_model(plan).to(device=device, dtype=execution_dtype(device))
            clone.load_state_dict(converted, strict=True)
            optimizer = adamw(clone, plan.optimizer)
            step = train_step(
                clone, optimizer, plan, table, stage["recipe"][table.kind], 0,
                device, V53LossConfig(**stage.get("loss", {})), namespace=stage["name"],
                objective=stage.get("objective"),
            )
            one_step.append({
                "selection": selection, "table": table.name, "episode_index": 0,
                "loss": step["loss"], "objective_loss": step["objective_loss"],
                "gradient_norm": step["gradient_norm"],
                "clipped_gradient_norm": step["clipped_gradient_norm"],
                "readout_scope": step["readout_scope"], "seconds": step["seconds"],
            })
            del clone, optimizer
        if not finite_state(converted) or any(
            not torch.equal(converted[name], value.detach().cpu())
            for name, value in target_model.state_dict().items()
        ):
            raise ValueError("warm-start initialization tensors changed during preflight")
        receipt["clone_one_step"] = one_step
        artifact_payload = {
            "schema": WARM_START_SCHEMA, "purpose": "weights_only_initialization",
            "target_identity": plan.identity, "target_model_config": plan.config.as_dict(),
            "target_runtime": {"device": runtime["device"], "dtype": runtime["dtype"]},
            "parent": {
                "identity": payload["identity"], "checkpoint_sha256": donor_digest,
                "checkpoint_update": payload["state"]["update"],
                "checkpoint_runtime": payload["runtime"],
                "resolved_sha256": resolved_digest,
            },
            "conversion": {
                "codec_signature": {"source": [4, 1], "target": [6, 1]},
                "tensor_count": len(mapping), "trainable_parameters": 2_082_688,
                "tensors": mapping,
            },
            "parent_lineage": payload.get("lineage", []),
            "preflight": {
                "step0_comparison_sha256": sha256(output / "step0-comparison.json"),
                "clone_one_step": receipt["clone_one_step"],
            },
            "model": converted,
        }
        validate_warm_start(artifact_payload, plan, target_model)
        artifact, digest = save_warm_start(output, artifact_payload)
        receipt.update(
            outcome="passed", artifact=str(artifact.resolve()), artifact_sha256=digest,
            parent_checkpoint_sha256=donor_digest,
            parent_update=payload["state"]["update"],
            parent_resolved_sha256=resolved_digest,
            conversion={"tensor_count": len(mapping), "trainable_parameters": 2_082_688,
                        "codec_signature": {"source": [4, 1], "target": [6, 1]}},
        )
    except (KeyboardInterrupt, Exception) as error:
        receipt.update(
            outcome="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__, error=str(error),
        )
    receipt["seconds"] = time.monotonic() - started
    receipt["claim_boundary"] = "same-device step-zero Query and cloned one-step admission only"
    atomic_json(output / "terminal.json", receipt)
    return receipt
