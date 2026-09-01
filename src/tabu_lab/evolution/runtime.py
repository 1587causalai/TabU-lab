"""Minimal vertical runtime for immutable TabU program snapshots."""

from __future__ import annotations

import importlib
import inspect
import json
import platform
import random
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from pydantic import ValidationError

from tabu_lab.contracts import EvidenceEpisode, TruthSidecar, canonical_hash, canonical_json
from tabu_lab.evidence import RunIdentity
from tabu_lab.models import ReferenceConfig, build_model
from tabu_lab.training import Objective, Trainer

from .checkpoint import (
    file_sha256,
    load_program_checkpoint,
    program_sidecar_path,
    read_checkpoint_model_state,
    read_program_checkpoint,
    save_program_checkpoint,
)
from .models import (
    CompatibilityDisposition,
    ComponentGraphNode,
    EvidenceStatus,
    FrozenProgram,
    GeneratorNode,
    ObjectiveBundleNode,
    ProgramArtifact,
    ProgramCheckpointKind,
    ProgramInitialization,
    ProgramInitializationMode,
    ProgramLane,
    ProgramRunReceipt,
    ProgramRunStatus,
    ResolvedProgramSnapshot,
    StateProjectionNode,
    TrainingRecipeNode,
    WorldMixtureNode,
)
from .policy import SamplingPolicyEngine
from .repository import EvolutionManifestError, EvolutionRepository


def identity_state_projection(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Strict weights-only projection for unchanged model/state topology."""

    if set(source_state) != set(target_state):
        raise ValueError("identity projection requires identical model-state keys")
    projected: dict[str, torch.Tensor] = {}
    for name, target in target_state.items():
        source = source_state[name]
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"identity projection tensor mismatch at {name}")
        projected[name] = source.detach().clone()
    return projected


def _import_symbol(symbol_ref: str) -> Any:
    module_name, separator, qualname = symbol_ref.partition(":")
    if not separator:
        raise ValueError("runtime symbol refs must use module:qualname")
    value: Any = importlib.import_module(module_name)
    for name in qualname.split("."):
        value = getattr(value, name)
    return value


def freeze_program(resolved: ResolvedProgramSnapshot) -> FrozenProgram:
    if resolved.lane is not ProgramLane.GROW:
        raise ValueError("only a grow snapshot can cross the freeze boundary")
    evidence_payload = resolved.model_dump(mode="python", exclude={"snapshot_hash"})
    evidence_payload["lane"] = ProgramLane.EVIDENCE
    evidence_payload["evidence_status"] = EvidenceStatus.FROZEN_NOT_RUN
    evidence_resolved = ResolvedProgramSnapshot(
        **evidence_payload,
        snapshot_hash=canonical_hash(evidence_payload),
    )
    payload = {
        "schema_version": "tabu.frozen-program.v1",
        "lane": ProgramLane.EVIDENCE,
        "evidence_status": EvidenceStatus.FROZEN_NOT_RUN,
        "resolved": evidence_resolved,
    }
    return FrozenProgram(**payload, freeze_hash=canonical_hash(payload))


def load_frozen_program(path: str | Path) -> FrozenProgram:
    try:
        return FrozenProgram.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"invalid frozen program: {exc}") from exc


def require_clean_git_source(repository: str | Path) -> str:
    root = Path(repository).resolve()
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise ValueError("evidence lane requires a readable Git source identity") from exc
    if Path(top).resolve() != root:
        raise ValueError("evidence lane repository must be the scoped Git root")
    if status:
        raise ValueError("evidence lane requires a clean source tree")
    return revision


def _resolved_for_grow(
    repository: EvolutionRepository,
    program_ref: str,
) -> ResolvedProgramSnapshot:
    resolved = repository.resolve(program_ref)
    if resolved.lane is not ProgramLane.GROW:
        raise ValueError("grow execution requires a grow ProgramSnapshot")
    return resolved


def _resolved_for_evidence(
    repository: EvolutionRepository,
    frozen_path: str | Path,
) -> ResolvedProgramSnapshot:
    frozen = load_frozen_program(frozen_path)
    source_program = repository.programs.get(
        f"{frozen.resolved.program_id}@{frozen.resolved.version}"
    )
    if source_program is None:
        raise ValueError("frozen program source is absent from the manifest repository")
    current = repository.resolve(source_program.ref)
    expected = freeze_program(current)
    if expected.freeze_hash != frozen.freeze_hash:
        raise ValueError("frozen program no longer matches immutable source manifests")
    require_clean_git_source(repository.root)
    return frozen.resolved


def _build_runtime_model(graph: ComponentGraphNode, device: str) -> torch.nn.Module:
    if not graph.executable:
        raise ValueError(f"component graph is exercise-only and not executable: {graph.ref}")
    options = dict(graph.builder_options)
    profile = options.pop("profile", None)
    config_values = options.pop("reference_config", None)
    if not isinstance(profile, str) or not isinstance(config_values, dict):
        raise ValueError("component graph runtime requires profile and reference_config")
    config = ReferenceConfig(**config_values)
    model = build_model(graph.builder_id, profile=profile, config=config, **options)
    return model.to(torch.device(device))


def _objective(bundle: ObjectiveBundleNode) -> Objective:
    weights = {term.objective_id: term.weight for term in bundle.objectives}
    raw_mse_id = "tabu.objective.numeric_mse"
    standardized_mse_id = "tabu.objective.context_standardized_numeric_mse"
    if raw_mse_id in weights and standardized_mse_id in weights:
        raise ValueError("objective bundle cannot mix raw and context-standardized numeric MSE")
    if standardized_mse_id in weights:
        mse = weights[standardized_mse_id]
        numeric_target_coordinate = "context_standardized"
    else:
        mse = weights.get(raw_mse_id, 1.0)
        numeric_target_coordinate = "raw"
    mae = weights.get("tabu.objective.numeric_mae", 0.0)
    categorical = weights.get("tabu.objective.categorical_nll", 1.0)
    return Objective(
        mse_weight=mse,
        mae_weight=mae,
        categorical_nll_weight=categorical,
        include_categorical="tabu.objective.categorical_nll" in weights,
        numeric_target_coordinate=numeric_target_coordinate,
    )


def _scheduler(
    trainer: Trainer,
    recipe: TrainingRecipeNode,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if recipe.scheduler == "none":
        return None
    assert recipe.scheduler_step_size is not None
    assert recipe.scheduler_gamma is not None
    return torch.optim.lr_scheduler.StepLR(
        trainer.optimizer,
        step_size=recipe.scheduler_step_size,
        gamma=recipe.scheduler_gamma,
    )


def _execution_code_hash(
    repository: EvolutionRepository,
    resolved: ResolvedProgramSnapshot,
    model: torch.nn.Module,
) -> str:
    repository_root = repository.root
    core_paths = {
        repository_root / "src/tabu_lab/evolution/models.py",
        repository_root / "src/tabu_lab/evolution/repository.py",
        repository_root / "src/tabu_lab/evolution/policy.py",
        repository_root / "src/tabu_lab/evolution/checkpoint.py",
        repository_root / "src/tabu_lab/evolution/runtime.py",
        repository_root / "src/tabu_lab/training/objective.py",
        repository_root / "src/tabu_lab/training/trainer.py",
        repository_root / "src/tabu_lab/models/builders.py",
        repository_root / "src/tabu_lab/models/component_registry.py",
        repository_root / "src/tabu_lab/models/reference.py",
        repository_root / "src/tabu_lab/models/types.py",
        repository_root / "src/tabu_lab/evidence/canonical.py",
        repository_root / "src/tabu_lab/evidence/schemas.py",
        repository_root / "pyproject.toml",
        repository_root / "uv.lock",
    }
    core_paths.update((repository_root / "src/tabu_lab/contracts").glob("*.py"))
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(core_paths):
        if not path.is_file():
            raise ValueError(
                "execution identity source is missing: "
                f"{path.relative_to(repository_root).as_posix()}"
            )
        files[path.relative_to(repository_root).as_posix()] = {
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    selected_bindings: dict[str, dict[str, Any]] = {}
    for resolved_node in resolved.dependency_closure:
        node = repository.node(resolved_node.ref)
        for field_name in ("source", "implementation"):
            binding = getattr(node, field_name, None)
            if binding is not None:
                selected_bindings[f"{resolved_node.ref}:{field_name}"] = {
                    "source_path": binding.source_path,
                    "symbol_ref": binding.symbol_ref,
                    "hash_mode": binding.hash_mode,
                    "sha256": binding.sha256,
                }
    runtime_symbols: dict[str, str] = {}
    for model_class in type(model).__mro__:
        if not model_class.__module__.startswith("tabu_lab.models"):
            continue
        source = inspect.getsource(model_class).encode("utf-8")
        runtime_symbols[
            f"{model_class.__module__}:{model_class.__qualname__}"
        ] = canonical_hash({"python_source": source.decode("utf-8")})
    return canonical_hash(
        {
            "schema_version": "tabu.execution-source-tree.v1",
            "manifest_closure_hash": resolved.manifest_closure_hash,
            "core_files": files,
            "selected_source_bindings": selected_bindings,
            "runtime_model_classes": runtime_symbols,
        }
    )


def _execution_environment(device: str, lane: ProgramLane) -> dict[str, Any]:
    return {
        "schema_version": "tabu.program-execution-config.v1",
        "device": str(torch.device(device)),
        "dtype": "float32",
        "lane": lane.value,
        "python_version": platform.python_version(),
        "operating_system": platform.system().lower(),
        "architecture": platform.machine().lower(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
        "cuda_runtime": None if torch.version.cuda is None else str(torch.version.cuda),
        "cudnn_version": torch.backends.cudnn.version(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def _run_identity(
    *,
    repository: EvolutionRepository,
    resolved: ResolvedProgramSnapshot,
    model: torch.nn.Module,
    mixture: WorldMixtureNode,
    training_config: dict[str, Any],
    execution_config: dict[str, Any],
    seeds: Mapping[str, int],
) -> RunIdentity:
    semantic_hash = getattr(model, "semantic_config_hash", None)
    if semantic_hash is None:
        semantic_hash = getattr(getattr(model, "config", None), "semantic_hash", None)
    if not isinstance(semantic_hash, str):
        raise ValueError("program model must expose semantic_config_hash")
    return RunIdentity.create(
        spec_hash=resolved.snapshot_hash,
        code_hash=_execution_code_hash(repository, resolved, model),
        data_hash=canonical_hash(
            {
                "world_mixture_hash": mixture.node_hash,
                "generator_hashes": {
                    node.ref: node.node_hash
                    for node in (
                        repository.node(entry.generator)
                        for entry in mixture.entries
                    )
                },
            }
        ),
        split_hash=canonical_hash({"partition": "synthetic_train", "policy": "world_addressed"}),
        compiler_hash=canonical_hash(
            {
                "evidence": "tabu.evidence-episode@3",
                "truth": "tabu.truth-sidecar@1",
                "prediction": "tabu.prediction-bundle@1",
            }
        ),
        semantic_config_hash=semantic_hash,
        execution_config_hash=canonical_hash(execution_config),
        training_config_hash=canonical_hash(training_config),
        seeds=seeds,
    )


def _episode_pair(
    node: GeneratorNode,
    *,
    recipe: TrainingRecipeNode,
    root_seed: int,
    step: int,
) -> tuple[EvidenceEpisode, TruthSidecar]:
    runtime = _import_symbol(node.runtime_ref)
    if not callable(runtime):
        raise TypeError(f"generator runtime is not callable: {node.runtime_ref}")
    overlap = set(node.immutable_config).intersection(recipe.episode_options)
    if overlap:
        raise ValueError(
            "training recipe cannot override immutable generator config: "
            + ", ".join(sorted(overlap))
        )
    options = {**node.immutable_config, **recipe.episode_options}
    generated = runtime(
        root_seed=root_seed,
        world_id=f"program-step-{step:08d}",
        partition="train",
        **options,
    )
    evidence = getattr(generated, "evidence", None)
    truth = getattr(generated, "sidecar", None)
    if not isinstance(evidence, EvidenceEpisode) or not isinstance(truth, TruthSidecar):
        raise TypeError("generator runtime must return evidence and sidecar contracts")
    return evidence, truth


def _validate_warm_start(
    repository: EvolutionRepository,
    source: ResolvedProgramSnapshot,
    target: ResolvedProgramSnapshot,
) -> StateProjectionNode:
    projection_ref = target.slots.get("state_projection")
    if projection_ref is None:
        raise ValueError("warm start requires a target StateProjection")
    projection = repository.node(projection_ref.ref)
    if not isinstance(projection, StateProjectionNode) or not projection.verified:
        raise ValueError("warm start StateProjection is not verified")
    source_graph = source.slots["component_graph"]
    target_graph = target.slots["component_graph"]
    if (
        projection.source_model.ref != source.slots["model_contract"].ref
        or projection.source_graph.ref != source_graph.ref
        or projection.target_model.ref != target.slots["model_contract"].ref
        or projection.target_graph.ref != target_graph.ref
    ):
        raise ValueError("StateProjection endpoints do not match warm-start snapshots")
    if not repository.compatibility_edges(
        source_graph,
        target_graph,
        CompatibilityDisposition.WARM_START_AVAILABLE,
    ):
        raise ValueError("warm start has no verified compatibility edge")
    return projection


@dataclass(frozen=True)
class ProgramRunResult:
    receipt: ProgramRunReceipt
    checkpoint: Path
    checkpoint_sidecar: Path
    receipt_path: Path


def run_program(
    repository: EvolutionRepository,
    *,
    lane: ProgramLane,
    output_root: str | Path,
    device: str = "cpu",
    program_ref: str | None = None,
    frozen_path: str | Path | None = None,
    resume_checkpoint: str | Path | None = None,
    warm_start_checkpoint: str | Path | None = None,
    warm_start_source_program: str | None = None,
    max_updates_this_invocation: int | None = None,
) -> ProgramRunResult:
    if (program_ref is None) == (frozen_path is None):
        raise ValueError("select exactly one grow program or frozen evidence program")
    if resume_checkpoint is not None and warm_start_checkpoint is not None:
        raise ValueError("exact resume and warm start are mutually exclusive")
    if warm_start_source_program is not None and warm_start_checkpoint is None:
        raise ValueError("warm-start source program requires a warm-start checkpoint")
    if max_updates_this_invocation is not None and max_updates_this_invocation <= 0:
        raise ValueError("max_updates_this_invocation must be positive")
    if lane is ProgramLane.GROW:
        if program_ref is None:
            raise ValueError("grow lane requires program_ref")
        resolved = _resolved_for_grow(repository, program_ref)
        evidence_status = EvidenceStatus.LOCAL_UNISSUED
    else:
        if frozen_path is None:
            raise ValueError("evidence lane requires frozen_path")
        resolved = _resolved_for_evidence(repository, frozen_path)
        evidence_status = EvidenceStatus.EVIDENCE_CANDIDATE_UNREVIEWED

    destination = Path(output_root)
    if destination.exists():
        raise ValueError("refusing to overwrite an existing program run directory")

    graph = repository.node(resolved.slots["component_graph"].ref)
    model_contract = repository.node(resolved.slots["model_contract"].ref)
    mixture = repository.node(resolved.slots["world_mixture"].ref)
    policy_node = repository.node(resolved.slots["sampling_policy"].ref)
    objective_bundle = repository.node(resolved.slots["objective_bundle"].ref)
    recipe = repository.node(resolved.slots["training_recipe"].ref)
    assert isinstance(graph, ComponentGraphNode)
    from .models import ModelContractNode

    assert isinstance(model_contract, ModelContractNode)
    if not model_contract.executable:
        raise ValueError(
            f"ModelContract is exercise-only and not executable: {model_contract.ref}"
        )
    assert isinstance(mixture, WorldMixtureNode)
    assert isinstance(objective_bundle, ObjectiveBundleNode)
    assert isinstance(recipe, TrainingRecipeNode)

    seeds = {"model": 1729, "episode": 2718, "sampler": 31415}
    torch.manual_seed(seeds["model"])
    random.seed(seeds["model"])
    np.random.seed(seeds["model"])
    model = _build_runtime_model(graph, device)
    training_config = {
        "schema_version": "tabu.program-training-config.v1",
        "snapshot_hash": resolved.snapshot_hash,
        "training_recipe_hash": recipe.node_hash,
        "objective_bundle_hash": objective_bundle.node_hash,
        "world_mixture_hash": mixture.node_hash,
        "sampling_policy_hash": policy_node.node_hash,
        "target_steps": recipe.max_steps,
    }
    resume_metadata = None
    warm_source_metadata = None
    warm_projection = None
    if resume_checkpoint is not None:
        resume_metadata = read_program_checkpoint(resume_checkpoint)
        initialization = resume_metadata.initialization
    elif warm_start_checkpoint is not None:
        warm_sidecar = program_sidecar_path(warm_start_checkpoint)
        if warm_sidecar.is_file():
            warm_source_metadata = read_program_checkpoint(warm_start_checkpoint)
            warm_source_resolved = warm_source_metadata.resolved_snapshot
            if warm_start_source_program is not None and (
                repository.resolve(warm_start_source_program).snapshot_hash
                != warm_source_resolved.snapshot_hash
            ):
                raise ValueError(
                    "declared warm-start source program does not match checkpoint"
                )
            source_checkpoint_kind = ProgramCheckpointKind.PROGRAM_FULL_STATE
        else:
            if warm_start_source_program is None:
                raise ValueError(
                    "weights-only warm start requires --warm-start-source-program"
                )
            warm_source_resolved = repository.resolve(warm_start_source_program)
            source_checkpoint_kind = ProgramCheckpointKind.WEIGHTS_ONLY
        warm_projection = _validate_warm_start(
            repository,
            warm_source_resolved,
            resolved,
        )
        initialization = ProgramInitialization(
            mode=ProgramInitializationMode.WARM_START,
            projection_ref=warm_projection.ref,
            source_checkpoint_kind=source_checkpoint_kind,
            source_snapshot_hash=warm_source_resolved.snapshot_hash,
            source_run_identity_hash=(
                None
                if warm_source_metadata is None
                else warm_source_metadata.run_identity_hash
            ),
            source_checkpoint_sha256=file_sha256(warm_start_checkpoint),
            source_lane=(
                None if warm_source_metadata is None else warm_source_metadata.lane
            ),
            source_evidence_status=(
                None
                if warm_source_metadata is None
                else warm_source_metadata.evidence_status
            ),
        )
    else:
        initialization = ProgramInitialization()
    training_config["initialization"] = initialization.model_dump(mode="json")
    execution_config = _execution_environment(device, lane)
    identity = _run_identity(
        repository=repository,
        resolved=resolved,
        model=model,
        mixture=mixture,
        training_config=training_config,
        execution_config=execution_config,
        seeds=seeds,
    )
    named_generators = {
        "episode": torch.Generator(device="cpu").manual_seed(seeds["episode"]),
        "sampler": torch.Generator(device="cpu").manual_seed(seeds["sampler"]),
    }
    trainer = Trainer(
        model,
        objective=_objective(objective_bundle),
        learning_rate=recipe.learning_rate,
        run_identity=identity,
        training_config=training_config,
        execution_config=execution_config,
        named_generators=named_generators,
    )
    scheduler = _scheduler(trainer, recipe)
    policy = SamplingPolicyEngine(policy_node, mixture)

    if resume_checkpoint is not None:
        load_program_checkpoint(
            trainer,
            resume_checkpoint,
            resolved_snapshot=resolved,
            lane=lane,
            evidence_status=evidence_status,
            policy=policy,
            scheduler=scheduler,
            initialization=initialization,
        )
    elif warm_start_checkpoint is not None:
        assert warm_projection is not None
        projection_runtime = _import_symbol(
            warm_projection.implementation.symbol_ref or ""
        )
        source_state = read_checkpoint_model_state(warm_start_checkpoint)
        projected = projection_runtime(source_state, model.state_dict())
        model.load_state_dict(projected, strict=True)

    remaining = recipe.max_steps - trainer.step
    if remaining <= 0:
        raise ValueError("program checkpoint already reached the immutable training budget")
    invocation_budget = (
        remaining
        if max_updates_this_invocation is None
        else min(remaining, max_updates_this_invocation)
    )
    destination.mkdir(parents=True)
    checkpoint: Path | None = None
    for _ in range(invocation_budget):
        generator_ref = policy.choose(trainer.named_generators["sampler"])
        generator_node = repository.node(generator_ref)
        if not isinstance(generator_node, GeneratorNode):
            raise EvolutionManifestError("sampling policy selected a non-generator node")
        episode_seed = int(
            torch.randint(
                0,
                2**31 - 1,
                (1,),
                generator=trainer.named_generators["episode"],
            ).item()
        )
        evidence, truth = _episode_pair(
            generator_node,
            recipe=recipe,
            root_seed=episode_seed,
            step=trainer.step,
        )
        result = trainer.train_step(evidence, truth)
        if scheduler is not None:
            scheduler.step()
        policy.observe(float(result.loss.total.detach().cpu()))
        if trainer.step % recipe.checkpoint_interval == 0:
            checkpoint = destination / f"checkpoint-step-{trainer.step:08d}.safetensors"
            save_program_checkpoint(
                trainer,
                checkpoint,
                resolved_snapshot=resolved,
                lane=lane,
                evidence_status=evidence_status,
                policy=policy,
                scheduler=scheduler,
                initialization=initialization,
                target_steps=recipe.max_steps,
            )
        # ``TrainStep`` intentionally exposes prediction/loss tensors for local
        # diagnostics.  Keeping the previous result alive while constructing
        # the next quadratic routing ledger doubles peak memory on broad-row
        # episodes, so the program runner releases its step-local graph here.
        del result, evidence, truth

    invocation_checkpoint = destination / f"checkpoint-step-{trainer.step:08d}.safetensors"
    if checkpoint != invocation_checkpoint:
        checkpoint = invocation_checkpoint
        save_program_checkpoint(
            trainer,
            checkpoint,
            resolved_snapshot=resolved,
            lane=lane,
            evidence_status=evidence_status,
            policy=policy,
            scheduler=scheduler,
            initialization=initialization,
            target_steps=recipe.max_steps,
        )
    assert checkpoint is not None
    sidecar = program_sidecar_path(checkpoint)
    status = (
        ProgramRunStatus.COMPLETED
        if trainer.step == recipe.max_steps
        else ProgramRunStatus.INTERRUPTED
    )
    artifacts = (
        ProgramArtifact(
            name=checkpoint.name,
            sha256=file_sha256(checkpoint),
            size_bytes=checkpoint.stat().st_size,
        ),
        ProgramArtifact(
            name=sidecar.name,
            sha256=file_sha256(sidecar),
            size_bytes=sidecar.stat().st_size,
        ),
    )
    receipt_payload = {
        "schema_version": "tabu.program-run-receipt.v1",
        "lane": lane,
        "evidence_status": evidence_status,
        "status": status,
        "resolved_snapshot": resolved,
        "run_identity": identity,
        "initialization": initialization,
        "training_config": training_config,
        "execution_config": execution_config,
        "snapshot_hash": resolved.snapshot_hash,
        "run_identity_hash": identity.identity_hash,
        "step": trainer.step,
        "target_steps": recipe.max_steps,
        "policy_state_hash": policy.state.state_hash,
        "artifacts": artifacts,
    }
    receipt = ProgramRunReceipt(
        **receipt_payload,
        receipt_hash=canonical_hash(receipt_payload),
    )
    receipt_path = destination / "run-receipt.json"
    receipt_path.write_text(
        canonical_json(receipt.model_dump(mode="python")) + "\n",
        encoding="utf-8",
    )
    return ProgramRunResult(
        receipt=receipt,
        checkpoint=checkpoint,
        checkpoint_sidecar=sidecar,
        receipt_path=receipt_path,
    )


__all__ = [
    "ProgramRunResult",
    "freeze_program",
    "identity_state_projection",
    "load_frozen_program",
    "require_clean_git_source",
    "run_program",
]
