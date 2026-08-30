#!/usr/bin/env python3
"""Evaluate one verified local-unissued fit checkpoint without catalog promotion.

This is an exploratory bridge, not a publication path.  It verifies the fit
attempt, reconstructs a transient produced artifact identity, runs the normal
isolated Evaluation Foundry adapter, and emits a local-unissued evaluation
receipt.  Formal evaluation still requires a cataloged artifact and reviewed
dataset/source authorities through ``tabu-lab eval run --formal``.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from safetensors import safe_open

from tabu_lab.adapters.checkpoint_model import (
    CatalogedCheckpointModelAdapter,
    _adapter_spec,
)
from tabu_lab.adapters.eval_data_workflow import load_prepared_eval_bundle
from tabu_lab.adapters.real_eval_data import (
    checkpoint_blind_example,
    checkpoint_topology_cases,
)
from tabu_lab.catalog import ArtifactStatusEvent, EvidencePointer, ModelArtifact
from tabu_lab.contracts import canonical_hash
from tabu_lab.evaluation.fit_artifacts import (
    capture_environment,
    verify_fit_attempt_artifacts,
)
from tabu_lab.evaluation.foundry import (
    AdapterLaunchSpec,
    EvalProducerBinding,
    bind_evaluation_receipt,
    load_suite,
    run_evaluation,
)
from tabu_lab.evidence import ReceiptStatus, RunBundle, SourceIdentity
from tabu_lab.experiments import ModelSemanticConfig
from tabu_lab.registry import get_model_spec


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _checkpoint_resume(path: Path) -> dict[str, object]:
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        encoded = (checkpoint.metadata() or {}).get("tabu_training_state")
    if not isinstance(encoded, str):
        raise ValueError("checkpoint is missing TabU training metadata")
    header = json.loads(encoded)
    if not isinstance(header, dict) or not isinstance(header.get("resume_contract"), dict):
        raise ValueError("checkpoint training metadata is malformed")
    return header["resume_contract"]


def _transient_artifact(attempt: Path) -> tuple[ModelArtifact, object, dict[str, object]]:
    receipt = verify_fit_attempt_artifacts(attempt)
    if receipt.status is not ReceiptStatus.SUCCEEDED:
        raise ValueError("local model evaluation requires a successful fit receipt")
    if receipt.metadata.get("issuance_status") != "local_unissued":
        raise ValueError("this bridge accepts local_unissued fit attempts only")

    bundle = RunBundle.model_validate(_read_json(attempt / "run_bundle.json"))
    metadata = bundle.metadata
    attempt_id = metadata.get("attempt_id")
    contract_version = metadata.get("contract_version")
    license_id = metadata.get("checkpoint_license_id")
    if not all(isinstance(value, str) for value in (attempt_id, contract_version, license_id)):
        raise ValueError("fit RunBundle lacks checkpoint identity metadata")

    checkpoints = tuple(item for item in receipt.artifacts if item.kind == "checkpoint")
    if len(checkpoints) != 1:
        raise ValueError("successful fit receipt must bind exactly one checkpoint")
    checkpoint_ref = checkpoints[0]
    checkpoint_path = attempt / checkpoint_ref.uri
    resume = _checkpoint_resume(checkpoint_path)

    model_spec = get_model_spec(bundle.model_id)
    semantic_payload = _read_json(attempt / "resolved-configs" / "semantic.json")
    semantic = ModelSemanticConfig.model_validate(semantic_payload)
    compiler = _read_json(attempt / "compiler-manifest.json")
    model_spec_hash = canonical_hash(model_spec.model_dump(mode="json"))
    if model_spec_hash != resume.get("model_spec_hash"):
        raise ValueError("installed ModelSpec differs from checkpoint")
    if semantic.content_hash != bundle.identity.semantic_config_hash:
        raise ValueError("semantic config differs from fit RunIdentity")
    if canonical_hash(compiler) != bundle.identity.compiler_hash:
        raise ValueError("compiler manifest differs from fit RunIdentity")

    prefix = f"local-unissued/{receipt.run_id}/{attempt_id}"
    receipt_pointer = EvidencePointer(
        uri=f"{prefix}/receipt.json",
        sha256=receipt.receipt_hash,
        media_type="application/json",
    )
    artifact = ModelArtifact(
        artifact_id=f"{receipt.run_id}.{attempt_id}.local-unissued-checkpoint",
        contract_id=bundle.model_id,
        contract_version=contract_version,
        producer_run_id=receipt.run_id,
        producer_receipt=receipt_pointer,
        checkpoint=EvidencePointer(
            uri=f"{prefix}/{checkpoint_ref.uri}",
            sha256=checkpoint_ref.sha256,
            media_type=checkpoint_ref.media_type,
        ),
        checkpoint_format="safetensors",
        checkpoint_schema_version=str(resume["checkpoint_schema_version"]),
        model_state_schema_version=str(resume["model_state_schema_version"]),
        model_spec=EvidencePointer(
            uri=f"specs/models/{bundle.model_id}.yaml",
            sha256=model_spec_hash,
            media_type="application/yaml",
        ),
        semantic_config=EvidencePointer(
            uri=f"{prefix}/resolved-configs/semantic.json",
            sha256=semantic.content_hash,
            media_type="application/json",
        ),
        compiler_manifest=EvidencePointer(
            uri=f"{prefix}/compiler-manifest.json",
            sha256=canonical_hash(compiler),
            media_type="application/json",
        ),
        license_id=license_id,
        status_history=(
            ArtifactStatusEvent(status="produced", evidence=receipt_pointer),
        ),
    )
    return artifact, semantic, compiler


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", required=True, type=Path)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    attempt = args.attempt.resolve()
    artifact, semantic, compiler = _transient_artifact(attempt)
    model_spec = get_model_spec(artifact.contract_id)
    checkpoint_path = attempt / "checkpoint" / "checkpoint.safetensors"
    adapter = AdapterLaunchSpec(
        module=CatalogedCheckpointModelAdapter.__module__,
        qualname=CatalogedCheckpointModelAdapter.__qualname__,
        kwargs={
            "artifact": artifact.model_dump(mode="json"),
            "checkpoint_path": str(checkpoint_path),
            "compiler_manifest": compiler,
            "device": args.device,
            "model_spec": model_spec.model_dump(mode="json"),
            "semantic_config": semantic.model_dump(mode="json"),
        },
        declared_spec=_adapter_spec(artifact, profile_id=semantic.profile_id),
    )
    producer = EvalProducerBinding(
        provenance="receipted_run",
        run_id=artifact.producer_run_id,
        receipt_sha256=artifact.producer_receipt.sha256,
        receipt_pointer=artifact.producer_receipt.uri,
        publication_eligible=False,
    )
    suite = load_suite(args.suite)
    prepared = load_prepared_eval_bundle(args.prepared).prepared
    blind = tuple(
        checkpoint_blind_example(prepared, example_id=item.example_id)
        for item in prepared.test
    )
    topology = checkpoint_topology_cases(prepared) if prepared.topology_checks else ()
    started_at = datetime.now(UTC)
    result = run_evaluation(
        suite,
        scenario_id=args.scenario,
        adapter=adapter,
        prepared=prepared,
        seed=args.seed,
        producer=producer,
        blind_examples=blind,
        topology_cases=topology,
    )
    environment, _ = capture_environment(args.device)
    result = bind_evaluation_receipt(
        result,
        environment=environment,
        source_identity=SourceIdentity(
            source_kind="local",
            issuance_status="local_unissued",
            reasons=("exploratory_local_checkpoint_evaluation",),
        ),
        started_at=started_at,
        completed_at=datetime.now(UTC),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite evaluation result: {args.output}")
    args.output.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_id": result.result_id,
                "status": result.status.value,
                "metrics": result.metrics,
                "coverage": result.coverage,
                "producer_publication_eligible": result.producer.publication_eligible,
                "execution_receipt_hash": result.execution_receipt.receipt_hash,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
