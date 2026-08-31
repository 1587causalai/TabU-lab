#!/usr/bin/env python3
"""Run the six-step TabUR evaluation ladder as one local diagnostic bundle.

Real-data stages use every train-partition row and every held-out row by
default; finite label/test limits are explicit bounded diagnostics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabu_lab.experiments import (
    make_query_row_synthetic_episode,
    run_query_row_finetune_lift,
    run_query_row_fixed_world_fit,
    run_query_row_frozen_icl,
    run_query_row_multi_world_fit,
    run_query_row_real_scratch_benchmark,
)
from tabu_lab.experiments.tabubase_scale import resolve_device
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig
from tabu_lab.verification import (
    QueryClaimStatus,
    QueryEvaluationStage,
    QueryEvaluationStageResult,
    QueryEvidenceLevel,
    QueryHarnessStatus,
    QueryRunStatus,
    QueryVerificationCheck,
    TabUQueryEvaluationLadder,
    assess_query_runtime_growth,
    verify_tabu_query_row_component_correctness,
    verify_tabu_query_row_component_evolvability,
)


def _config(*, row_token_count: int) -> ReferenceConfig:
    return ReferenceConfig(
        d_model=8,
        n_heads=2,
        d_ff=16,
        n_blocks=1,
        inducing_slots=2,
        matched_slots=row_token_count,
        max_features=256,
    )


def _executed_stage(
    stage: QueryEvaluationStage,
    *,
    passed: bool,
    checks: tuple[tuple[str, bool], ...],
) -> QueryEvaluationStageResult:
    return QueryEvaluationStageResult(
        schema_version="tabu.query-evaluation-stage.v1",
        stage=stage,
        harness_status=QueryHarnessStatus.IMPLEMENTED,
        run_status=QueryRunStatus.PASSED if passed else QueryRunStatus.FAILED,
        evidence_level=QueryEvidenceLevel.LOCAL_UNISSUED,
        claim_status=QueryClaimStatus.NONE,
        checks=tuple(QueryVerificationCheck(check_id=name, passed=value) for name, value in checks),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--row-token-count", type=int, default=4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--pretrain-steps", type=int, default=20)
    parser.add_argument("--pretrain-worlds", type=int, default=4)
    parser.add_argument(
        "--label-budget",
        type=int,
        default=None,
        help="Optional bounded context override; default uses every train-partition row.",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional bounded query override; default evaluates every held-out row.",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = _config(row_token_count=args.row_token_count)
    device = resolve_device(args.device)
    row = build_model(
        "tabu.query.row",
        config=config,
        profile="completion.artificial_mask.v1",
        row_token_count=args.row_token_count,
    ).to(device)
    correctness = verify_tabu_query_row_component_correctness(row)
    base = build_model(
        "tabu.query.base",
        config=config,
        profile="completion.artificial_mask.v1",
    ).to(device)
    episode = make_query_row_synthetic_episode(
        seed=args.seed + 1,
        rows=16,
        row_token_count=args.row_token_count,
    ).evidence
    growth = assess_query_runtime_growth(base, row, episode)
    evolvability = verify_tabu_query_row_component_evolvability((growth,))

    f0 = run_query_row_fixed_world_fit(
        seed=args.seed,
        steps=args.steps,
        row_token_count=args.row_token_count,
        device=device,
    )
    s1 = run_query_row_multi_world_fit(
        seed=args.seed,
        steps=args.steps,
        train_worlds=4,
        validation_worlds=2,
        row_token_count=args.row_token_count,
        device=device,
    )
    real = run_query_row_real_scratch_benchmark(
        dataset_ids=("iris", "diabetes"),
        label_budget=args.label_budget,
        updates=args.updates,
        test_limit=args.test_limit,
        row_token_count=args.row_token_count,
        device=device,
        seed=args.seed,
    )
    frozen = run_query_row_frozen_icl(
        seed=args.seed,
        pretrain_steps=args.pretrain_steps,
        pretrain_worlds=args.pretrain_worlds,
        row_token_count=args.row_token_count,
        device=device,
    )
    lift = run_query_row_finetune_lift(
        dataset_ids=("iris", "diabetes"),
        label_budget=args.label_budget,
        updates=args.updates,
        pretrain_steps=args.pretrain_steps,
        pretrain_worlds=args.pretrain_worlds,
        test_limit=args.test_limit,
        row_token_count=args.row_token_count,
        seed=args.seed,
        device=device,
    )

    stages = (
        correctness,
        evolvability,
        _executed_stage(
            QueryEvaluationStage.SYNTHETIC_FIT,
            passed=f0.status == "pass" and s1.status == "pass",
            checks=(
                ("f0_fixed_world", f0.status == "pass"),
                ("s1_multi_world", s1.status == "pass"),
            ),
        ),
        _executed_stage(
            QueryEvaluationStage.REAL_SCRATCH_PREDICTION,
            passed=real.status == "pass",
            checks=(("scratch_only_finite", real.status == "pass"),),
        ),
        _executed_stage(
            QueryEvaluationStage.FROZEN_ICL,
            passed=frozen.status == "pass",
            checks=(("no_optimizer_parameter_mutation", frozen.status == "pass"),),
        ),
        _executed_stage(
            QueryEvaluationStage.FINETUNE_LIFT,
            passed=lift.status == "pass",
            checks=(("paired_profile_compatible_finite", lift.status == "pass"),),
        ),
    )
    ladder = TabUQueryEvaluationLadder.with_stage_results(
        stages,
        contract_id="tabu.query.row",
        contract_version="0.2.0",
    )
    payload = {
        "schema_version": "tabu.query-row.evaluation-ladder-result.v1",
        "row_readout": row.geometry.readout_identity(),
        "model_spec_hash": row.model_spec_hash,
        "variant_hash": row.variant_ref.semantic_hash,
        "ladder": ladder.model_dump(mode="json"),
        "stage_results": [stage.model_dump(mode="json") for stage in stages],
        "stage_3_synthetic_fit": {"f0_fixed_world": f0.as_dict(), "s1_multi_world": s1.as_dict()},
        "stage_4_real_scratch": real.as_dict(),
        "stage_5_frozen_icl": frozen.as_dict(),
        "stage_6_finetune_lift": lift.as_dict(),
        "evidence_status": "local_unissued",
        "claim_boundary": (
            "six-step TabUR local diagnostics only; no formal receipt or accepted "
            "capability claim"
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
        return
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
