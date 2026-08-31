# Experiments ledger

## Current lane

The active research lane is
`tabu.query.row@0.2.0` + `supervised.label_broadcast.v1` + symmetric anchored
readout, moving through:

1. broad supervised synthetic prior v3 validation;
2. identity-bound, resumable TabUR pretraining;
3. held-out synthetic frozen ICL;
4. full-train/full-test real-data frozen ICL;
5. same-initialization pretrained-vs-scratch real-task fine-tuning.

Only the first item has a candidate implementation. It has targeted local smoke
evidence but is not yet the standard long-run data source. Items 2–5 must not
inherit v1/v2 or `tabu.query.row@0.1.0` results.

## Default real-data estimand

For real-data model comparisons, the default estimand is the complete train/test
split:

- make one deterministic split before compilation;
- place every labeled train row in model context;
- place every held-out test row in the query set;
- score every held-out row against scorer-only truth;
- fit classical baselines on all and only the same train partition.

Finite context or query limits are diagnostic overrides. They must be explicit
in the manifest and report and must not replace the default table-foundation-
model result. See
[`real-evaluation-default-protocol.md`](../docs/architecture/real-evaluation-default-protocol.md).

## State routing

Every experiment surface must declare one of these roles:

| Role | Meaning |
| --- | --- |
| `active_candidate` | code or config under current validation; not yet the default runner |
| `candidate_preregistered` | protocol frozen enough to run, but execution or review is incomplete |
| `local_unissued` | a local run exists; it is not a formal receipt or accepted claim |
| `historical` | preserved for provenance; not an active instruction or inherited baseline |
| `formal` | immutable receipt and required independent review are complete |

YAML parseability, a completed process, or a summary score never promotes an
experiment between these roles by itself.

## Current routing

- Synthetic-prior candidate:
  [`query_row_supervised_synthetic_v3.py`](../src/tabu_lab/experiments/query_row_supervised_synthetic_v3.py)
- Focused validation:
  [`test_query_row_supervised_synthetic_v3.py`](../tests/unit/test_query_row_supervised_synthetic_v3.py)
- Existing R5 pretraining runner: v2-bound historical execution path; not a v3
  long-run entry
- Full-context OpenML protocol:
  [`transfer-query-v2/openml-full-context-2026-08-31.yaml`](./transfer-query-v2/openml-full-context-2026-08-31.yaml)
  is a candidate preregistration and must be versioned for the 0.2/v3 identity
  before execution
- Historical results and artifact hashes:
  [`local-artifact-index.json`](../docs/reports/local-artifact-index.json)

The dated one-shot optimization guide applies only to the legacy free-readout
`tabu.query.row@0.1.0`. It is a provenance back-link, not the current execution
guide.

## Per-run record

One directory or file per experiment. Record at least:

- hypothesis and gate criterion before execution;
- exact contract, generator, source commit, config hash, seed, and compute;
- train/validation/test split identities and data authority;
- checkpoint identity and, for long runs, optimizer/cursor/RNG resume state;
- raw metrics and curves, including baseline arms on identical examples;
- verdict: `pass`, `kill`, or `revise`, with failures preserved;
- `harness_status`, `run_status`, `evidence_level`, and `claim_status` separately.

Naming remains `G<N>-<slug>/` for gate directories. A new active configuration
must replace the previous default in the entry documents; old configurations
move to `historical` rather than accumulating as competing defaults.

<!-- seed: If v3 cannot be made runner-bound without changing its loss coordinate
or compute envelope, version the prior/runner contract explicitly instead of
silently patching the meaning of an existing run. -->
