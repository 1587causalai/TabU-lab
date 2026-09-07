# Experiments ledger

## TAR training routes

The current research model is **TabU-TAR**, explicitly selected as `tabu.tar`.
Single-episode and batch-episode training are both continuing research routes.
Preserve their separate configurations, source snapshots, optimizer states and
results as each route develops.

| Route | Available entry in this checkout | Evidence and development scope |
| --- | --- | --- |
| Single episode | `uv run tabu-lab tar joint-fit` with explicit preregistration | One episode and one update per table per round; published 120-table baseline completed 768 rounds. |
| Batch episodes | `TARTrainer` with an explicit `effective_episode_batch` accumulates episodes serially | The separate parallel episode runner is not yet integrated here. Batch construction, optimization and execution remain active research questions. |

The existing `forward_batch` and trainer loops do not establish parallel episode
execution. Row/column batching inside one episode is a separate implementation
feature. Integrate a parallel runner with its own reviewed recipe and evidence;
do not relabel the existing accumulation API or overwrite the single-episode route.

Current entries:

- [120-table recipe](local/tar-diverse-120-fit/README.md): frozen corpus,
  same-model fitting and source-bound segments.
- [Completed 768-round result](../docs/research/tar-shared-fit-20260907/README.md):
  fixed-training-table evaluation, curves and next research questions.
- [Eight-table shared-fit recipe](local/tar-unified-joint-fit/README.md).
- [TAR guide](../docs/tutorials/tabu-tar.md): size/encoding selection, objective
  and the distinction between accumulation and parallel execution.
- [Small validation](local/tar-small-validation/README.md): bounded component
  and real-data diagnostics, separate from the shared-fit baseline.

For cross-route comparisons, report both episode and optimizer-update budgets,
wall-clock time, the exact episode/evaluation streams, precision and initial
weights. Equal episode counts do not imply equal optimizer trajectories. A
batch-size result describes its recipe, not the entire batch training route.

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

## Historical query-family routing

The entries below preserve the query-row/query-base program lane. They do not
select the TAR research model. `MAINLINE.yaml` continues to bind that separate
program system; see the [compatibility guide](../docs/history/compatibility-runtime.md).

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

Naming remains `G<N>-<slug>/` for gate directories. Within each research route,
identify the exact active configuration and preserve superseded runs as history.
Single-episode and batch-episode routes coexist; a new configuration in one
route does not retire the other.

<!-- seed: If v3 cannot be made runner-bound without changing its loss coordinate
or compute envelope, version the prior/runner contract explicitly instead of
silently patching the meaning of an existing run. -->
