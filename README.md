# TabU-lab

An open research lab for tabular foundation models, inspired by
[Marin](https://github.com/marin-community/marin).

The repository's single pretraining-direction pointer is
[`MAINLINE.yaml`](./MAINLINE.yaml).
It selects a complete immutable `ProgramSnapshot`: model contract, component
graph, data mixture and policy, objective, training recipe, and evaluation
protocol. Generated catalogs are query projections, not the source of truth.

## Current focus: pretrain TabUBase and TabUR

The primary experimental candidate is **TabUR** under
`tabu.query.row@0.2.0`, using
`supervised.label_broadcast.v1` and the default symmetric `anchored` readout.
**TabUBase** under `tabu.query.base@0.1.0` is an independently trainable sibling,
not a prerequisite checkpoint. Both share the Episode/Prediction and evaluation
protocol boundaries while retaining separate model and run identities.
The immediate research question is whether diverse supervised synthetic
pretraining produces useful frozen ICL and then improves real-task fine-tuning.

| Surface | Current status |
| --- | --- |
| TabUR contract and runtime | implemented under `tabu.query.row@0.2.0` |
| Evolvable program kernel | implemented with immutable manifests, typed DAG validation, impact analysis, freeze, exact resume, and explicit warm start |
| Broad supervised synthetic prior v3 | candidate implementation selectable through versioned Generator/Mixture manifests |
| v3 long-run pretraining | activated as scratch-first Grow snapshots `tabu.pretraining.query-{base,row}@1.2.0`; execution remains `local_unissued` until a run receipt exists |
| frozen ICL for the 0.2/v3 lane | `not_run` |
| real-task pretrained-vs-scratch fine-tuning for the 0.2/v3 lane | `not_run` |
| formal evidence / accepted capability claim | none |

Results from Axis-B TabUBase, `tabu.query.row@0.1.0`, or synthetic priors v1/v2
remain immutable historical evidence. They do not transfer to the current model,
checkpoint identity, or capability claim.

## Five-step runtime contract

The mathematical authority is the Axis-C TabUR source. Runtime preserves the
same five-step boundary:

1. **Compile evidence.** An `EvidenceEpisode` contains only model-visible table
   evidence. Query truth is held outside the model in `TruthSidecar`.
2. **Tokenize query cells.** Visible values, roles, masks, and null state produce
   typed initial cell states $h^{(0)}_{ra}$; target truth is absent.
3. **Run typed dynamics.** The declared row/column source plan updates the
   carrier to $h^{(L)}_{ra}$. Query labels, artificial masks, and null cells are
   receiver-only where the contract requires it.
4. **Read out coordinates.** With $c_{ra}=h^{(L)}_{ra}$, default TabUR uses

   $$
   z_{ra}=\left(W+\gamma\widehat U_rA^\top\right)c_{ra},
   \qquad
   \widehat U_r=\operatorname{LN}(U_r).
   $$

   The general typed form is
   $z_{ra}=(\beta W+\gamma\widehat U_rA^\top)c_{ra}$:
   `homogeneous` gives $Wc$, `anchored` gives the expression above, and `free`
   gives $U_rc$. Here $W$ is a global response parameter, not a Unit; $U_r$ is
   the row-token bank; and effective $A$ has spectral norm one.
5. **Score externally.** The typed terminal returns a `PredictionBundle`; the
   evaluator alone pairs it with `TruthSidecar`. The canonical numeric loss
   coordinate is context-standardized. `numeric_raw_prediction` is an auxiliary
   inverse projection, not the Step-5 training target.

The construction enforces
$K=\text{row-token count}=\text{rows}(W)=\text{coordinate width}=\texttt{matched_slots}$.
See the complete [query runtime mapping](./docs/architecture/query-model-runtime-mapping.md)
and the [TabUR ModelSpec](./specs/models/tabu.query.row.yaml).

## Active defaults

| Decision | Default |
| --- | --- |
| contract | `tabu.query.row@0.2.0` |
| profile | `supervised.label_broadcast.v1` |
| row readout | `anchored` |
| $K$ | `row_token_count=4`, `matched_slots=4` |
| anchored initialization | $\gamma_0=10^{-2}$ |
| synthetic data | broad supervised synthetic prior v3 candidate |
| v3 model capacity | `max_features=1024` for this lane only |
| real-data estimand | all labeled train rows as context; all held-out test rows as queries |
| evidence level before review | `local_unissued` |

The v3 prior currently samples up to 256 predictor columns, plus one response
column. `max_features=1024` is deliberate headroom for the v3 TabUR lane; it is
not a QueryBase-wide architectural default.

## Local readback

Install the frozen development environment and run the focused contract and v3
generator checks:

```bash
uv sync --frozen --extra dev
uv run pytest \
  tests/contract/test_query_base.py \
  tests/unit/test_query_row_supervised_synthetic_v3.py
```

Inspect the complete pretraining program and rehearse a change before spending
compute:

```bash
tabu-lab program validate
tabu-lab program resolve --program tabu.pretraining.query-row@1.2.0
tabu-lab program impact \
  --from-program tabu.pretraining.query-base@1.0.0 \
  --to-program tabu.pretraining.query-base-generator-v3@1.1.0-exercise
```

See [evolvable pretraining programs](./docs/architecture/evolvable-pretraining-programs.md)
for manifest ownership, lane semantics, resume rules, and the three evolution
exercises.

Build the current model explicitly:

```python
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

model = build_model(
    "tabu.query.row",
    config=ReferenceConfig(
        matched_slots=4,
        max_features=1024,
    ),
    profile="supervised.label_broadcast.v1",
    row_token_count=4,
    row_readout_mode="anchored",
    anchored_gamma_initial=1.0e-2,
)
```

The existing `scripts/run_tabur_r5_pretraining.py` is v2-bound. V3 execution
instead goes through `tabu-lab program run`, which binds the generator,
1024-feature capacity graph, loss coordinate, checkpoint identity, policy
state, and exact-resume state in one snapshot. A v2 checkpoint may initialize
an explicit `warm_start` arm through the checked projection, but cannot resume
or inherit the v3 run identity.

## Evaluation default

For the familiar table-foundation-model evaluation, first make one deterministic
train/test split. The model receives every labeled train row as context and must
predict every held-out test row. A finite `context_row_limit` is an explicit
diagnostic override, not the default estimand; it must not be called $K$, which
already denotes TabUR's row-token/coordinate width.

Frozen ICL compares `pretrained_frozen`, `random_init_frozen`, and
`pretrained_shuffled` without constructing an optimizer and with unchanged
parameter hashes. Real-task fine-tuning compares pretrained and scratch arms
from the same root initialization, split, budget, schedule, and seeds.

See the [experiment ledger](./experiments/README.md) and
[real-evaluation default protocol](./docs/architecture/real-evaluation-default-protocol.md).

## Navigation

- Model/runtime authority: [TabUR ModelSpec](./specs/models/tabu.query.row.yaml)
  and [query runtime mapping](./docs/architecture/query-model-runtime-mapping.md)
- Current synthetic-prior candidate:
  [`query_row_supervised_synthetic_v3.py`](./src/tabu_lab/experiments/query_row_supervised_synthetic_v3.py)
- Current evaluation routing: [experiments/README.md](./experiments/README.md)
- Historical local evidence: [local artifact index](./docs/reports/local-artifact-index.json)
- Compiler boundary: [compiler-data-boundary.md](./docs/architecture/compiler-data-boundary.md)
- Evidence semantics: [evidence-core.md](./docs/architecture/evidence-core.md)
- Program evolution kernel:
  [evolvable-pretraining-programs.md](./docs/architecture/evolvable-pretraining-programs.md)
- Public research surface: https://research.wehub.us/tabu-lab/

QueryBase remains the Unit-silent architectural anchor. TabUC and TabURC remain
`design_open`; they are not current training targets and cannot inherit TabUR
checkpoints or evidence.

<!-- seed: If another model family becomes the active experiment, replace this
current-focus surface in place. Do not append a second competing default. -->

## Repository layout

- `src/tabu_lab/` — contracts, runtime, registry, generators, and experiment code
- `specs/models/` — public ModelSpecs
- `tests/contract/` — model and evidence boundaries
- `experiments/` — preregistrations and experiment ledger
- `docs/reports/` — historical local findings and artifact identities
- `site/public/` — public projection; not an evidence authority
