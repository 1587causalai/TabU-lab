# TabU-lab

An open research lab for tabular foundation models, inspired by
[Marin](https://github.com/marin-community/marin).

The repository's single pretraining-direction pointer is
[`MAINLINE.yaml`](./MAINLINE.yaml).
It selects a complete immutable `ProgramSnapshot`: model contract, component
graph, data mixture and policy, objective, training recipe, and evaluation
protocol. Generated catalogs are query projections, not the source of truth.

## Current research direction: TabU-TAR

The current mathematical and implementation research direction is fourth-generation
**TabU-TAR (Typed Additive Readout)**. Its additive response has independent
Feature and Unit gates. The default realization uses fixed single-axis inducing
slots; the direct, without-inducing realization remains an explicit baseline.
Recommendation-specific two-axis designs are separate explorations.

TAR integration into this branch is in progress. The existing executable factory
and pretraining programs below retain their current identities while that work
is validated. A research-direction change does not reinterpret an old checkpoint.

New TAR validation starts with **Small** for rapid feedback. Standard remains the
reference model design; important findings require separate larger-size checks.
The evaluation ladder is: component correctness; decoupling, extension and growth;
synthetic fitting; real-data prediction; synthetic pretraining with frozen ICL;
and pretrained-versus-scratch real-task fine-tuning. Each stage needs its own
evidence. Local fitting results do not establish pretrained or generalization
capability.

## Existing compatibility runtime: TabU-v2 / TabUR

The executable model-factory default is **TabU-v2 / TabUR** under
`tabu.v2.tabur@0.1.0`. It implements the cell-as-query structural design in
the readonly `TabU-v2` source closure. Historical query-family models remain
available through explicit contract ids and are not silently aliased to v2.
TabU-v2 is still experimental: selecting it as the default is a routing
decision, not an evidence-backed capability claim.

The historical **TabUR** contract under `tabu.query.row@0.2.0` remains an
explicit compatibility model, using
`supervised.label_broadcast.v1` and the default symmetric `anchored` readout.
**TabUBase** under `tabu.query.base@0.1.0` is an independently trainable sibling,
not a prerequisite checkpoint. Both share the Episode/Prediction and evaluation
protocol boundaries while retaining separate model and run identities.
The immediate research question is whether diverse supervised synthetic
pretraining produces useful frozen ICL and then improves real-task fine-tuning.

| Surface | Current status |
| --- | --- |
| TabU-v2 / TabUR contract and runtime | default via `tabu.v2.tabur@0.1.0` |
| historical TabUR compatibility runtime | explicit `tabu.query.row@0.2.0` |
| Evolvable program kernel | implemented with immutable manifests, typed DAG validation, impact analysis, freeze, exact resume, and explicit warm start |
| Broad supervised synthetic prior v3 | candidate implementation selectable through versioned Generator/Mixture manifests |
| v3 long-run pretraining | activated as scratch-first Grow snapshots `tabu.pretraining.query-{base,row}@1.2.0`; execution remains `local_unissued` until a run receipt exists |
| frozen ICL for the 0.2/v3 lane | `not_run` |
| real-task pretrained-vs-scratch fine-tuning for the 0.2/v3 lane | `not_run` |
| formal evidence / accepted capability claim | none |

Results from Axis-B TabUBase, `tabu.query.row@0.1.0`, or synthetic priors v1/v2
remain immutable historical evidence. They do not transfer to the current model,
checkpoint identity, or capability claim.

## Existing TabU-v2 runtime contract

The mathematical authority for the existing TabU-v2 runtime is its source closure.
The historical Axis-C TabUR source binds only the explicit legacy contract. Runtime preserves the
same five-step boundary:

1. **Compile evidence.** An `EvidenceEpisode` contains only model-visible table
   evidence. Query truth is held outside the model in `TruthSidecar`.
2. **Tokenize query cells.** Visible values, roles, masks, and null state produce
   typed initial cell states $h^{(0)}_{ra}$; target truth is absent.
3. **Run typed dynamics.** The visible-only source mask updates one extended
   carrier of shape $(N+K)\times(M+K)\times d$ by column OMAB and then row OMAB.
   Cell Query, Unit Query, Feature Query, and Null slots are receiver-only.
4. **Read out coordinates.** With $c_{ra}=h^{(L)}_{ra}$, canonical TabU-v2 uses

   $$
   A_{ra}=W+\lambda_F(F_a+\lambda_U U_r),
   \qquad z_{ra}=A_{ra}c_{ra}.
   $$

   The default regime sets $\lambda_F=\lambda_U=0$, so the response field is
   shared across datasets. Feature- and Unit-adjusted regimes are explicit
   ablations. $W$ is a shared response parameter, not a semantic Unit; $F_a$
   and $U_r$ are address-indexed views of the final carrier.
5. **Score externally.** The typed terminal returns a `PredictionBundle`; the
   evaluator alone pairs it with `TruthSidecar`. The canonical numeric loss
   coordinate is context-standardized. `numeric_raw_prediction` is an auxiliary
   inverse projection, not the Step-5 training target.

The construction defaults to $K=\texttt{matched\_slots}$ (an explicit `k` overrides it) and keeps the numeric
terminal in context-standardized coordinates; `numeric_raw_prediction` is an
auxiliary inverse projection. See the [TabU-v2 ModelSpec](./specs/models/tabu.v2.tabur.yaml)
and the historical [query runtime mapping](./docs/architecture/query-model-runtime-mapping.md).

## Existing runtime and pretraining defaults

| Decision | Default |
| --- | --- |
| contract | `tabu.v2.tabur@0.1.0` |
| historical compatibility contract | `tabu.query.row@0.2.0` (explicit only) |
| response regime | shared across datasets, `lambda_F=0`, `lambda_U=0` |
| numeric terminal | `local_linear` in context-standardized coordinates |
| nominal tokenizer | `source_scoped_frozen_codebook.v2` |
| $K$ | `matched_slots=4` |
| pretraining direction | `MAINLINE.yaml` remains the existing query-base/query-row program pointer |
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

Build the existing compatibility default model explicitly (the model id may be omitted):

```python
from tabu_lab.models import build_model
from tabu_lab.models.types import ReferenceConfig

model = build_model(
    config=ReferenceConfig(
        matched_slots=4,
        max_features=1024,
    ),
)
# model.model_id == "tabu.v2.tabur"

# Pair v2 predictions with truth in the same numeric coordinate:
from tabu_lab.training import MixedObjective
objective = MixedObjective(numeric_target_coordinate="context_standardized")

# Historical TabUR remains an explicit compatibility choice:
legacy = build_model(
    "tabu.query.row",
    profile="supervised.label_broadcast.v1",
    row_token_count=4,
    row_readout_mode="anchored",
    anchored_gamma_initial=1.0e-2,
)
```

The existing `scripts/run_tabur_r5_pretraining.py` is bound to synthetic prior v2
(not the TabU-v2 model family). Those historical training programs remain explicitly
pinned; changing the model-factory default does not rewrite their identities. Prior-v3 execution
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

- Default model/runtime authority: [TabU-v2 ModelSpec](./specs/models/tabu.v2.tabur.yaml)
  and the historical [query runtime mapping](./docs/architecture/query-model-runtime-mapping.md)
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
