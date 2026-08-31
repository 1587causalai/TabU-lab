# Evolvable TabUBase / TabUR pretraining programs

Status: implemented as a tiny vertical slice. Every run produced before formal
review remains `local_unissued`; this document does not claim pretraining
quality or capability.

## Identity before execution

A pretraining program is the immutable research snapshot

$$
S=(M,C,G,W,P,O,T,E),
$$

where $M$ is the model contract, $C$ is the typed component graph, $G$ is the
set of immutable generators, $W$ is the world mixture, $P$ is the sampling
policy, $O$ is the objective bundle, $T$ is the training recipe, and $E$ is the
evaluation protocol.

The canonical manifests live only under `specs/evolution/`:

- `nodes/` versions each independently changing choice;
- `edges/` records only tested compatibility;
- `programs/` composes exact node references into research snapshots;
- `manifest-lock.json` binds every `(id, version)` to one semantic hash.

Changing semantic content under an existing `(id, version)` fails validation.
A prose-only `description` change is outside the semantic hash. `catalog.json`
and the public catalog are generated projections; they cannot override a
manifest. `MAINLINE.yaml` points to complete programs rather than bare model
contracts.

The repository-wide lock enforces immutability but is not part of every run's
identity. A resolved snapshot hashes only its selected program and transitive
manifest closure. Its code identity likewise binds that closure plus the shared
execution kernel. Adding an unrelated candidate therefore changes the catalog
lock without invalidating existing TabUBase or TabUR snapshots/checkpoints.

## Stable external ABI

Only three external boundaries are intended to remain stable across model and
data redesigns:

- `EvidenceEpisode` carries observed context, query coordinates, masks, and
  support. It never contains query truth visible to `forward`.
- `PredictionBundle` carries predictions, support/status, and bounded trace.
- `ProgramRunReceipt` binds the resolved snapshot, code/data identities, seeds,
  checkpoint artifacts, policy state, lane, and status.

`TruthSidecar` enters only the objective/loss boundary. Components declare
`truth_visible: false`, and the component graph has no truth input port.

Everything inside this ABI is versioned: model mathematics, components and
ports, generators, mixtures, sampling policies, objectives, recipes,
evaluation, and state projections.

## Open typed component graph

`ComponentGraphSpec` is compiled as a typed DAG. Validation fails closed on:

- dangling references;
- dependency or component cycles;
- incompatible versioned port interfaces;
- duplicated or unbound required input ports;
- unconsumed component outputs;
- unverified compatibility edges.

The current tokenizer, axis-source, dynamics, geometry/readout, terminal, and
supervised adapter registries are represented through source-bound adapters.
This preserves old experiments while allowing a future component to be added as
a new node and graph version.

TabUBase and TabUR are sibling graphs. They share the stable ABI and selected
data/evaluation nodes, but neither program references the other's checkpoint.
The existing matched-state conformance test verifies that TabUR with
$\gamma=0$ has the same whole-forward result as TabUBase.

## Impact semantics

`program impact` compares two resolved snapshots. Its result classifies each
affected object as `unchanged`, `reuse_exact`, `rescore`, `rerun_inference`,
`retrain`, `warm_start_available`, or `blocked`.

The default is incompatibility. A reuse exception exists only when a versioned
edge and its verifier are present. Current golden behavior is:

| Change | Minimum action |
| --- | --- |
| ModelContract or ComponentGraph | retrain the selecting model lane; warm start is blocked unless explicitly projected |
| Generator, WorldMixture, or SamplingPolicy | retrain only programs selecting the new data snapshot; unchanged model code remains reusable |
| ObjectiveBundle or TrainingRecipe | new run; exact resume is invalid |
| EvaluationProtocol with compatible stored predictions | reuse checkpoint and predictions, then rescore |
| EvaluationProtocol requiring another prediction form | reuse checkpoint, rerun inference, then score |
| Description-only edit | semantic identity unchanged; no compute action |

Three checked-in exercise programs keep these rules executable without making
their proposed mathematics authoritative:

- `tabu.pretraining.query-base-math-exercise@1.1.0-exercise` adds a deliberately
  non-executable placeholder `ModelContract` and graph. It must report retrain
  and blocked warm start.
- `tabu.pretraining.query-base-generator-v3@1.1.0-exercise` selects a new
  immutable generator/mixture. It must report retrain plus an explicit
  weights-only identity projection.
- `tabu.pretraining.query-base-component-adapter@1.1.0-exercise` inserts a
  component graph node. It must preserve data and evaluation identities while
  retraining the affected graph.

An evaluation-only exercise additionally guards the `rescore`, not `retrain`,
path.

The four generated reports live under `docs/reports/evolution-impact/`. They are
query projections and deliberately carry no evidence or claim status.

## Grow, freeze, and evidence

The grow lane runs smoke tests, pilots, and candidate comparisons. Its receipts
are always `local_unissued`. A grow artifact cannot be edited or relabeled into
evidence.

`program freeze` converts one fully resolved grow snapshot into a separate
evidence-lane identity with all hashes fixed. `program run --lane evidence`
accepts only that frozen file and requires the scoped Git root to be clean.
Successful execution produces `evidence_candidate_unreviewed`, not an accepted
claim. A dirty or nested source identity fails before training.

Formal synthetic fit, real scratch, frozen ICL, and fine-tune transfer remain
independent evidence receipts. No implementation gate forces them into one
serial pipeline.

## Checkpoint and restart contract

A program checkpoint consists of a trainer `.safetensors` file and a required
`.program.json` sidecar. Together they contain:

- model and objective tensors;
- optimizer and scheduler state;
- update/world cursor;
- complete sampling-policy state;
- Python, NumPy, Torch, named-generator, CUDA, and MPS RNG states where present;
- resolved snapshot, run identity, lane, status, target steps, and artifact
  hashes.

Exact resume requires all of these identities and states to match. Missing
policy state, changed dependency hash, lane mismatch, scheduler mismatch, or a
weights-only file fails. The tiny conformance test proves that interruption plus
resume reaches byte-identical checkpoint and sidecar bytes to an uninterrupted
run.

Warm start is a different operation. It loads model weights only through a
verified `StateProjection`, discards optimizer/scheduler/policy/RNG continuity,
and always creates a new run identity with no inherited evidence status.
Legacy weights-only files additionally require an explicit
`--warm-start-source-program`; they remain invalid for exact resume.

## CLI

All commands resolve against the scoped repository root (the current directory
by default):

```bash
tabu-lab program validate
tabu-lab program resolve --program tabu.pretraining.query-base@1.0.0
tabu-lab program diff \
  --from-program tabu.pretraining.query-base@1.0.0 \
  --to-program tabu.pretraining.query-base-generator-v3@1.1.0-exercise
tabu-lab program impact \
  --from-program tabu.pretraining.query-base@1.0.0 \
  --to-program tabu.pretraining.query-base-eval-v2@1.1.0-exercise
tabu-lab program freeze \
  --program tabu.pretraining.query-base@1.0.0 \
  --output /new/path/base.frozen.json
tabu-lab program run \
  --lane grow \
  --program tabu.pretraining.query-row@1.0.0 \
  --output-root /new/path/query-row-tiny
```

`program lock` checks the manifest lock. `program lock --write` may append a new
versioned identity but rejects removal or rewriting of every previously locked
identity.

## Current bounded milestone

The implementation closes the evolution mechanism. The selected mainline now
adds a bounded CUDA Grow pilot without rewriting the completed tiny milestone:

- TabUBase and TabUR each produce an independent two-update tiny checkpoint;
- `tabu.training.dgx2-grow-pilot@1.0.0` adds 1500 updates over the complete
  supervised-v2 generator support with checkpoints every 500 updates;
- `tabu.pretraining.query-base@1.1.0` and
  `tabu.pretraining.query-row@1.1.0` remain independent `local_unissued`
  snapshots and change only the training-recipe slot from their 1.0.0 parents;
- fixed, piecewise, and adaptive policies have serializable deterministic
  state;
- `tabu.pretraining.query-base@1.2.0` and
  `tabu.pretraining.query-row@1.2.0` select the immutable v3 mixture and new
  1024-feature-capacity graph versions while keeping the 1500-update recipe
  fixed for a controlled scratch-first comparison;
- v2-to-v3 weights may cross only through model-specific verified scale
  projections; optimizer, scheduler, policy, RNG, and evidence status remain
  forbidden;
- math, generator, component, and evaluation evolution reports have golden
  tests;
- exact resume and warm-start boundaries have executable tests;
- catalog projections contain the versioned evolution graph but remain
  explicitly non-evidentiary.

This pilot is an engineering and learning-dynamics gate, not formal evidence or
full-scale pretraining. Evidence-lane execution still requires a clean source,
frozen split/seed/budget/evaluation identities, and independent review.
