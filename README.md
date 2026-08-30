# TabU-lab

An open research lab for tabular foundation models, inspired by [Marin](https://github.com/marin-community/marin).

## Current model anchor

The first executable contract in this repository is `tabu.cell.base@0.2.0`, also
called **TabUBase**. It treats each table cell as one Unit and exposes two explicit
forward profiles:

- `completion.artificial_mask.v1` for artificial-mask completion;
- `supervised.label_broadcast.v1` for a single declared response column.

This anchor establishes a buildable, identity-bound, truth-free reference forward.
Numeric terminals and the canonical `numeric` prediction use the Step-1
context-standardized scale; `numeric_raw_prediction` is an explicitly named
inverse projection for inference display, not the Step-5 loss value.
It does **not** claim useful fitting, real-data prediction, frozen ICL, fine-tuning
lift, or foundation-model evidence. Those are separate evaluation gates.

The public and packaged ModelSpec files are byte-identical:

- `specs/models/tabu.cell.base.yaml`
- `src/tabu_lab/specs/models/tabu.cell.base.yaml`

The ModelSpec binds both the TeX entrypoint hash and its recursive semantic source
tree through the public and packaged `model-factory-source-manifest.json`. The
first-public-anchor authority decision is recorded in
`docs/decisions/tabubase-0.2.0-source-authority.md`.

Run the focused contract gate with:

```bash
uv sync --frozen --extra dev
uv run pytest
```

## Compiler and data boundary

Raw tables do not go directly into a model. The compiler first binds a complete,
disjoint split; fits statistics, imputation, categorical vocabularies, and feature
selection only on the declared fit partition; then emits separate model-visible
evidence and host-side truth. Typed row topology is preserved when present.

See [`docs/architecture/compiler-data-boundary.md`](./docs/architecture/compiler-data-boundary.md).
This is leakage prevention and deterministic episode construction, not a model-
quality result.

## Evidence core

The evidence layer defines strict, content-addressed schemas for preregistrations,
run receipts, claim ledgers, and source identities. Receipt files are immutable and
self-verifying; public evidence rejects local paths and likely secrets. A source is
`formal` only when its review and immutable source bindings close, otherwise it
remains `local_unissued` with explicit reasons.

See [`docs/architecture/evidence-core.md`](./docs/architecture/evidence-core.md).
These contracts make evidence auditable; they do not issue a receipt or accept a
scientific claim by themselves.

## YAML mathematics and TeX projection

`ModelSpec` may carry an optional, typed `mathematics` block: named notation,
ordered equations, and falsifiable invariants. `render_model_tex` turns that block
into deterministic standalone TeX while escaping prose and preserving authored
formula LaTeX.

See [`docs/architecture/yaml-mathematics.md`](./docs/architecture/yaml-mathematics.md).
The existing `tabu.cell.base@0.2.0` YAML is intentionally unchanged; adding
mathematics to that immutable contract requires a reviewed version decision.

## Current catalog projection

The bounded catalog indexes only canonical ModelSpecs already consolidated on the
current branch, checks public/package byte parity, and produces deterministic JSON
and HTML projections. Its public boundary explicitly remains at zero formal
receipts and zero accepted claims.

See [`docs/architecture/catalog-projection.md`](./docs/architecture/catalog-projection.md).
The larger donor catalog still depends on evaluation and verification contracts
that are outside this consolidation sequence.

## Stage 2: bounded composability

The next gate checks whether the existing tokenizer, dynamics, and readout
alternatives can be substituted one axis at a time while the public forward
interface stays fixed and model identity changes honestly. It also checks that
the model registry can add a namespaced builder without replacing the protected
TabUBase anchor.

See [`docs/architecture/tabubase-composability.md`](./docs/architecture/tabubase-composability.md)
for the reader-facing boundary. This is an architecture-evolvability check, not
a fitting or prediction-quality result.

## Stage 3: synthetic-data basic fitting

The bounded synthetic gate asks whether a fresh TabUBase can reduce masked
response loss on a fixed linear synthetic world. It keeps response truth in an
evaluation sidecar, uses a separate validation world, and reports only
`local_unissued` diagnostic evidence. See
[`docs/architecture/synthetic-fit-gate.md`](./docs/architecture/synthetic-fit-gate.md)
and run `uv run python scripts/run_tabubase_synthetic_fit.py`.

## Synthetic pretraining implementation history

The repository now also preserves the latest local implementation for expanded
synthetic-world generation, response readout, scale-transfer training, and their
versioned YAML/schema inputs. The accompanying reports record the development
history exactly as `local_unissued` material; they are not formal receipts,
accepted claims, or evidence that synthetic pretraining improves a real task.

See
[`docs/architecture/tabubase-expanded-synthetic-pretraining-data.md`](./docs/architecture/tabubase-expanded-synthetic-pretraining-data.md).

## Real-data evaluation implementation history

The latest local real-data stack is preserved for optimizer-free frozen ICL,
pinned OpenML panels, exact-split classical baselines, and paired TabUBase
fine-tuning. Its YAML registrations and reports remain `local_unissued` inputs
and historical observations. Checked-in code and reports do not by themselves
establish a benchmark, a formal receipt, a foundation-model claim, or a causal
effect from synthetic pretraining.

The eight-dataset cached OpenML panel records pinned source hashes and historical
cache paths; those paths are provenance references, not portable bundled data.

## Website

- Public entrance: https://research.wehub.us/tabu-lab/
- Chinese entrance: https://research.wehub.us/tabu-lab/zh/
- Static source: [`site/public/`](./site/public/)
- Machine-readable project card: https://research.wehub.us/tabu-lab/agent.json

The website is a public research surface, not a source of model or benchmark evidence. Experiment receipts remain under `experiments/` and consolidated findings under `docs/reports/`.

## Core values

- **Open development**: every experiment, config, curve and failed result is recorded as it happens.
- **Reproducible at small scale**: fixed splits, fixed seeds, explicit compute budget; every run produces a receipt.
- **Falsifiable gates**: each training step has a pass/kill signal written down before the run.

## Layout

- `src/tabu_lab/` — typed contracts, model kernel, registry, and reference implementation
- `specs/models/` — public ModelSpec source
- `tests/contract/` — correctness and boundary tests for the model anchor
- `experiments/` — future preregistered evaluation work; not evidence by itself
- `docs/reports/` — reviewed findings and retrospectives when evidence exists

## Related

- TabPFN v2 (Prior Labs): https://github.com/PriorLabs/TabPFN
- Marin: https://github.com/marin-community/marin
- WeHub Research: https://research.wehub.us/
