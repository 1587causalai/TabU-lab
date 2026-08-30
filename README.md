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

## Stage 2: bounded composability

The next gate checks whether the existing tokenizer, dynamics, and readout
alternatives can be substituted one axis at a time while the public forward
interface stays fixed and model identity changes honestly. It also checks that
the model registry can add a namespaced builder without replacing the protected
TabUBase anchor.

See [`docs/architecture/tabubase-composability.md`](./docs/architecture/tabubase-composability.md)
for the reader-facing boundary. This is an architecture-evolvability check, not
a fitting or prediction-quality result.

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
