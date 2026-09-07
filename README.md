# TabU-lab

An open research lab for tabular foundation models, inspired by
[Marin](https://github.com/marin-community/marin).

## Current research: TabU-TAR

**TabU-TAR (Typed Additive Readout)** is the current mathematical and implementation
research direction. Its additive response has independent Feature and Unit gates.
The reference realization uses fixed single-axis inducing slots; the direct,
without-inducing realization remains an explicit baseline.

The research retains two training routes:

| Route | Research role |
| --- | --- |
| Single episode | One episode per optimizer update; a continuing research route and the basis of the completed 768-round shared-fit result. |
| Batch episodes | Multiple episodes per optimizer update; a continuing route for iteration on batch construction, gradient aggregation, optimization and parallel execution. |

Both routes retain their implementations, configurations and experiment histories.
A particular batch configuration's result does not retire either route. Compare
training episodes, optimizer updates and wall-clock time separately, with explicit
initial weights, data, evaluation masks and source identities.

The published [120-table shared-fit report](docs/research/tar-shared-fit-20260907/README.md)
records **768 rounds / 92,160 updates** with one shared Small-128 model. It also
records the separate eight-table experimental FP32/MPS result. These are
`local_unissued` training-fit results; unseen-world and real-table transfer need
their own evaluations.

## Start here

- [TAR implementation guide](docs/tutorials/tabu-tar.md): model configuration,
  typed episodes, training semantics and checkpoints.
- [Experiment routes](experiments/README.md): single-episode and batch-episode
  research, available runners and per-run evidence.
- [Mathematical design snapshot](docs/design/README.md): equations, figures and
  the distinction between the reference design and configured experiments.
- [120-table recipe](experiments/local/tar-diverse-120-fit/README.md) and
  [eight-table recipe](experiments/local/tar-unified-joint-fit/README.md): frozen
  inputs, explicit preregistration and source-bound execution.

```bash
uv sync --frozen --extra dev
uv run tabu-lab tar sizes
uv run tabu-lab tar inspect --size small-128
uv run tabu-lab tar verify                         # named Small
```

The model API selects TAR explicitly:

```python
from dataclasses import replace
from tabu_lab.models import build_model
from tabu_lab.tar_sizes import config_for_size

config = replace(config_for_size("small-128"),
                 value_encoding="unified_constant_weight")
model = build_model("tabu.tar", config=config)
```

This constructs the 1,267,136-parameter configuration used by the shared-fit
recipe. It does not load a trained checkpoint. `small-128` alone uses the legacy
encoder default and has a different parameter count; the encoding is part of the
configuration identity.

## Available execution and compatibility

The packaged `tar joint-fit` runner explicitly uses one episode per update.
The packaged `TARTrainer` can accumulate multiple episodes serially before one
update. A batch count alone does not select parallel episode execution. The
separate parallel batch-episode research implementation is not yet integrated
into this checkout; its runner, configuration and evidence must be integrated
together. See the [training-route details](experiments/README.md#tar-training-routes).

The no-argument `build_model()` remains the compatibility factory for
`tabu.v2.tabur`; new TAR examples use `build_model("tabu.tar", ...)`.
[`MAINLINE.yaml`](MAINLINE.yaml) selects the existing query-row/query-base
pretraining programs. It is scoped to that program system and does not select
the TAR model or either TAR training route.

Older model names, runtime formulas, program commands and checkpoints are
preserved in the [historical compatibility guide](docs/history/compatibility-runtime.md).
They retain their identities; current research does not reinterpret them.

## Evidence and evaluation

Record implementation checks, fitting, throughput, frozen ICL and real-task
transfer as separate results. Fit metrics on training rows do not establish
held-out capability. The [shared-fit report](docs/research/tar-shared-fit-20260907/README.md)
identifies the rows and masks used, historical source identities, and next
research questions.

Experiments bind their configuration, seeds, data and checkpoints. Exact resume
requires the original source and optimizer identity. Public catalogs and the
[research website](https://research.wehub.us/tabu-lab/) are projections of
recorded evidence.

## Repository layout

- `src/tabu_lab/` — model implementations, runtime, generators and experiment code
- `specs/models/` — model contracts
- `tests/` — implementation and evidence-boundary checks
- `experiments/` — preregistrations, frozen inputs and experiment routes
- `docs/research/` — bounded results and research interpretation
- `docs/history/` — preserved historical and compatibility guidance
- `site/public/` — public projection
