# TabU-lab

**Learning from tables. Building the evidence in the open.**

TabU-lab develops tabular foundation models and the open research software needed
to understand how they learn. Our central question is whether reconstructing
available table evidence can teach a model the relationships specific to each
unit (a row or entity), and whether those representations transfer to new tables.

[Research website](https://research.wehub.us/tabu-lab/) ·
[Research and support brief](docs/research-support.md) ·
[Try the implementation](#try-the-implementation) ·
[Evidence](#what-exists-today) ·
[中文](https://research.wehub.us/tabu-lab/zh/)

## Why this research

Tables bring together different variable types, incomplete observations, and
relationships that can change across datasets. We investigate how a shared model
can organize the evidence relevant to each unit and reconstruct table values.
Prediction is one downstream use of that learned structure.

The long-term aim is useful, transferable tabular models. The public contribution
also includes inspectable implementations, training recipes, evaluation protocols,
and documented limitations that other researchers can reuse as the work develops.
Inspired by [Marin](https://github.com/marin-community/marin), we make the research
process part of the deliverable.

## What exists today

Status as of **2026-09-16**: active experimental development. The repository contains
implementation checks and exploratory training reports; broad unseen-table
generalization remains an open research question.

| Research asset | What a reader can inspect | What the evidence supports |
| --- | --- | --- |
| Table-restoration implementation | [Five-step guide](docs/tutorials/table-restoration.md), [implementation review](docs/reviews/restoration-five-step-20260915/README.md) | Typed value encoding, contextual representations, Unit-based geometry, and LL/NW restoration; bounded implementation checks |
| V5.3 reference implementation | [Design and Python entry point](docs/design/restoration-v53.md), [CPU example](examples/restoration_v53_smoke.py) | Affine numeric encoding and column-shared LL, with local correctness checks; a separate API from the historical training CLI |
| V5.3 curriculum runner | [Protocol and runnable fixture](docs/tutorials/v53-curriculum.md) | Explicit stage questions, frozen probes, train/validation/test isolation, budgets, atomic checkpoints, and exact local resume; single-process FP64 |
| Reusable execution work | [Prepared-execution review](docs/reviews/restoration-prepared-20260916/README.md) | Scoped equivalence and timing checks, with source, configuration, and limitations recorded |
| Historical TabU-TAR experiments | [120-table shared-fit report and curves](docs/research/tar-shared-fit-20260907/README.md) | Joint fitting on fixed training tables under its own recipe; these are TAR results, not restoration or unseen-table results |
| Reproducible research tools | [Program manifests](docs/architecture/evolvable-pretraining-programs.md), [evaluation protocol](docs/architecture/real-evaluation-default-protocol.md), [catalog](catalog.json) | Versioned model/data/recipe identities and explicit evaluation boundaries |

These records are `local_unissued` where stated. The catalog records no formal
receipts or accepted capability claims. Each linked review describes its dated
source and scope; its test totals are not a current whole-repository certification.

## Try the implementation

Use Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/1587causalai/TabU-lab.git
cd TabU-lab
uv sync --frozen --extra dev
uv run tabu-lab restoration inspect
uv run tabu-lab restoration verify
```

The verifier uses a reduced CPU reference model and returns JSON with check
outcomes and source identity. It checks implementation behavior, including
forward/backward paths and checkpoint continuation. It does not launch a training
campaign. See the [restoration guide](docs/tutorials/table-restoration.md) for
the Python API, check scope, and optional device probes.

The separate TAR implementation is selected explicitly as `tabu.tar`; start
with its [guide](docs/tutorials/tabu-tar.md) and
[design snapshot](docs/design/README.md). The no-argument model factory retains
`tabu.v2.tabur` for compatibility. [Historical runtime documentation](docs/history/compatibility-runtime.md)
preserves those model contracts and formulas. [MAINLINE.yaml](MAINLINE.yaml)
continues to select the query-family pretraining program; it does not select the
restoration model or change the identity of previous experiments.

## What additional support would make possible

We organize potential collaborations around a scientific question, a bounded
work package, a resource budget, and a public result. The next questions are
which reconstruction choices learn reliably, whether they transfer beyond
training tables, and when additional data or compute changes that answer.

| Support | Research or maintenance work it can enable | Proposed public output |
| --- | --- | --- |
| Research grants or fellowships | Researcher time for controlled comparisons, analysis, and independent reproduction | Methods, comparison reports, reusable protocols, and documented negative results |
| GPU or cloud credits | Matched training and evaluation budgets across model/data configurations | Learning curves, resource measurements, and reproducible recipes |
| Coding tools or API credits | Review, numerical regression checks, documentation, and release maintenance | Tested changes, reproducibility fixes, and contributor documentation |
| Data or research partnerships | Permissioned evaluation data, domain questions, and independent validation | Agreed evaluation reports and reusable adapters where sharing is permitted |

These are candidate work packages, not funded commitments. Budget, milestones,
data permissions, and publication scope are agreed for each collaboration.
The [research and support brief](docs/research-support.md) connects the questions,
existing evidence, next decisions, and possible deliverables.

## Participate

Maintained by [Heyang Gong](https://github.com/1587causalai) / WeHub Research.
Researchers, maintainers, and potential supporters can
[open a public issue](https://github.com/1587causalai/TabU-lab/issues/new) with a
research question, reproduction result, bug, or collaboration outline.
Please keep private data and confidential terms out of public issues.

Code is licensed under [Apache-2.0](LICENSE); dataset and artifact permissions are
recorded separately. Useful contributions include reproducing a published check,
testing an evaluation assumption, improving an adapter, or making an experiment
easier for the next researcher to run.

## Explore the repository

- [Restoration experiments](experiments/local/restoration/README.md) and
  [TAR / historical experiment routes](experiments/README.md)
- [Research reports](docs/reports/README.md) and [review records](docs/reviews/)
- [Compiler and truth boundary](docs/architecture/compiler-data-boundary.md)
- [Evidence semantics](docs/architecture/evidence-core.md)
- [Source](src/tabu_lab/), [model specifications](specs/models/), and [tests](tests/)
