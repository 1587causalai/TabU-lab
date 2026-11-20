# TabU-lab documentation

## Start here

- [Project overview and quick start](../README.md) introduces the research and
  the runnable code on `main`. Use the [V8 completion guide](../src/tabu_lab/models/completion_v8/README.md)
  for masked-cell completion and the opt-in V8.2 supervised settings.
- [Research questions](research-questions/README.md) records what each study
  has answered and what remains open. The V11 prototype has completed initial
  training checks on synthetic classification and regression tables.
  [V11 research progress](../README.md#v11-research-roadmap)
- [Version and design index](design/README.md) maps each model family to its
  implementation, design, or dated evidence. [Research reports](reports/README.md)
  collects the published results and historical records.

## Follow an implementation or evaluation

- [Repository layout](architecture/README.md) shows where the package, contracts,
  experiments and generated files live and how to skip the large V6 object archive.

- [V7 and V7.3 restoration](tutorials/v7-restoration-modes.md) covers masking
  modes, version selection, and a synthetic fixture.
  [Earlier implementations](design/README.md#implementations-and-archives)
  includes the original restoration reference, TAR, and compatibility runtimes.
- [Full-context query-family evaluation](architecture/real-evaluation-default-protocol.md)
  defines its split, support, and metrics. [Tabular model and benchmark references](references/tabular-foundation-models/README.md)
  links the external protocols and model releases used in comparisons.
- [Program manifests](architecture/evolvable-pretraining-programs.md),
  [the data boundary](architecture/compiler-data-boundary.md), and [evidence semantics](architecture/evidence-core.md)
  explain how to identify a run and interpret its receipts.

For collaboration and resourcing, see the [research and support brief](research-support.md).
