# TabU-lab

TabU-lab is WeHub Research's open lab for tabular foundation models. We study
whether a model can learn how each row or entity responds from the values
observed in a table, and whether it can use those representations to fill in
missing values and make predictions on new tables.

[Documentation](docs/README.md) ·
[Research questions](docs/research-questions/README.md) ·
[Research website](https://research.wehub.us/tabu-lab/) ·
[中文](https://research.wehub.us/tabu-lab/zh/)

## What exists today

- **Code you can run.** V8.2 is the most recent model integration on `main`.
  The V8 completion API fills masked cells. V8.2 adds opt-in attention and
  memory settings for supervised prediction. [Implementation and examples](src/tabu_lab/models/completion_v8/README.md)
- **Published experiments.** The October V9 and V10.1 reports examine
  whether better reconstruction helps on new observations, how predictive
  models compare on matched tasks, and where training time goes. [Version and evidence index](docs/design/README.md#published-research)
- **A public checkpoint.** Kumo Small + CellRidge-1024 is our first TabArena
  submission and a V10.4 milestone. Classification and regression inference
  weights are available. TabArena closed the submission after reviewing the
  local Lite results against Kumo Small. [Weights release](https://github.com/1587causalai/kumo-small-cellridge-1024/releases/tag/v1.0) ·
  [TabArena PR #735](https://github.com/autogluon/tabarena/pull/735)

## Try the implementation

With Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/), inspect the
reference restoration configuration:

```bash
git clone https://github.com/1587causalai/TabU-lab.git
cd TabU-lab
uv sync --frozen --extra dev
uv run tabu-lab restoration inspect
```

For a small example with synthetic data, use the [V8 completion example](examples/completion_v8_smoke.py)
or follow the [V7 restoration guide](docs/tutorials/v7-restoration-modes.md).
The [version index](docs/design/README.md#implementations-and-archives) links older
runtimes, checkpoint contracts, and experiment records.

## V11 research roadmap

We plan to study whether learned representations make unit-specific
relationships easier to predict with a simple linear readout. The proposed
design fits a shared rule from observed rows, then uses residuals from similar
units to estimate a local correction. The research will test accuracy after
decoding predictions back to table values and measure cost as tables grow.
This is a design-stage research question.

## Participate

Maintained by Heyang Gong and WeHub Research. The code is
[Apache-2.0](LICENSE). Model weights and data have their own terms.
[Issues](https://github.com/1587causalai/TabU-lab/issues) ·
[Research and support brief](docs/research-support.md)
