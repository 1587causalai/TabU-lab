# Repository layout

Use this map to find the source of a model, contract, example, or experiment.

| Path | What it contains |
| --- | --- |
| [`src/tabu_lab/`](../../src/tabu_lab/) | The Python package, including the [CLI](../../src/tabu_lab/cli.py) and the versioned [model implementations](../../src/tabu_lab/models/). |
| [`specs/`](../../specs/) and [`schemas/`](../../schemas/) | Public model and program definitions and their schemas. The installed factory resolves the [packaged ModelSpecs](../../src/tabu_lab/specs/models/), whose YAML matches `specs/models/`. [Catalog contract](catalog-projection.md) |
| [`examples/`](../../examples/) and [`tests/`](../../tests/) | Small runnable examples and automated checks. They include the V8 completion smoke and V7/V7.3 configuration fixtures. |
| [`experiments/`](../../experiments/README.md) | Dated recipes, recorded results and reproducibility artifacts. The recipe and report for a run belong together. |
| [`history/imports/`](../../history/imports/) | Frozen donor snapshots and their original paths. Read their provenance before reusing them. [Migration provenance](../../MIGRATION_PROVENANCE.md) |
| [`docs/`](../README.md) | Model designs, operating guides, research questions and readable reports. [Version index](../design/README.md) |
| [`scripts/`](../../scripts/) | Checkout-level helpers for building catalogs and schemas, verification, and version-specific experiments. |
| [`site/`](../../site/README.md) | Checked-in website files and their deployment documentation. The [catalog builder](../../scripts/build_public_catalog.py) generates the repository and site catalog projections. |

[`MAINLINE.yaml`](../../MAINLINE.yaml) names the query-family pretraining program
snapshots. The [model version index](../design/README.md) explains the runtime
defaults and opt-in model paths.

## Lightweight checkout

The V6 archive is about 509 MiB of tracked data. Its content-addressed
`objects/` directory accounts for almost all of that size. Use this checkout
for development that needs the code, examples, tests and archive indexes.

```bash
git clone --filter=blob:none --sparse https://github.com/1587causalai/TabU-lab.git
cd TabU-lab
printf '%s\n' '/*' '!/experiments/history/v6-20260930/objects/' |
  git sparse-checkout set --no-cone --stdin
```

To verify or reconstruct the complete V6 archive, first materialize its objects
with `git sparse-checkout disable`, then follow [the V6 archive guide](../../experiments/history/v6-20260930/README.md).
See [Git sparse-checkout](https://git-scm.com/docs/git-sparse-checkout) for how
the checkout patterns work.
