# Compiler and data boundary

The compiler turns a raw table into a model-ready episode without allowing held-out
or target truth to leak into the forward pass.

The boundary is deliberately ordered:

1. `split_dataset` defines a complete, pairwise-disjoint split and declares its fit
   partition.
2. `bind_split_view` binds each partition to the exact dataset and split manifest.
3. Fit-derived artifacts such as `NumericNormalizer`, `Imputer`,
   `CategoricalCodebook`, and `FeatureSelectionManifest` may be learned only from
   the declared fit view. They remain bound to that split definition.
4. `EpisodeCompiler` checks the source view, fit view, recipe, target roles, and any
   typed graph topology before constructing an episode.
5. The result separates `EvidenceEpisode`, which may cross the model-forward
   boundary, from `TruthSidecar`, which remains host-side for scoring.

The compiler fails closed when it receives raw data, a recipe from another view,
statistics from another split, conflicting topology, or forward roles that could
expose unavailable truth. Compilation provenance binds the dataset, split, views,
recipe, topology, and fitted normalizer by content hash.

This layer establishes data isolation and reproducible construction. It does not
show that a model fits synthetic data, predicts real data, performs frozen ICL, or
benefits from fine-tuning.

Run the focused checks with:

```bash
uv run pytest -q \
  tests/contract/test_compiler_binding.py \
  tests/contract/test_compiler_statistics.py \
  tests/contract/test_compiler_topology.py
```
