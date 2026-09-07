# Diverse 120-table shared fitting recipe

The frozen `corpus/data` tables and `corpus/preregistration.yaml` preserve the
historical 120-table recipe. Dataset bytes, split identities and the original
corpus manifest are unchanged. This integration is a new code identity.

```bash
uv run tabu-lab tar joint-fit \
  --preregistration experiments/local/tar-diverse-120-fit/corpus/preregistration.yaml \
  --device cuda:0 --stop-after-round 1 --output-root runs/diverse-pilot
```

The full recipe allows 1024 rounds with a four-hour soft budget checked at round
boundaries, followed by finalization. Choose an explicit segment bound. CPU is
only available with `--smoke`, which uses a reduced model and two rounds.
Resume requires the identical preregistration, source, model, and optimizer
configuration; do not use a checkpoint from an older source snapshot.

For MPS, prepare a separate preregistration selecting
`numerical_backend: experimental_fp32`, run PyTorch 2.13.0, and launch with
`PYTORCH_ENABLE_MPS_FALLBACK=0 PYTORCH_MPS_FAST_MATH=0`. Do not mutate the
historical recipe to describe a new run.

## Corpus provenance and regeneration

The MIT-licensed generator snapshot is in `corpus/generator-source`, with its
license and original requirements retained. Content digests are recorded in
`corpus/manifest.json`; see `MIGRATION_PROVENANCE.md` at the repository root.

```bash
uv run --python 3.11 --with numpy==2.4.6 --with scikit-learn==1.9.0 \
  tabu-lab tar freeze-diverse-corpus \
  --generator-root experiments/local/tar-diverse-120-fit/corpus/generator-source \
  --output-root runs/regenerated-corpus --rows 256 --seed 20260907
```

The historical generator runtime was Python 3.11.14 / NumPy 2.4.6 /
scikit-learn 1.9.0. Compare regenerated hashes before assuming byte equivalence.
The public fixture includes the 120 training-ready tables, request census,
rejected-attempt metadata and structural diagnostics. Large raw world/episode
exports are represented by hashes in the historical manifest and can be
regenerated; they are not included in this fixture directory.

57 targets are numeric, 23 binary, 20 ordinal and 20 categorical. All tables have
204 train / 52 reserved rows. Source families and seeds are not counted as 120
independent laws; pure-noise and unsupported-category controls are retained.

See the [captured results and next steps](../../../docs/research/tar-shared-fit-20260907/README.md).
