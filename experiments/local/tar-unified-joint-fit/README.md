# Eight-table unified shared fit

This portable recipe uses eight frozen 96-row tables (76 train / 20 reserved),
a shared Small-128 model, unified constant-weight encoding, and one update per
table per shuffled round. All fitting metrics cover the training rows only.

```bash
uv run tabu-lab tar joint-fit \
  --preregistration experiments/local/tar-unified-joint-fit/preregistration.yaml \
  --device cpu --smoke --output-root runs/eight-table-smoke
```

For the full 1024-round recipe, select `--device cuda:0` and omit `--smoke` after
resource preflight. The historical provenance path has been normalized in this
portable preregistration; its digest and the integrated source identity differ
from historical runs. Historical results are documented separately, not
re-issued by this recipe. Do not bypass checkpoint identity checks.
