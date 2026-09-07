# Integration validation

Environment: frozen development lock, Python 3.11.14, PyTorch 2.13.0, CPU arm64.

- `uv sync --frozen --extra dev` completed.
- `uv run pytest -q`: **542 passed, 9 skipped**. Skips are optional sklearn /
  xgboost-dependent legacy checks; two expected observer fallback warnings.
- Focused TAR/model/fit/corpus suite: **96 passed**.
- Ruff checks over all changed authored Python modules, tests and plotting code:
  passed. Repository-wide Ruff is not clean: retained historical import trees,
  one unchanged registry file and the byte-preserved third-party generator have
  existing/style findings. These are not reported as a passing global lint run.
- The canonical eight-table CPU smoke CLI below completed 2 rounds / 16 updates.
- `uv run python scripts/verify_site.py`: passed; existing site claim boundaries
  remain unchanged.
- `uv build`: sdist and wheel built. The source distribution includes all 120
  typed table fixtures, the generator MIT license and the result figure.
- `git diff --check`: passed.

```bash
uv run tabu-lab tar joint-fit \
  --preregistration experiments/local/tar-unified-joint-fit/preregistration.yaml \
  --device cpu --smoke --output-root runs/eight-table-smoke
```

The displayed output directory is a portable equivalent of the temporary local
validation directory. The smoke used the working integration source and remains
local_unissued; it is not a full fitting result or a GPU verification run.

## Independent review

A separate read-only reviewer found no blocking compatibility, evidence-boundary
or publication-safety issue. It independently checked:

- all 120 table hashes and the corpus manifest binding;
- seven summary evaluation points, improvement counts 118/115/91, and six
  exported content checksums;
- 11 targeted tests;
- 174 implementation/data/export paths for private paths and credentials;
- preservation of legacy defaults, current mainline CLI behavior and strict
  source/optimizer/checkpoint resume checks.

Final packaging additions retain the same source/data bytes; the plotting
script received formatting and an explicit zip-length check. Historical source
identities remain distinct from this integration. No public maturity status or
accepted claim was advanced.

## Completed 768-round update

The earlier 416-round snapshot is superseded by the completed 512-round segment
plus 256 continuation rounds. All 42 artifact hashes in each of the 512/768
segments were reverified. The parent final model state exactly matches the
continuation initial state; all-table metrics share the 768-round final state.
Only result exports, figures and documentation changed; implementation tests
above describe the unchanged integration code.
