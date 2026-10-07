# V7.3, asinh and row dual-stream integration

Date: 2026-10-08. Base: `af7e5f519df781dd42e402359e280295b28edf51`.

This integrates the reviewed V7/V7.3 version boundary, current asinh numeric
coordinates, and the explicitly selected row dual-stream encoder. It preserves
the recorded meaning of historical V7 configurations and keeps `phi_lift` as the
default value encoder. It does not select an adopted checkpoint, change a running
experiment, or publish a new standalone inference package.

## Included behavior

- Explicit `v7` and `v7.3` factories, versioned Unit source/initialization behavior,
  zero-preserving coupling, optional balanced visible reconstruction, and
  fixed-candidate classification probabilities.
- `standard_asinh_v1` for newly constructed V7.3 configurations, with stable
  extreme-value encoding/decoding. Explicit legacy and softlog configurations
  remain available and retain their checkpoint identities.
- Opt-in `row_dual_stream`: four independent row-local attention/FFN coupling
  blocks, 64+64 streams, 128-dimensional Support responses for analytical LL,
  simultaneous Query assembly and whole-row inverse/mean readback.
- Explicit weights-only migration between encoders. Generic initialization
  rejects an implicit encoder switch; strict continuation still checks complete
  model, source and runtime identity.
- Native checkpoint inference following the recorded ModelSpec, including
  guarded historical overlays. The separately frozen standalone package is a
  different artifact and is not silently updated.

See the [usage guide](../../tutorials/v7-restoration-modes.md) and
[bounded CPU example](../../../examples/v7-restoration/row-dual-stream.yaml).

## Review provenance

The model implementation is reused from existing independently reviewed work,
rather than rewritten for this integration. The V7.3 changes originate in the
reviewed `4f7f6193` snapshot; the subsequent asinh and dual-stream source identities
are pinned by the 2026-10-07 review baseline and final acceptance receipt.
All 21 changed product files match one of those reviewed sources exactly.
[Verification](verification.json) records the file-by-file hashes.

The [prior review evidence](prior-review-evidence.json) preserves source digests,
the independent CUDA acceptance results, and the identity of 17 repaired test
cases. Its 806 selected regression cases passed across an initial run and a
17-case environment repair; that historical result is not presented as one
all-green full-suite execution. CPU/FP64 and actual MPS/FP32 acceptance covered
row invertibility, cross-row isolation, simultaneous recovery, finite-difference
gradients, multi-round backpropagation and checkpoint migration. The independent
CUDA check also covered original-operator parity, Support-response gradients,
LL-to-inverse gradients, three optimizer steps and strict save/reload.

CUDA acceptance preceded the final strict-migration guard and equivalent local
expression cleanup. The final product hashes and selected regression results
are recorded separately. No fresh reviewer or remote training job was launched
for this integration.

## Current verification

| Check | Result |
| --- | --- |
| Complete core suite | **1839 passed**, zero skips, Python 3.11.14 / Torch 2.13.0; CPU and real MPS paths |
| Added native inference cases | **6 passed**: V7, V7.3/asinh and V7.3/dual-stream, each regression/classification |
| Row dual-stream CLI example | **8 updates completed**, CPU/FP64, mixed single/joint tasks |
| Product source against prior reviews | **21/21 exact hashes** |

The first full run found one MPS test affected by a prior test's global strict
determinism setting (1838 passed, one failed). MPS `index_put` backward has no
deterministic implementation. The dual-stream test fixture now selects its
supported test mode and restores incoming determinism and thread settings in
`finally`. No product-side algorithm setting changed. The complete rerun passed.

The six native inference tests were added after the core collection and run
separately, with the optional pandas dependency supplied by an existing local
dependency directory. They validate actual estimator loading and prediction,
including probability normalization and empty queries. The shared development
environment and training hosts were not modified.

To reproduce with development and native inference dependencies installed:

```sh
OMP_NUM_THREADS=1 python -m pytest -q tests
python -m tabu_lab.cli restoration v7-fit \
  --config examples/v7-restoration/row-dual-stream.yaml --execute \
  --output-dir /path/to/new-output-directory
```

## Research limits

The dual-stream mean readback and assembly rule remain experimental choices.
Exploratory training has shown finite gradient spikes; invertibility and passing
engineering tests do not establish numerical conditioning or predictive
superiority. Asinh compresses numeric inputs but does not bound the full
multi-round model Jacobian. Training loss is computed in code space and does not
pass through the final `sinh` decoder.

This integration makes no unseen-table generalization or benchmark-ranking
claim. Historical source-bound continuation still requires its frozen source
and runtime, and no compatibility check here substitutes for reproducing an old
experiment's next optimizer update in its original environment.
