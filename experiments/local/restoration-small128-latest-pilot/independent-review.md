# Independent review: latest restoration trial

- Status: `local_unissued`.
- Reviewed commit: `363ac0a9e1a6989b9cd4d4a605f7a2c544d61205`.
- Working-source capture: `2026-09-15T23:37:40+0800`.
- Scope: frozen batched restoration implementation, exact-zero source correction,
  numeric codec API compatibility, and bounded CUDA qualification harness.
- Conclusion: no unresolved code-review blocker for the bounded CUDA trial.

## Independently checked evidence

The restoration CPU suite passed: **135 passed in 5.00s**, using the existing
project Python environment with `PYTHONDONTWRITEBYTECODE=1`,
`OMP_NUM_THREADS=1`, `PYTHONPATH=src`, and arguments
`-m pytest tests/unit/restoration -q -p no:cacheprovider --tb=short`.

Eight separate FP64 equivalence probes also passed, including a repeat against
the reviewed source after the zero-presence correction:

- Four backbone comparisons loaded the original implementation from commit
  `4806471` as an independent reference. Direct and inducing variants each
  covered mixed source masks, an empty source column, and an all-empty source
  case. Outputs and gradients of inputs and parameters were compared.
- Four readout comparisons checked batched NW/LL against individual
  `RestorationReadout.__call__` calls, with prefix and non-prefix masks,
  unequal target/support counts, and zero padding of supports and answer
  coordinates. Outputs, equivalent coefficients, and geometry/Cell gradients
  were compared.

The largest absolute output/coefficient difference was `4.551914400963142e-15`;
the largest gradient difference was `4.440892098500626e-15`. Probe tolerances
were `rtol=1e-8`, `atol=1e-9`, with RNG seed `19`.

The review reproduced a numerical regression in the initial snapshot: an
exact-zero projected-presence source could overflow content logits before its
mass was removed. The correction zeros these sources before K/V projection.
Its regression test passes, including exact output agreement with zero sources
and zero source gradients. Ineligible NaN payload isolation also passed a
separate forward/gradient comparison. The inherited fit assertion now uses the
current `epsilon` codec field.

## CUDA harness review and limits

The harness copies one initialized model to CPU/CUDA, performs separate forward
and backward passes, and compares loss, per-target losses, encoded/decoded
outputs, and every named parameter gradient. The continuation probe serializes
model, AdamW, and CPU/CUDA RNG state after one update, then compares uninterrupted
and restored second updates with zero tolerance, including model and optimizer
tensors and RNG state. Negative comparator probes rejected incorrect/missing
gradients, nonfinite values, shape mismatches, and a `1e-12` continuation delta.

Reviewed harness SHA256:
`d104bc76fd1a0dc847474a9814c81af30049d06ce417cce2f318b5efa23bbb97`.
Its bytes and the reviewed backbone, CLI, and added tests match the stated commit.

**CUDA had not been run at the time of this review.** The intended device probe
covers FP64 inducing/identity128/LL, width 128, one layer, two slots, and a mixed
damage episode. This review establishes CPU checks and harness correctness;
it does not establish CUDA success, full-scale execution, fit improvement,
held-out performance, generalization, or issued evidence. Actual CUDA and pilot
execution outcomes must be recorded separately without rewriting this review.
