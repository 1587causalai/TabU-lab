# Independent review: restoration parallel v2

- Reviewed final commit: `a50ea62085a4c700ed7bbba19487de6dd430024b`.
- Frozen reference: `4e2c04e`.
- Status: `local_unissued`.
- Conclusion: no unresolved review blocker within the checked scope.

## Implementation and independent CPU checks

The changes group visible numeric statistics, typed features, readout packing,
loss reductions, numeric truth encoding, and inverse scaling, and reuse collect
presence. Visible facts remain separate from scorer-only `TruthSidecar` values.
Truth encoding detaches targets; custom codec subclasses retain their methods.
Padded targets are excluded before inverse scaling.

During core review, 16 independent CPU comparisons loaded the complete frozen
reference dependencies. They covered direct/inducing, four category mappings,
NW/LL, ragged supports, untargeted empty numeric/categorical columns, mixed
FP32/FP64 input values, reordered requests, G/Q/Z/B states, and nonuniform state
weights. Maximum loss/per-target difference was `1.4210854715202004e-14`;
maximum parameter-gradient difference was `2.1316282072803006e-14`.
Zero-row and empty-request inference checks also passed.

Subsequent review findings were addressed before the final run:

- The pipeline comparator now rejects matching nonfinite values and compares
  discrete outputs and reporting counts exactly.
- Native numeric codecs with answer width other than one are rejected before
  slicing. Batched inverse scaling preserves scalar FP32/FP64 dtype behavior;
  regression tests cover rejection, detached truth, and decode gradients.

## Independently verified final GPU receipts

The copied raw `benchmark.json` and `cuda-check.json` were read and checked
against Git source bytes. All 17 current source-file hashes match the final
commit; all 10 reference-file hashes match `4e2c04e`. The benchmark harness hash
also matches. The 14-file component digest recomputed from the final commit
matches the CUDA receipt:
`2b5caf3012cb1d5cfbbe8baf1512252a7cd14a613bbe531219b918aac195b0ac`.

On NVIDIA GB10, FP64, the complete-pipeline comparison passed all four fixed
masks, including loss, outputs, decoded values, reporting statistics, and
parameter gradients. Maximum absolute difference was `6.217248937900877e-15`.
Three AdamW update comparisons passed with maximum model/optimizer-state
difference `4.9144716074422945e-15`. Initial weights were restored before timing.

The separate mixed numeric/nominal/ordinal damage device probe passed CPU/CUDA
forward/backward comparison with maximum difference `6.49147402498329e-13`.
Its two-update checkpoint continuation compared model, AdamW, and CPU/CUDA RNG
state exactly; every recorded difference was zero.

All 16 timing records are finite and positive, with four alternating rounds per
variant and phase, eight calls per round, and four warmup calls. Recomputed
medians and ranges match the receipt:

| Timed phase | Previous batched | Parallel v2 | Latency reduction |
|---|---:|---:|---:|
| Forward, no grad | 35.224 ms | 29.478 ms | 16.31% |
| Forward + backward | 92.161 ms | 84.305 ms | 8.52% |

Timing includes encoding, validation, scorer work, and gradient clearing, with
CUDA synchronization at measurement boundaries. Optimizer updates and
checkpoint I/O are outside the timed phase.

## Evidence boundaries and receipt identities

Timing applies to the fixed **32-row, 8-numeric-column** panel with four masks,
width 128, three inducing layers, eight heads, 256 slots, and LL readout. The
mixed-type device probe uses one layer and two slots. These are bounded
correctness and execution measurements; they establish neither a new training
result nor unseen-table performance, broad throughput, or issued evidence.

- `benchmark.json` SHA256:
  `c5b34eec2e5b30123358814676beec2a2553968adc2f91074f0c93ba320c682f`.
- `cuda-check.json` SHA256:
  `940bdd5e2b55e356cc2587f8751bd0db8557c0d7039bc1a6945ffbced00c25eb`.
