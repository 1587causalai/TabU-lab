# TabUBase frozen nominal codebook B=100 ablation

Date: 2026-08-29
Evidence status: `local_unissued`
Contract: `tabu.cell.base@0.2.0`
Profile: `supervised.label_broadcast.v1`

## Question

Does replacing the historical per-Episode random-sphere nominal tokenizer with
a source-scoped frozen codebook improve synthetic fitting and real
classification transfer? The treatment freezes 100 unit-sphere codes and maps
`codebook_id + declared domain label` deterministically. Domains above 100
categories fail closed.

The historical `cell-tokenizer.v1` remains unchanged. The treatment is the
distinct `cell-tokenizer.v2` composition identity and cannot inherit v1
receipts or silently load into a v1 model.

## Paired PT-S0 and PT-S1, seed 1729

Both arms use the same synthetic world IDs, model seed, architecture, optimizer,
learning rate, update budget, and validation worlds.

| Phase | Tokenizer | Initial validation loss | Final validation loss | Reduction | Gates |
|---|---|---:|---:|---:|---|
| PT-S0 | v1 Episode random sphere | 1.167515 | 0.658676 | 43.6% | 3/3 pass |
| PT-S0 | v2 frozen codebook B=100 | 1.161379 | 0.480340 | 58.6% | 3/3 pass |
| PT-S1 | v1 Episode random sphere | 1.167515 | 0.287935 | 75.3% | 3/3 pass |
| PT-S1 | v2 frozen codebook B=100 | 1.161379 | 0.078806 | 93.2% | 3/3 pass |

At PT-S0 the v2 endpoint is 27.1% lower than the paired v1 endpoint. At PT-S1
it is 72.6% lower. This is strong synthetic-fit evidence for stable nominal
geometry, but it does not by itself establish better real-data transfer.

## Corrected real-classification diagnostic, seed 1729

Each arm uses 128 labels, 400 updates, and the validation-selected learning
rate `1e-4`. Log Loss is primary and lower is better. XGBoost and MLP use the
same split and labeled subset.

| Dataset | v2 pretrained | v1 pretrained | v2 scratch | XGBoost | MLP |
|---|---:|---:|---:|---:|---:|
| Iris | 0.2628 | 0.5391 | 0.7961 | 0.3963 | 0.1155 |
| Wine | 0.0027 | 0.0850 | 0.1760 | 0.0540 | 0.0168 |
| Breast Cancer | 0.4524 | 0.3616 | 0.1675 | 0.1378 | 0.2321 |

The v2 checkpoint improves over v1 pretrained by 51.2% on Iris and 96.8% on
Wine, and it beats XGBoost on both. Wine v2 also beats the MLP. Breast Cancer
remains negative transfer: v2 has higher accuracy than the MLP (0.9391 versus
0.8870) but much worse Log Loss, indicating overconfidence/calibration failure
rather than lack of class-separation accuracy.

## Runtime diagnostic

Running v1 and v2 simultaneously on the same GB10 was counterproductive:
1000 updates took 54.1s and 73.6s, versus approximately 22--24s per 1000 when
run sequentially. The concurrent attempt was stopped without producing a
receipt and preserved under the remote `aborted-concurrency/` directory.

## Three-seed PT-S1 and classification result

The v2 PT-S1 extension passed all three gates for all three root seeds:

| Seed | Initial validation loss | Final validation loss | Reduction |
|---:|---:|---:|---:|
| 1729 | 1.161379 | 0.078806 | 93.2% |
| 2718 | 1.475494 | 0.149931 | 89.8% |
| 31415 | 1.104564 | 0.141769 | 87.2% |

The corrected three-seed real classification panel gives:

| Dataset | v2 pretrained | v2 scratch | XGBoost | MLP | Wins vs scratch / XGB / MLP |
|---|---:|---:|---:|---:|---:|
| Iris | 0.4939 | 0.3692 | 0.4006 | 0.2378 | 1/3 · 2/3 · 1/3 |
| Wine | 0.0039 | 0.2942 | 0.0491 | 0.0122 | 3/3 · 3/3 · 3/3 |
| Breast Cancer | 0.5540 | 0.1967 | 0.1310 | 0.1796 | 0/3 · 0/3 · 0/3 |

The robust positive result is Wine. Iris is seed-sensitive, and Breast Cancer
is a stable negative-transfer result under the current schedule.

## Validation-only temperature diagnostic

High accuracy with poor Log Loss on Iris and Breast Cancer points to
overconfidence. A fixed grid selected $T$ using validation only and then
applied $p_T(c)\propto p(c)^{1/T}$ once on test. It does not change the predicted
class or model weights.

| Dataset | Pretrained raw | Pretrained calibrated | Scratch calibrated | Selected pretrained T |
|---|---:|---:|---:|---|
| Iris | 0.4939 | 0.1965 | 0.2351 | 5 / 5 / 5 |
| Wine | 0.0039 | 0.0443 | 0.2366 | 5 / 3 / 3 |
| Breast Cancer | 0.5540 | 0.2083 | 0.1988 | 3 / 5 / 5 |

Calibration materially repairs Iris and Breast Cancer, confirming that much
of their NLL failure is probabilistic overconfidence rather than class-ranking
failure. It hurts Wine because the small validation partitions select excess
softening even though raw test calibration is already excellent. Temperature
scaling therefore cannot be applied unconditionally; a future rule needs
shrinkage toward $T=1$ or a larger calibration authority.

## Frozen ICL: fixed-K pretraining failure

The first executable Link-5 harness used 512 held-out
heteroscedastic/missingness worlds, split evenly between classification and
regression, with $K\in\{0,1,2,4,8,16,32\}$. Frozen arms created no optimizer
and preserved identical parameter hashes before and after evaluation.

The original PT-S1 checkpoint, trained with 64 context labels on every update,
passed regression but failed classification:

| Modality | Pretrained AULC | Random AULC | Random-minus-pretrained gain (95% CI) | Normal-minus-shuffled gain (95% CI) |
|---|---:|---:|---:|---:|
| Classification normalized NLL | 4.8504 | 4.2706 | -0.5798 [-0.6494, -0.5198] | 0.3822 [0.2971, 0.4671] |
| Regression scaled RMSE | 1.1874 | 1.3732 | 0.1858 [0.1567, 0.2138] | 0.3482 [0.3131, 0.3845] |

Classification therefore used the label/context relationship but the fixed-K
pretrained initialization was worse than random. A training-family replay had
the same low-K classification failure and turned positive only at $K=32$,
which localized the dominant mismatch to the training context-size schedule,
not only to held-out family shift.

## Variable-K curriculum intervention

A distinct local-unissued variant changed only the pretraining context schedule
to $(2,4,8,16,32,64)$. It retained the 100-code tokenizer, architecture,
world families, optimizer, learning rate, weight decay, and response schedule.
The historical fixed-K checkpoint and its failed ICL result remain immutable.

PT-S0 passed finite, exact-resume, and validation-improvement with validation
loss $3.8623\rightarrow3.3983$. All three PT-S1 seeds also passed those gates:

| Seed | Initial validation loss | Final validation loss |
|---:|---:|---:|
| 1729 | 3.8623 | 3.1110 |
| 2718 | 2.7978 | 1.7681 |
| 31415 | 2.4062 | 1.6023 |

On the same 512 held-out worlds, the variable-K checkpoint passed every frozen
ICL gate:

| Modality | Pretrained AULC | Random AULC | Random-minus-pretrained gain (95% CI) | Normal-minus-shuffled gain (95% CI) |
|---|---:|---:|---:|---:|
| Classification normalized NLL | 4.2387 | 4.2706 | 0.0319 [0.0278, 0.0364] | 0.0304 [0.0262, 0.0346] |
| Regression scaled RMSE | 1.1151 | 1.3732 | 0.2581 [0.2324, 0.2848] | 0.3138 [0.2790, 0.3491] |

This is mechanism evidence that variable-context pretraining, rather than the
codebook alone, is necessary for low-context classification ICL in the current
architecture. The effect is small for classification and large for regression;
the pending scratch-finetune reference and independent replay still prevent a
formal Link-5 receipt.

The single-checkpoint gates were heterogeneous: seed 1729 passed both
modalities, seed 2718 had a classification interval that slightly crossed
zero, and seed 31415 failed the regression pretrained-vs-random gate. The
protocol-level analysis therefore used one fixed held-out world panel and
treated checkpoint seeds as repeated measurements inside each world cluster.
The three-seed world-clustered result passed both modalities:

| Modality | Pretrained-vs-random gain (95% CI) | Normal-vs-shuffled gain (95% CI) |
|---|---:|---:|
| Classification | 0.0229 [0.0166, 0.0290] | 0.0796 [0.0698, 0.0907] |
| Regression | 0.0987 [0.0696, 0.1277] | 0.2322 [0.2082, 0.2566] |

This aggregate pass must not be restated as every checkpoint seed independently
passing every gate.

## Variable-K real-classification replay, three seeds

The curriculum checkpoints were then fine-tuned with the same 128-label,
400-update real-data protocol:

| Dataset | Curriculum pretrained | Scratch | XGBoost | MLP | Seed wins vs scratch / XGB / MLP |
|---|---:|---:|---:|---:|---:|
| Iris | 0.6189 | 0.3078 | 0.4010 | 0.2378 | 2/3 · 1/3 · 1/3 |
| Wine | 0.0062 | 0.2896 | 0.0472 | 0.0122 | 3/3 · 3/3 · 3/3 |
| Breast Cancer | 0.5654 | 0.3333 | 0.1354 | 0.1796 | 0/3 · 0/3 · 0/3 |

The real-data result does not establish general superiority. Wine is a stable
positive result. Iris is seed-sensitive, while Breast Cancer is a stable
negative result under the current fine-tune/calibration protocol.

## Current boundary and next gate

These are exploratory results, not a sealed benchmark or formal receipt. They
support retaining the 100-code tokenizer and variable-K curriculum as explicit
variants, not making either the default or claiming general classification
superiority. The immediate open gates are the pending scratch-finetune ICL
reference, independent replay, and a preregistered calibration policy. Breast
Cancer still requires a separate calibration/generalization diagnosis.

Execution ID: `tabubase-codebook-b100-v1`. Logical artifact IDs and content
hashes are recorded in the [portable local-artifact index](local-artifact-index.json).
Machine-local paths are intentionally omitted, and artifacts are not bundled
with this repository.

Result SHA-256:

- PT-S1 seed 1729: `f43dbc718eecd0ccaae582a802f89f0da66af51f00e0bb0cc8ee61da1a07ec77`
- PT-S1 seed 2718: `7621546dbe029c114055ef3fc86792f14784f7e305b514595f4019e3f9d46d62`
- PT-S1 seed 31415: `1eea6105b854d0a936d85de31356450057f4aee10fa06bdfaf496be63e0b4148`
- Three-seed raw classification panel: `dd2bb79bcb80706ec6d782c6db7607b58e73f8748adc70db47c44dd06b7d3c28`
- Three-seed temperature diagnostic: `1a23d2b328f160b6583773aab4f13d53d7cb991c928066f10068cbc413e7b8ab`
- Fixed-K held-out ICL seed 1729: `82e2856cb96358cc0f8701ca521138d6c7b9c30f761635ed91e9ce748bc2aa89`
- Variable-K PT-S0 seed 1729: `8f124c1c0b4c0ed5cfb0253db197c35ff84434f1f1f746e5cd93fc70d9be5a02`
- Variable-K PT-S1 seed 1729: `eb803ca62c5caede88c8f56b20f1c606537bffab7362d4d0fb80c63d1fff2cb1`
- Variable-K held-out ICL seed 1729: `cb4e48cfdeef33b6c5d74c3764eccb88e385db85fc714d2af31d467e8087bb66`
- Variable-K real classification seed 1729: `0c5d3b635dfc27c34c9ca9c9c9ebf907a9ccae17f3139f0feb791be7e5a4c796`
- Variable-K PT-S1 seed 2718: `bfee5218e635612b524fd2b25569e318ab62b2867c4fbbf725ed14baa15cdcdd`
- Variable-K PT-S1 seed 31415: `7ac3f697f4cf734b4c13ab64c71f9a084f3b76496d22210c98e069353a477769`
- Variable-K common-world ICL seed 2718: `cfb82b0fb3cb5d5bf2cc819395336dddf4e0d968192eb6008d5065d85648deef`
- Variable-K common-world ICL seed 31415: `823bc862c5626b994ee944c4914ff6ad6c41dc9f753366c315cd18ee33d32e01`
- Variable-K three-seed clustered ICL: `afea37d6ff06f79cadb29b16d1c77259a30db1991228a7919de89177dddeae21`
- Variable-K three-seed real classification: `298c116b5ea5957b975003a98ff0ef59deaa666b1a0f9b265025a1a7430e7ca5`
