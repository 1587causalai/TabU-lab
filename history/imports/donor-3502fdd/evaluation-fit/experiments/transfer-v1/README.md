# TabU Synthetic → Real Transfer v1

This directory is the reviewed protocol surface for the first TabUL transfer
wedge. The files are versioned specifications, not evidence receipts:

```text
synthetic-prior.yaml       synthetic generator mixture and cache identity
pretrain.yaml              S1 scale/checkpoint/resume contract
icl-harness.yaml           Link 5 ICL protocol: context-size sweep plus no-pretrain and scratch-finetune control arms
r1-comparison.yaml         paired R1 gate and claim boundary
finetune-template.yaml     blocked example until the task passport is bound
task-manifest.json         raw-data-free R1/R2 task inventory (generated)
```

The specs deliberately contain no raw OpenML rows or labels. A formal
run must bind an immutable cache manifest, dataset passports, split manifests,
source identity, and receipt before it can be counted as evidence. `TabUL`
is the only model in this v1 wedge; `tabu4do` remains `design_open`.

Validate the specs with:

```bash
tabu-lab experiments validate experiments/transfer-v1/synthetic-prior.yaml
tabu-lab experiments validate experiments/transfer-v1/pretrain.yaml
tabu-lab experiments validate experiments/transfer-v1/icl-harness.yaml
tabu-lab experiments transfer-manifest --json
```

The public claim remains narrow: under this frozen protocol, synthetic-
pretrained initialization improves low-data full-parameter fine-tuning over a
same-architecture scratch arm. This does not establish architecture
superiority, persistent-Unit use, a foundation model, or causal
identification.

The `icl-harness.yaml` protocol is the executable form of Link 5 in the
[six-link evaluation ladder](../../docs/architecture/eval-chain.md): it binds
`pretrain_spec_sha256` to this directory's pretrain contract, sweeps held-out
in-context accuracy over `context_sizes`, and closes the arm set to
`icl_pretrained`, `icl_no_pretrain`, and `finetune_scratch` so that any
in-context ability is attributed to pretraining rather than to the prompt
format or to gradient adaptation.
