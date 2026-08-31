# TabUBase / TabUR v3.1 best-checkpoint independent evaluation

Date: 2026-08-31  
Status: `local_unissued`  
Claim boundary: independent validation-panel Grow evaluation; not formal evidence,
transfer evidence, or an accepted capability claim.

## Outcome

Two independent checkpoint evaluation receipts were issued on dgx2:

- TabUBase: selected step 500 checkpoint;
- TabUR: selected step 1500 checkpoint.

The checkpoints share one newly frozen 256-world panel so their losses are paired,
but each checkpoint has its own checked-in request, request hash, output file, and
receipt hash. Neither receipt inherits the training run's evidence status.

## Independence from checkpoint selection

Checkpoint selection used the earlier 64-world validation panel with addresses hash

`d08ca4c2b769512a0126e29a3116b89c1388c249303e635d74681b35c03f98dc`.

Before either selected checkpoint was evaluated, two requests froze a new panel with:

- partition: `validation`;
- root seed: `271828182`;
- addresses: `program-independent-eval-v1-{index:08d}`;
- worlds: 256;
- addresses hash:
  `75fbe7f811ce732b32916deacdc116325adf1a3ece46ddcde7dca3eeca0f381d`.

The new addresses hash differs from the selection panel. The protocol remains
`tabu.eval.transfer-lanes@1.0.0`, whose split authority is
`validation_only_until_final_evidence`; consequently these receipts remain Grow
diagnostics rather than final held-out evidence.

The frozen request hashes are:

- TabUBase: `fb802eead42e3939671a705b9f9938c0bd380848e5b0175edc76e2fc1190afc4`;
- TabUR: `dfbbf6cb34ddc668bd5679709af7082d8314f0ade6838473cce0108af9259e03`.

## Source and protocol identity

- Evaluation source revision:
  `dc2f2c832819028a2fab283e1777082f729f6844`.
- Evaluation source tree:
  `fbd8471bdfa5ad1c840eb948be21ff8e70cf3aa2`.
- Source archive SHA-256:
  `e9ac8ab088c753e8729067a03ca4f437f889356e534261b64307c583488bab85`.
- Evolution repository hash:
  `64336ec9dc4c5ba8021964f09765cd8c2de7934b4aa0deacfcd2aa05954760af`.
- Evaluation protocol hash:
  `13b4fa632e6e9a8eaeab11906243f45912a1f10166b137f970b66f18fa4c513a`.
- Objective bundle hash:
  `713a1f025674efe2a196e12fd9acbcedbcf0e5cf0544bb50d541db324f9f3d29`.
- World mixture hash:
  `8cc82cf3f98917fc4184b8b67df90a56393f0f70f49cd2c734c8bf4ad32d28f3`.
- Generator hash:
  `db76aeac391a40cd99bc39ccc269c24da9c90be0c51e0202791c94ebbcd2df40`.
- Evaluator SHA-256:
  `d5ff36125eb4052e2aa09a73e0b4b14fddbc960efeac92844670a30c839a265c`.

Execution used Python 3.12.3, NumPy 2.2.6, PyTorch
`2.12.0.dev20260322+cu130`, CUDA 13.0, and NVIDIA GB10.

## Independent receipts

Every receipt scores the same 16,582 targets over all 256 worlds and records zero
abstentions.

| Lane | Step | Mean loss | Median loss | Minimum | Maximum | Receipt hash |
|---|---:|---:|---:|---:|---:|---|
| TabUBase | 500 | 1.309189 | 0.693438 | 0.074616 | 38.606236 | `3c2992da496daca98b53de2a4564a9f15bd4f66fadf09e5bbcf2ba5c5435fc5b` |
| TabUR | 1500 | 1.258690 | 0.666501 | 0.007878 | 40.477627 | `4332dc0a237507b11a47ba55053c22b87e57470e6cb3258f00bc05e22c5d5932` |

Receipt file hashes:

- TabUBase:
  `8fc2f59c3b42aa5262c87834ae8780aeaae855cb6d8930a603c435c7e2d838de`;
- TabUR:
  `62346ae31287f9098a22a61208eb3ae7703d4f724ec2f9da10a590876b3504a1`.

## Paired descriptive readback

Let $L_R(w)$ and $L_B(w)$ denote TabUR and TabUBase loss on world $w$. On this
single frozen panel,

$$
\frac{\overline{L}_B-\overline{L}_R}{\overline{L}_B}=0.03857,
\qquad
\overline{L}_R-\overline{L}_B=-0.050499.
$$

- TabUR has lower loss on 147 of 256 worlds.
- TabUBase has lower loss on 109 of 256 worlds.
- There are no ties.
- The median paired delta $L_R(w)-L_B(w)$ is `-0.011177`.

This is a descriptive comparison only. The loss distribution has large world-level
outliers, and the panel remains validation-authorized rather than a final evidence
split. It does not establish a general capability advantage for TabUR.

## Integrity checks

The consolidated audit passed all checks:

- both receipt schemas and self-hashes validate;
- checkpoint, sidecar, snapshot, RunIdentity, and completed training receipt lineage
  match the frozen requests;
- the two evaluations use identical panel addresses and coverage;
- the evaluation panel differs from the checkpoint-selection panel;
- no optimizer is constructed;
- `TruthSidecar` enters only the objective boundary;
- both model state hashes are unchanged before and after inference;
- every loss is finite and every target is scored;
- both receipts remain `local_unissued`.

Audit hash:
`64a880dc9d81e6e089569f0030d887d253d8c1729209226dfb1107fc643767a8`.
Audit file SHA-256:
`a23de2f23da72510574c1a5018cf399b032df7979639646e5a0c6e2c7e06b71d`.

The local implementation gate passed 44 change-scoped tests, Ruff, and full manifest
validation before the clean source archive was transferred to dgx2.

## Artifact locations

Run root:

`/home/cms/tabubase-runs/20260831-v3.1-best-checkpoint-eval-v1`

Receipts:

- `receipts/query-base-step500.json`;
- `receipts/query-row-step1500.json`;
- `receipts/audit-summary.json`.

After both evaluations, no evaluation process remained and GPU utilization returned
to zero. The host still reports the pre-existing post-OOM memory condition of roughly
101 GiB used and 18 GiB available; this run did not worsen that boundary.
