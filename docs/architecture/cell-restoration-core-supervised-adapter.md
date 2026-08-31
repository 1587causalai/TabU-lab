# Cell-restoration core and supervised-response adapter

Status: architecture seam implemented; runtime integration not yet activated
Evidence status: `local_unissued`; no pretraining or capability claim

## Decision

The query-family mainline is typed cell restoration under an arbitrary
artificial target mask. Supervised tabular prediction is not a second parent
objective. It is a task adapter that chooses one response column, exposes its
context-row values, hides its query-row values, and may add bounded residual
label conditioning.

For a table $X$ and artificial target set $Q$,

$$
\widehat X_Q=F_\theta(\operatorname{Mask}(X,Q)),
\qquad
\mathcal L_{\mathrm{cell}}(Q)
=\frac1{|Q|}\sum_{(r,a)\in Q}
\ell_{\tau_a}(\widehat x_{ra},x_{ra}).
$$

The supervised-response adapter selects a response column $a_y$, context rows
$C$, and query rows $R_Q$:

$$
Q_{\mathrm{sup}}=R_Q\times\{a_y\},
\qquad
\mathcal L_{\mathrm{sup}}
=\mathcal L_{\mathrm{cell}}(Q_{\mathrm{sup}}).
$$

Natural missing cells are not artificial targets and do not acquire truth.
`ARTIFICIAL_MASK`, `QUERY`, natural missing, and the exact-zero null contract
remain distinct.

## Residual seam

The first protected seam is `SupervisedResponseAdapter`, inserted after typed
tokenization and before dynamics. It wraps the existing truth-free label
broadcast proposal $B$ as

$$
H^{(0)}_{\mathrm{adapter}}
=H^{(0)}_{\mathrm{core}}
+\rho_{\mathrm{sup}}
\left(B(H^{(0)}_{\mathrm{core}},\mathcal E)
-H^{(0)}_{\mathrm{core}}\right).
$$

The adapter receives only model-visible evidence. `TruthSidecar` remains outside
the public forward boundary. At $\rho_{\mathrm{sup}}=0$, the output is exactly
the core token carrier. At $\rho_{\mathrm{sup}}=1$, it equals the existing
label-broadcast route.

The seam is implemented in
`src/tabu_lab/models/query_task_adapters.py`. It is intentionally not yet registered
inside `tabu.query.row@0.2.0`: silently inserting it would change variant and
checkpoint identity. Existing profiles and checkpoints remain untouched.

## Target architecture

The durable composition is:

```text
typed table + arbitrary target mask
  -> cell-restoration core
     tokenizer -> dynamics -> geometry -> typed terminal
  -> optional task-adapter composition
     supervised-response / recommendation / future adapters
  -> evaluator-only TruthSidecar loss
```

A task adapter must carry its own source, factory, configuration, composition,
and checkpoint identity. Anonymous callables are not valid adapters. Loading a
core checkpoint into an adapter composition requires an explicit core-state projection;
the existing profile identity gate must never be bypassed.

## Pretraining migration

1. **Foundation core.** Reuse complete v3 synthetic worlds but project them
   through a broad typed artificial-mask sampler. Pretrain only generic cell
   restoration and issue a core-bound checkpoint.
2. **Task specialization.** Attach `SupervisedResponseAdapter`, initially freeze
   the core, and train the adapter on response-column/context-query episodes.
3. **Joint alignment.** Only after the adapter-only gate passes, unfreeze the core
   with a smaller learning rate while retaining generic cell-restoration
   episodes to detect catastrophic specialization.
4. **Independent evaluation.** Report held-out typed-cell restoration, frozen
   supervised ICL, and pretrained-versus-scratch real-task fine-tuning as
   separate evidence layers.

## Activation gates

Before runtime integration, all of the following must pass:

- adapter absent preserves the existing model identity and forward exactly;
- enabled adapter has a distinct composition/variant/checkpoint identity;
- $\rho_{\mathrm{sup}}=0$ is an exact whole-forward degeneration;
- $\rho_{\mathrm{sup}}=1$ reproduces the current label-broadcast behavior;
- changing query truth leaves the core and adapter forward unchanged;
- completion checkpoint tensors are imported only through an explicit,
  validated core-state projection;
- broad cell-restoration v3 and supervised-response v3 share world provenance
  while retaining different mask/adapter manifests;
- no masked-cell, frozen-ICL, or transfer result is promoted from a smoke test.

## Deliberately deferred

- changing the current public ModelSpec or contract version;
- registering the adapter in the canonical component manifest;
- defining the core-state checkpoint projection schema;
- choosing the final mask curriculum, adapter/core learning-rate ratio, or
  residual-gate initialization;
- starting DGX pretraining.

These require the next checkpoint because each changes durable model or
checkpoint identity.
