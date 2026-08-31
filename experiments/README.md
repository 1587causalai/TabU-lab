# Experiments ledger

For real-data model comparisons, the default estimand is the complete train/test
split: all labeled train rows are context and all held-out rows are queries.
See [`docs/architecture/real-evaluation-default-protocol.md`](../docs/architecture/real-evaluation-default-protocol.md).
Finite context or query limits are diagnostic overrides and must be explicit in
the manifest and report.

One directory or file per experiment. Required per run:

- hypothesis & gate criterion (written down before running)
- exact command, seed, config hash, compute used
- raw metrics + curves (or wandb link)
- verdict: pass / kill / revise — failures included

Naming: `G<N>-<slug>/`, continuing the team's gate convention.
