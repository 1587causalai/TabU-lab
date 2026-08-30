# Fit-first experiment program

This directory contains the preregistered fit-first gates. The order is intentionally
strict:

```text
F0 fixed-episode fit
→ seven-model synthetic checkpoint review
→ S1 multi-episode synthetic fit
→ seven-model synthetic checkpoint review
→ R1 classic real-data training fit
```

`tabu4do` is `design_open` and cannot enter this program. Every stage starts from a new
random initialization; no checkpoint is inherited between stages. Validation and test
partitions are diagnostic only at this stage.

The checked-in F0 preregistrations target `cuda:0` on the trusted primary experiment
host. A CPU or MPS replay is a separate execution config and therefore a separate
`run_id`; all three backends use the repository's frozen `float32` model path.
MPS currently runs with `deterministic_algorithms: false`,
`evidence_mode: diagnostic_nondeterministic`, and `exact_resume: false`. Such a run may
finish as `diagnostic_pass`, but can never satisfy Gate 1 or issue formal evidence.
Passing F0 proves only support-realizable fitting of one frozen episode under the named
contract, data hash, budget, and seeds. It is not evidence of generalization,
pretraining benefit, recommendation quality, graph-learning advantage, or a foundation
model.

Every F0 lane now carries a baseline ladder. The preregistration and receipt must keep
these questions separate:

- **objective fit:** can a parameterized reference drive the masked-target objective
  down when truth is available only at the objective boundary;
- **context recovery:** can a declared classical baseline trained only on visible
  observations recover the same masked observed targets;
- **architecture position:** how does the TabU implementation compare with the trivial
  and classical baselines across all required seeds and the same declared budget.

The baseline ladder is diagnostic evidence, not a replacement for the contract gate.
A model that improves over the trivial baseline but trails a stable classical baseline
is recorded as `trainable, baseline-behind`, not killed or promoted. Baseline identity,
fit/target cells, normalization, optimizer, budget, and stopping reason are hashed into
the experiment identity. A target-only reference fit may establish objective
optimisability, but it cannot be described as context recovery.

F0 experiment identities are append-only. In particular, `F0-017` is retained as a
three-seed diagnostic of the TabUFL joint architecture, but its 12-target F ledger does
not satisfy the frozen 16-F contract. `F0-018-tabufl-balanced-16f-v5` is the current
TabUFL checkpoint candidate and uses an independent generator source artifact so its
registration does not rewrite the source hash of F0-001 through F0-017.

The canonical F0 revision chains are:

| Lane | Append-only chain | Current draft candidate |
|---|---|---|
| `tabuf` | `G000 → F0-001 → F0-008` | `F0-008-tabuf-identifiable-v2` |
| `tabu.unit_row` | `F0-002 → F0-009` | `F0-009-tabu-unit-row-identifiable-v2` |
| `tabu.unit_pair` | `F0-003 → F0-010 → F0-020` | `F0-020-tabu-unit-pair-local-linear-contract-v1` |
| `tabu4graph` | `F0-006 → F0-011` | `F0-011-tabu4graph-row-unit-v2` |
| `tabu4rec` | `F0-007 → F0-014 → F0-021 → F0-022` | `F0-022-tabu4rec-cell-global-support-v1` |
| `tabul` | `F0-004 → F0-012 → F0-015` | `F0-015-tabul-unit-linked-address-v3` |
| `tabufl` | `F0-005 → F0-013 → F0-016 → F0-017 → F0-018` | `F0-018-tabufl-balanced-16f-v5` |

Every arrow is declared by the successor's `supersedes_experiment_ids` and
`revision_rationale`. It replaces the protocol candidate, not the historical record.
All listed objects remain `draft` until review evidence promotes them; a supersedes edge
must never be inferred as `revised`, `preregistered`, or accepted evidence. A custom
diagnostic experiment ID does not inherit canonical supersession automatically.

`F0-020-tabu-unit-pair-local-linear-contract-v1` is the contract-alignment variant for
the Unit-as-cell lane: it makes Local Linear the declared numeric default, uses the
literal direct projection `z = Wc`, and removes implicit row-centering from the readout.
It supersedes the F0-010 protocol candidate without rewriting F0-010's historical
receipt or changing the meaning of the earlier NW alternative.

`F0-021-tabu4rec-axis-address-wide-v1` is a separate TabU4Rec readout realization. It
keeps the dual-axis inducing dynamics and equal active-arm evidence contract, but
widens the truth-free user/item axis summaries from 2 to 8 dimensions and increases
the bounded matched residual scale from 0.1 to 1.0. It is a diagnostic successor to
F0-014; its pass does not rewrite the earlier result or establish recommendation
quality.

`F0-022-tabu4rec-cell-global-support-v1` is the direct unit-as-cell readout variant:
it uses one shared projection `Wc` and one routing denominator over the union of
same-user and same-item observed supports. It removes the per-arm `0.5` mixture from
the prediction path while retaining support provenance in the trace.

Generate and validate the tracked assets with:

```bash
python scripts/build_fit_first_assets.py --device cuda --device-index 0
tabu-lab experiments validate \
  experiments/fit-first/F0/F0-001-tabuf-v1/preregistration.yaml
```

Formal runs must use a source commit containing the preregistration. Receipt directories
are immutable and are not accepted claims until a non-developer replay, review report,
and Gong approval exist.
