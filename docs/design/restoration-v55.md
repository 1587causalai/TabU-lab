# V5.5 implementation and version boundaries

This is the implementation entry point for the V5.5 restoration experiments. The
owner-maintained design remains in
`TabU/latex/model-factory/table-restoration/end-to-end-design.tex`; the V5.5
revision package is a separate design candidate. Code availability and local
checks do not establish model fit or transfer.

## Model identity

`V55Model` and `V55Config` live in `tabu_lab.models.restoration_v55`. The model
uses the shared V5.3 execution engine and V5.4 named sizes. The constructor
defaults to `constant_weight_composition_v2` and zero Unit layers. Positive Unit
depth must be selected explicitly; the existing Small/H4 and Small/H8 training
lineages used three Unit layers and must be reconstructed from their resolved
manifests and checkpoints, not from the constructor defaults.

`constant_weight_composition_v3` is a separately identified nominal direction
candidate. Legacy composition versions remain explicit choices. Codec version,
Unit depth, source digest, data, recipe, and objective belong to the resolved
experiment identity; selecting another value does not reinterpret a historical
checkpoint.

## Curriculum and continuation

`tabu-lab curriculum-v55` exposes `plan`, `preflight`, `run`, and `evaluate`.
V5.5 uses a versioned manifest and source digest. The outer objective defaults to
`squared`; `query_log1p_scaled` is an explicit Query-only candidate with its own
`tau`. Evaluation metrics do not select the training objective.

The V5.5 CLI also exposes guarded `prepare-v54-warm-start` and
`prepare-v55-warm-start` conversions. `--initialize-from` creates a model-only
continuation with fresh optimizer, RNG, cursor, and exposure. The special V5.4
conversion has narrower old120, codec, and model-config requirements; inspect
`warm_start.py` and its receipt before using it. `--resume-checkpoint` requires
the same experiment identity and restores the exact training state. Every
invocation uses a new attempt directory.

The opt-in `query_cycle` window sampler applies to V5.5 numeric train-role
tables with a bounded window and supervised-row recipe. It records actual
train-row and Query exposure. Its schedule guarantee is conditional on enough
visits; adding it to a manifest alone does not prove full-pool coverage.

## Local verification scope

The unit checks cover versioned plan parsing, codec identities, Query objective
behavior, warm-start guards, query-cycle sampling, evaluation, and checkpoint
continuation. They use local fixtures. Device readiness, a formal training run,
fixed-Query fit, retention, and held-out evaluation require their own receipts.

The V5.4 implementation remains documented in
[restoration-v54.md](restoration-v54.md). Its model and curriculum identities
remain distinct from V5.5.
