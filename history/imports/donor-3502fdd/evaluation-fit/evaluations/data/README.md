# Evaluation v0 retained-data registration

This directory contains **request templates, not datasets and not evidence**.
Every `*.template.yaml` is intentionally invalid until its placeholders are
replaced with facts derived from caller-retained source bytes.

The workflow never downloads data:

```text
official or reviewed retained local file
  + explicit format/split/mask authority request
  -> private PreparedScenario bundle (source bytes + evaluator truth)
  -> public DatasetSnapshot manifest (hashes and boundaries only)
```

## One-command boundaries

After filling one request template with the exact source SHA-256, byte size,
serialization, exhaustive train/validation/test assignment, and any required
mask or perturbation authority:

```bash
tabu-lab eval data prepare request.yaml \
  --source .local-runs/eval-data/retained/source.bin \
  --output .local-runs/eval-data/private/<scenario-id>.prepared.json

tabu-lab eval data register \
  .local-runs/eval-data/private/<scenario-id>.prepared.json \
  --output datasets/<snapshot-id>.json

tabu-lab eval data check \
  .local-runs/eval-data/private/<scenario-id>.prepared.json \
  --snapshot datasets/<snapshot-id>.json
```

`prepare` rejects URLs and only reads the explicit local file. Both `prepare`
and `register` are create-once: a byte-identical retry is idempotent, while a
different payload at the same path is rejected. `check` is read-only.
Within the active repository, retained sources and prepared bundles must be
Git-ignored; an unignored path fails before evidence is written. Repo-external
retained sources remain valid. The public `datasets/` manifest is intentionally
tracked and is not subject to the private-output rule.

## Evidence boundary

- The private bundle is marked `private_evaluator_input` and contains retained
  source bytes plus train/validation/test truth. Do not place it under
  `datasets/`, commit it, publish it, or serve it from the public site.
- The registered `DatasetSnapshotSpec` contains no source bytes, local path, or
  per-example truth. It binds source, prepared content, split, recipe, and truth
  sidecar hashes.
- Offline hash consistency does not prove that arbitrary bytes were acquired
  from the named upstream URI. Acquisition provenance remains a separately
  reviewed authority claim; these commands neither fetch nor attest it.
- Adult/Diabetes templates require the exact retained delimited representation
  declared in the authority (including a stable row-id column). Karate requires
  the versioned retained JSON representation. MovieLens consumes the official
  ZIP with explicitly named base/test members and a train-side validation
  carve.
- A split authority must exhaustively and disjointly assign every retained row.
  The workflow will not invent a validation carve, split seed, parser, row ID,
  mask seed, topology perturbation, or MovieLens validation interaction list.

The six templates correspond to the six scenarios across the four frozen v0
suites. Real candidate snapshots, when available, live under
`datasets/candidates/`; they remain `self_consistent_unreviewed`, carry no
review ids, are not publication eligible, and cannot authorize a formal
evaluation. No EvalResult is checked in by this data-registration step.

As of the current candidate freeze, four scenario snapshots are registered:

- scikit-learn Diabetes supervised regression;
- scikit-learn Diabetes feature completion;
- Zachary Karate Club graph completion;
- MovieLens-100K interaction completion.

The two Adult-backed scenarios remain absent and fail closed until their fold,
row-id semantics, and retained license evidence have been independently
resolved. Registration records hash consistency and review inputs; it is not
an upstream provenance approval.

## Deterministic candidate freeze

The repository also provides an offline exporter for the exact retained source
bytes accepted by the v0 data program. It emits one private create-once bundle:

```text
freeze-manifest.json
requests/*.request.json
retained/*
```

The manifest is a review subject, not an approval. Its state is fixed to
`self_consistent_unreviewed`, `publication_eligible: false`, empty `review_ids`,
and `network_access: false`. The exporter validates every generated request by
running the live scenario materializer before it writes the bundle. Different
bytes at an existing output path are rejected; a byte-identical retry is
idempotent. Because the bundle contains labels and evaluator truth, its output
root must be outside a Git worktree or covered by `.gitignore`.

For Diabetes, the outer freeze manifest pins the exact scikit-learn 1.9.0 raw
feature and target files. The two preparation requests retain the broader
`scikit-learn-1.x-*` identifiers because those strings are the compatibility
identities frozen by the current suite YAML and enforced by the materializer.
The manifest records both layers explicitly; the broader request identifier
must not be read as weakening the exact 1.9.0 byte pins.

Using the retained preflight files as an example:

```bash
uv run python scripts/freeze_eval_data_authority.py diabetes \
  --data /path/to/diabetes_data_raw.csv.gz \
  --target /path/to/diabetes_target.csv.gz \
  --split-seed 1729 \
  --mask-seed 1729 \
  --output-root /private/ignored/eval-data/diabetes

uv run python scripts/freeze_eval_data_authority.py karate \
  --split-seed 1729 \
  --output-root /private/ignored/eval-data/karate

uv run python scripts/freeze_eval_data_authority.py movielens \
  --zip /path/to/ml-100k.zip \
  --validation-seed 1729 \
  --validation-count 8000 \
  --output-root /private/ignored/eval-data/movielens

uv run python scripts/freeze_eval_data_authority.py check \
  --output-root /private/ignored/eval-data/movielens
```

The generated request and retained representation feed the existing workflow
without transcription:

```bash
uv run tabu-lab eval data prepare \
  /private/ignored/eval-data/diabetes/requests/diabetes-supervised.request.json \
  --source /private/ignored/eval-data/diabetes/retained/sklearn-diabetes-1.9.0-raw.csv \
    --output /private/ignored/eval-data/prepared/diabetes-supervised.prepared.json
```

For TabUBase 0.2.0, pass `--suite-version v1` to the `adult` and `diabetes`
exporter commands. This emits the independent Base scenario identities and
does not rewrite the retained v0 candidates. Base v1 keeps the train-only
boundary but fits numeric statistics and categorical codebooks on the complete
declared train partition (`statistics_fit_scope: full_train_partition`); the
validation/test selections remain unchanged and never contribute fitting
evidence.

The Adult exporter intentionally has no default fold. The data ARFF and task
split ARFF do not themselves establish which of task 7592's ten folds should be
the public v0 fold, that `rowid` means zero-based ARFF data-row order, or the
license claim attached to those bytes. Adult therefore writes nothing unless
all three are supplied explicitly:

```bash
uv run python scripts/freeze_eval_data_authority.py adult \
  --data-arff /path/to/openml-adult-v2-1590.arff \
  --task-splits-arff /path/to/openml-task-7592-splits.arff \
  --license-evidence /path/to/retained-license-evidence \
  --fold 0 \
  --rowid-semantics openml-task-rowid-zero-based-arff-data-order-v1 \
  --validation-seed 1729 \
  --mask-seed 1729 \
  --output-root /private/ignored/eval-data/adult
```

Even with those arguments, the result remains unreviewed. Independent review
must inspect the freeze manifest, the bound upstream bytes, split/validation
decision, representation, and the create-once
`evidence/adult-license-evidence.bin` copy before the normal dataset
authority promotion workflow can make a `DatasetSnapshotSpec` publication
eligible. The exporter never performs that promotion and never starts an
evaluation or training run.
