# Provenance of the TAR shared-fitting integration

The TAR implementation and experiment harness are integrated from the project's
local reviewed snapshots, preserving current mainline compatibility behavior.
Historical run identities and the new integration identity are distinct; see
`docs/research/tar-shared-fit-20260907/README.md`.

## Synthetic episode generator snapshot

Donor repository: https://github.com/1587causalai/tabuf-episode-api

License: MIT, copyright (c) 2026 Heyang Gong. The complete license is retained at
`experiments/local/tar-diverse-120-fit/corpus/generator-source/LICENSE`.

The copied files are `generator.py`, `scm_v1.py`, `sources.py`, `requirements.txt`
and `LICENSE`. This snapshot included local generator changes, so a repository
HEAD is insufficient to identify it. Exact SHA256 digests are preserved under
`generator_files` in `corpus/manifest.json`; they are the authoritative locator
for the vendored bytes. No API server, credentials or private configuration is
vendored. The files are fixture-generation support, not import-time package
runtime dependencies.

The 120 generated typed tables retain their original bytes and hashes. They
were selected by predefined mechanism/type strata and finite/nondegenerate
validation, not by model performance. The associated requests and 34 rejected
attempts are retained. This preserves failed generation evidence without
publishing raw infrastructure logs.
