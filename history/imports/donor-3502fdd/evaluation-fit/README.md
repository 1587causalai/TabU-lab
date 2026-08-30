# Evaluation and fit history import

This directory is an immutable, non-canonical snapshot of selected implementation
files from the preserved dirty donor whose Git base was
`3502fdd80539f2a8b9703cc4e4546fd01f3826ce`.

## Scope

The snapshot preserves 138 donor files byte-for-byte:

- evaluation-foundry contracts, adapters, scoring, runners, suites, and schemas;
- fit-first F0/S1/R1 preregistrations and execution helpers;
- dataset candidates, evaluation chain/passport records, and local-unissued
  verification results;
- checkpoint/data-freeze adapters, formal-receipt and authorization scaffolding;
- transfer-v1/base-v1 contracts, supporting scripts, and historical tests.

Paths below this directory preserve their donor-relative layout. They are deliberately
outside active package, schema, experiment, dataset, evaluation, verification, and test
discovery roots.

## Claim boundary

This snapshot preserves source history only. Historical result JSON remains
`local_unissued`; no receipt, dataset authority, benchmark, replay, capability, or
maturity claim is promoted by importing it. Compatibility and activation issues are
recorded in `docs/reviews/history-import-register.md` for later governance.
