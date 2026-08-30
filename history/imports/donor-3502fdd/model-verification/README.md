# Model and verification history import

This directory is an immutable, non-canonical snapshot of selected implementation
files from the preserved dirty donor whose Git base was
`3502fdd80539f2a8b9703cc4e4546fd01f3826ce`.

## Why this exists

The current `origin/main` is the execution baseline. Historical implementation is
first collected here without changing active package, schema, registry, or test
discovery paths. Integration findings are registered separately and are not repaired
during history consolidation.

## Scope

The snapshot preserves 47 donor files byte-for-byte, plus the donor's migration
provenance note. The selected files cover:

- model-family ModelSpec projections and their packaged copies;
- the component registry growth seam;
- component-correctness and architecture-evolvability verification contracts,
  probes, runners, suites, and schemas;
- associated scripts and historical tests.

Paths below this directory preserve their donor-relative layout. Their presence does
not activate them as canonical package code, public schemas, registry entries, or test
suites.

## Claim boundary

This is source-history preservation only. It is not evidence that the snapshot is
compatible with current `main`, executable, reviewed as a model implementation, or
eligible for maturity promotion. The import commit provides the immutable content
identity for this selected snapshot.
