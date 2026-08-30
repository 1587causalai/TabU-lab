# Overlapping active-tree history import

This directory preserves the donor versions of 41 files whose paths also exist on
current `main` but whose bytes differ.

The donor Git base was `3502fdd80539f2a8b9703cc4e4546fd01f3826ce`. The active
main base for this comparison was `116e0975654f39fc07b4b82b7d7fa7e4555d7602`.

## Import rule

Donor-relative paths are retained below this directory, but no file is copied over its
active counterpart. Git therefore preserves both sides of each future reconciliation:

- active-main content at the canonical path;
- dirty-donor content at this history path.

## Scope

The overlaps include ModelSpec/source manifests, registry and model builders,
mathematics/catalog projections, source/evidence identity, TabUBase experiment code,
verification exports, and their historical tests.

## Claim boundary

This snapshot does not decide which side is semantically newer or correct. Current
`main` remains authoritative until a later governance PR performs an explicit
field-by-field reconciliation with tests and review. The import commit provides the
immutable identity of the preserved donor side.
