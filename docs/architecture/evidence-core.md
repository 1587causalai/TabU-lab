# Evidence core

The evidence core gives later experiment and publication layers a small, strict
vocabulary. It separates four things that are easy to blur together:

- a `Preregistration` states a proposed test before a run;
- a `Receipt` records one completed attempt and its bound artifacts;
- a `ClaimRecord` tracks whether a bounded statement is proposed, accepted, or
  rejected;
- a `SourceIdentity` states exactly which reviewed source produced an attempt.

All public models reject unknown fields and expose deterministic content hashes.
The checked-in JSON Schemas are generated from the same runtime models.

`write_receipt` publishes canonical JSON with create-if-absent semantics, then
reads it back and verifies its embedded hash. It refuses overwrites, malformed
files, non-canonical serialization, and payload tampering.

Source identity is fail-closed. A clean Git checkout can be marked `formal` only
when the exact commit, remote ref, source tree, dependency lock, preregistration,
and independent review bindings are present. A verified immutable distribution
has an equivalent archive-and-installation path. If any required binding is
missing, the identity stays `local_unissued` and records why.

Public evidence strings are screened for host-local paths, `file:` URIs, private
host or user labels, credentials, and common token shapes. Public remote URIs and
ordinary slash notation remain valid.

This package supplies evidence primitives only. It does not make a local run
formal, approve a review, accept a claim, or publish a catalog entry.
