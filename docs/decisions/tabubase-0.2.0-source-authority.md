# TabUBase 0.2.0 source authority

## Decision

The first public `tabu.cell.base@0.2.0` implementation anchor binds:

- TeX entrypoint SHA-256
  `e239b5f67e2fa55111c201b9497e5bf812b6997bc0e9cc2bf88a1c2d8d8f29a0`;
- recursive semantic source-tree SHA-256
  `28ea55544184e18f23fb378b563b62224b190ecb912b34ab1545d82c33d7e7f2`;
- the exact packaged ModelSpec canonical hash generated from those identities.

This decision becomes authoritative only after independent review, gong approval,
and merge to the public default branch.

## Why the earlier local snapshot does not freeze the public version

An earlier dirty donor workspace used entrypoint SHA-256
`2769168c7e60d1b419d7308c9baec494aa1c851bcdd77c580c02ee311e3bd501`
and ModelSpec hash
`415103ae057e5cf63d033a3685c30b96a2f400575101bf17ee17acc2bb22f452`.
The audit found:

- that source byte snapshot is no longer retrievable in the scoped TabU workspace
  or its Git objects;
- the six F0/S1 directories bound to it contain preregistration YAML only, with no
  formal receipt or gong approval in those directories;
- related diagnostics explicitly remain `local_unissued`;
- the candidate source and its ModelSpec were never reachable from `origin/main`;
- the public project card reports zero formal training receipts.

Therefore the older hashes identify a local exploratory candidate, not an issued
or accepted public `0.2.0` contract. They are not migrated, rewritten, or promoted
by this anchor.

## Treatment of old local artifacts

- Preserve them as historical `local_unissued` diagnostics.
- Do not cite them as evidence for the public anchor.
- Do not reuse their preregistrations or results under the new ModelSpec hash.
- Regenerate any future evaluation registration from the merged public ModelSpec.

After this anchor is merged, the tuple of contract id, contract version, entrypoint
hash, semantic source-tree hash, and ModelSpec hash is immutable. Any later semantic
source change requires a new contract version; it must not repoint `0.2.0` in place.
