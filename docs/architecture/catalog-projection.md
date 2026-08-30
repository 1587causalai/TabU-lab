# Current catalog projection

The catalog is an index over canonical sources, not a second authority.

`build_catalog` currently reads only direct YAML files under `specs/models`,
requires byte-identical packaged copies, validates each file as a `ModelSpec`, and
sorts entries by stable identity. Each entry binds its repository-relative source,
version, public payload, and content hash. The index binds the ordered entries with
one `source_tree_hash`.

`render_catalog_json` and `render_catalog_html` are deterministic presentation
functions. Their fixed claim boundary states that a catalog projection is not
evidence or claim acceptance. This first consolidated index deliberately reports
zero formal receipts and zero accepted claims.

Experiments, evaluations, verification suites, formal authorization, and richer
lineage remain outside this PR. They can extend the catalog only after their own
canonical contracts are consolidated.
