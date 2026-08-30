# YAML mathematics and deterministic TeX

The optional `ModelSpec.mathematics` block makes a model's mathematical narrative
machine-checkable without creating a second source of truth.

It contains:

- a short abstract and explicit Unit semantics;
- named notation with optional domains;
- ordered steps containing stable equation ids, authored LaTeX, meanings, and
  stage invariants;
- falsifiable invariants paired with the evidence expected to test them.

Ids are unique within notation, steps, equations, and invariants. Unknown fields
are rejected. `render_model_tex` fails closed when the block is absent, escapes all
human-facing prose, preserves formula LaTeX, and emits sections in declared order.

The YAML remains semantic authority; generated TeX is only a readable projection.
It is not an implementation, receipt, maturity promotion, or accepted claim.

This PR adds the schema and renderer but does not edit the existing
`tabu.cell.base@0.2.0` YAML. That contract's identity therefore stays unchanged.
