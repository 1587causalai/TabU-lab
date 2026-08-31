# Evolution impact projections

These JSON files are deterministic query projections generated from
`specs/evolution/` by `scripts/build_evolution_impact_reports.py`.

They exercise model-mathematics, generator/mixture, component-graph, and
evaluation-protocol changes. The `query-*-v3-mainline.json` projections bind
the actual `1.1.0` v2-pilot to `1.2.0` v3-scale transition. They are not
training receipts, formal evidence, or accepted research claims. Regenerate
them after adding a new versioned manifest; never edit a report in place.
