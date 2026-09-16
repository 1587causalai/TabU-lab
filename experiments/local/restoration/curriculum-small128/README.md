# Restoration Small-128 curriculum

`preregistration.yaml` is the frozen three-stage recipe for the restoration
model.  The `curriculum-fit` command accepts a `--corpus-root` containing four
directories named `old120`, `recent120`, `old3`, and `new9`; each directory is
the corresponding frozen historical corpus with its `preregistration.yaml` and
`data/` directory.  The archived historical corpora are deliberately not
regenerated or downloaded by the runner.

Plan the run with:

```text
tabu-lab restoration curriculum-fit \
  --preregistration experiments/local/restoration/curriculum-small128/preregistration.yaml \
  --corpus-root /path/to/frozen-curriculum-corpora \
  --output-root /path/to/new-attempt
```

Add `--execute` only after the CUDA size preflight and resource gate pass.
