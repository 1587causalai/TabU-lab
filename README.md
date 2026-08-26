# TabU-lab

An open research lab for tabular foundation models, inspired by [Marin](https://github.com/marin-community/marin).

## Website

- Public entrance: https://research.wehub.us/tabu-lab/
- Chinese entrance: https://research.wehub.us/tabu-lab/zh/
- Static source: [`site/public/`](./site/public/)
- Machine-readable project card: https://research.wehub.us/tabu-lab/agent.json

The website is a public research surface, not a source of model or benchmark evidence. Experiment receipts remain under `experiments/` and consolidated findings under `docs/reports/`.

## Core values

- **Open development**: every experiment, config, curve and failed result is recorded as it happens.
- **Reproducible at small scale**: fixed splits, fixed seeds, explicit compute budget; every run produces a receipt.
- **Falsifiable gates**: each training step has a pass/kill signal written down before the run.

## Layout

- `experiments/` — one file/directory per experiment; hypothesis, gate, command, results, receipt
- `docs/reports/` — consolidated learnings and retrospectives
- `lib/` — model / data / evaluation code

## Related

- TabPFN v2 (Prior Labs): https://github.com/PriorLabs/TabPFN
- Marin: https://github.com/marin-community/marin
- WeHub Research: https://research.wehub.us/
