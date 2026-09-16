# TabU-lab public site

This directory owns the source for the public TabU-lab entrance.

## Route

- Canonical URL: `https://research.wehub.us/tabu-lab/`
- Chinese URL: `https://research.wehub.us/tabu-lab/zh/`
- Project source: `site/public/`
- dgx2 staging: `/home/cms/wehub-sites/research/tabu-lab/`
- dgx2 public root: `/var/www/research.wehub.us/tabu-lab/`

The page is a WeHub public research surface. It can summarize verified project state and link to receipts, but it does not turn a proposal, local run, or website deployment into model evidence.

## Research narrative

The overview follows the shared [research and support brief](../docs/research-support.md):
scientific question, existing evidence, remaining uncertainty, bounded work and
resources, inspectable results, and value for other researchers. Research grants,
compute, coding tools, and partnerships use this same argument; proposal-specific
budgets and agreements live outside the general project overview.

The English and Chinese homepages are directly authored in `site/public/index.html`
and `site/public/zh/index.html`. Update both together with `agent.json` when changing
the overview. They link to dated evidence records and distinguish the restoration
implementation from historical TAR experiments. The status card's formal-receipt
and accepted-claim counts must match the existing generated catalog.

`scripts/build_public_catalog.py` owns the catalog and `models/` projection.
Narrative edits do not change those research records. `verify_site.py` checks
local assets, anchors, language routes, catalog-count consistency, and the status
card's evidence links. The shared page marker is
`tabu-lab-site-v20260916-research`.

## Verify

```bash
python3 scripts/verify_site.py
```

## Deploy

```bash
scripts/deploy_site.sh
```

The deployment script verifies the local projection, backs up any existing remote route, syncs only `site/public/`, installs the public copy with the host's established credential-backed `sudo` route, and checks the page-specific marker on both staging and public roots. It reads `SSH_PASS_CMS` from the environment or `~/.openclaw/.env` without printing it.
