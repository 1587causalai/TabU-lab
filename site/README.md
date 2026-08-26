# TabU-lab public site

This directory owns the source for the public TabU-lab entrance.

## Route

- Canonical URL: `https://research.wehub.us/tabu-lab/`
- Project source: `site/public/`
- dgx2 staging: `/home/cms/wehub-sites/research/tabu-lab/`
- dgx2 public root: `/var/www/research.wehub.us/tabu-lab/`

The page is a WeHub public research surface. It can summarize verified project state and link to receipts, but it does not turn a proposal, local run, or website deployment into model evidence.

## Verify

```bash
python3 scripts/verify_site.py
```

## Deploy

```bash
scripts/deploy_site.sh
```

The deployment script verifies the local projection, backs up any existing remote route, syncs only `site/public/`, and checks the page-specific marker on both staging and public roots.
