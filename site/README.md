# TabU-lab public site

The public TabU-lab site has an [English homepage](https://research.wehub.us/tabu-lab/),
a [Chinese homepage](https://research.wehub.us/tabu-lab/zh/), and a
[Chinese research support page](https://research.wehub.us/tabu-lab/zh/support/).

## Source

- [English homepage](public/index.html)
- [Chinese homepage](public/zh/index.html), [support page](public/zh/support/index.html),
  and their [shared stylesheet](public/zh/research.css)
- [Experiment index](public/experiments/index.html) and [research notes](public/blog/index.html).
  The checked-in reports preserve the published wording and dates of the pages
  linked by the homepages.

Keep shared research status and evidence links consistent across the English
and Chinese homepages. The [research support brief](../docs/research-support.md)
records the collaboration scope. [`build_public_catalog.py`](../scripts/build_public_catalog.py)
generates the catalog and model index. The counts in `public/agent.json` follow
the generated catalog.

## Verify

```bash
python3 scripts/verify_site.py
```

This checks page metadata, local files and anchors, and the project card's
catalog counts and evidence links. Mount `site/public/` at `/tabu-lab/` in a
browser preview to match the production route.

## Production updates

The managed WeHub workflow takes an explicit list of paths relative to
`site/public/`. It backs up each target and verifies the updated origin files.
The production site also serves independently published reports. Verify each
changed public URL after release.
