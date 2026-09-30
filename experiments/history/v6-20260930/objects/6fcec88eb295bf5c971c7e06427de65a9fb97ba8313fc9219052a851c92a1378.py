"""Presentation-only projection from one validated catalog index."""

from __future__ import annotations

import html

from .models import CatalogIndex


def render_catalog_html(catalog: CatalogIndex) -> str:
    items = "\n".join(
        "<li><code>"
        + html.escape(entry.object_id)
        + "@"
        + html.escape(entry.version)
        + "</code> — "
        + html.escape(entry.source_path)
        + "</li>"
        for entry in catalog.entries
    )
    return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><title>TabU-lab catalog</title></head>
<body>
<h1>TabU-lab catalog</h1>
<p><strong>Claim boundary:</strong> {html.escape(catalog.claim_boundary)}.</p>
<p>Formal receipts: {catalog.formal_receipt_count};
accepted claims: {catalog.accepted_claim_count}.</p>
<ul>
{items}
</ul>
<p>Source tree: <code>{catalog.source_tree_hash}</code></p>
</body>
</html>
"""


__all__ = ["render_catalog_html"]
