"""Deterministic public research pages derived from one catalog index.

The renderer is intentionally presentation-only.  It never infers maturity, edits a
catalog record, or turns a local object into evidence.  Every page exposes the exact
catalog/source identities from which its human-readable fields were rendered.
"""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import quote

from tabu_lab.catalog import CatalogEntry, CatalogIndex, CatalogObjectKind

PUBLIC_MARKER = "tabu-lab-research-index-v1"
PROJECTION_SCHEMA = "tabu.public-research-projection.v1"
CLAIM_BOUNDARY = "catalog projection; not independent evidence or maturity promotion"

_COLLECTIONS: tuple[tuple[str, str, frozenset[CatalogObjectKind]], ...] = (
    ("models", "Models", frozenset({CatalogObjectKind.MODEL_CONTRACT})),
    ("experiments", "Experiments", frozenset({CatalogObjectKind.EXPERIMENT})),
    ("runs", "Runs", frozenset({CatalogObjectKind.RUN})),
    (
        "evaluations",
        "Evaluations",
        frozenset(
            {
                CatalogObjectKind.EVAL_SUITE,
                CatalogObjectKind.EVAL_RESULT,
                CatalogObjectKind.EVAL_COMPARISON,
            }
        ),
    ),
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(value: object, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ) + "\n"


def _safe_segment(value: str) -> str:
    """Return a URL segment without changing the catalog identity displayed on-page."""

    return quote(value, safe=".@_-")


def _entry_route(entry: CatalogEntry) -> str | None:
    collection = {
        CatalogObjectKind.MODEL_CONTRACT: "models",
        CatalogObjectKind.EXPERIMENT: "experiments",
        CatalogObjectKind.RUN: "runs",
        CatalogObjectKind.EVAL_SUITE: "evaluations",
        CatalogObjectKind.EVAL_RESULT: "evaluations",
        CatalogObjectKind.EVAL_COMPARISON: "evaluations",
    }.get(entry.kind)
    if collection is None:
        return None
    return f"{collection}/{_safe_segment(entry.object_id)}/"


def _lineage_route(entry: CatalogEntry) -> str:
    return f"lineage/{_safe_segment(entry.object_id)}/"


def _page(
    *,
    title: str,
    eyebrow: str,
    body: str,
    root_prefix: str,
    canonical_path: str,
) -> str:
    escaped_title = html.escape(title)
    canonical_url = f"https://research.wehub.us/tabu-lab/{canonical_path}"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title} · TabU-lab</title>
  <meta name="tabu-claim-boundary" content="catalog-projection-not-evidence">
  <link rel="canonical" href="{html.escape(canonical_url)}">
  <link rel="icon" href="{root_prefix}assets/tabu-mark.svg" type="image/svg+xml">
  <link rel="stylesheet" href="{root_prefix}research.css">
</head>
<body data-research-index-version="{PUBLIC_MARKER}">
  <header class="research-header">
    <a class="research-brand" href="{root_prefix}">TabU-lab</a>
    <nav aria-label="Research catalog">
      <a href="{root_prefix}models/">Models</a>
      <a href="{root_prefix}experiments/">Experiments</a>
      <a href="{root_prefix}runs/">Runs</a>
      <a href="{root_prefix}evaluations/">Evaluations</a>
      <a href="{root_prefix}lineage/">Lineage</a>
    </nav>
  </header>
  <main>
    <p class="eyebrow">{html.escape(eyebrow)}</p>
    <h1>{escaped_title}</h1>
    <div class="claim-boundary"><strong>Claim boundary</strong> {CLAIM_BOUNDARY}.</div>
{body}
  </main>
  <footer>
    Generated from <a href="{root_prefix}catalog.json">catalog.json</a>.
    Blog prose cannot override this state.
  </footer>
</body>
</html>
"""


def _immutable_source_url(catalog: CatalogIndex, entry: CatalogEntry) -> str | None:
    """Return an exact-revision source URL or fail closed without a link."""

    revision = catalog.source_revision
    if revision is None:
        return None
    return (
        f"{revision.repository_uri}/blob/{revision.commit}/"
        f"{quote(entry.source_path, safe='/')}"
    )


def _source_block(catalog: CatalogIndex, entry: CatalogEntry, *, root_prefix: str) -> str:
    source_url = _immutable_source_url(catalog, entry)
    if source_url is None:
        source_identity = (
            f"<code>{html.escape(entry.source_path)}</code><br>"
            '<span class="source-link-unavailable">Exact public revision not recorded; '
            f"source SHA-256 <code>{entry.source_hash}</code>.</span>"
        )
    else:
        revision = catalog.source_revision
        if revision is None:  # pragma: no cover - narrowed by _immutable_source_url
            raise AssertionError("source revision disappeared during rendering")
        source_identity = (
            f'<a href="{html.escape(source_url)}">{html.escape(entry.source_path)}</a><br>'
            f'<span>Git commit <code>{revision.commit}</code>.</span>'
        )
    status = entry.status if entry.status is not None else "not-declared"
    return f"""
<dl class="identity-grid">
  <div><dt>Kind</dt><dd><code>{html.escape(entry.kind.value)}</code></dd></div>
  <div><dt>Status</dt><dd><code>{html.escape(status)}</code></dd></div>
  <div><dt>Schema</dt><dd><code>{html.escape(entry.object_schema_version)}</code></dd></div>
  <div><dt>Object SHA-256</dt><dd><code>{entry.object_hash}</code></dd></div>
  <div><dt>Source SHA-256</dt><dd><code>{entry.source_hash}</code></dd></div>
  <div><dt>Canonical source</dt><dd>{source_identity}</dd></div>
  <div><dt>Lineage</dt><dd>
    <a href="{root_prefix}lineage/{_safe_segment(entry.object_id)}/">inspect dependencies</a>
  </dd></div>
</dl>
"""


def _related_artifacts(catalog: CatalogIndex, contract_id: str) -> tuple[CatalogEntry, ...]:
    return tuple(
        entry
        for entry in catalog.entries
        if entry.kind is CatalogObjectKind.MODEL_ARTIFACT
        and entry.data.get("contract_id") == contract_id
    )


def _related_list(title: str, entries: Iterable[CatalogEntry], *, root_prefix: str) -> str:
    rows = []
    for entry in entries:
        route = _entry_route(entry) or _lineage_route(entry)
        rows.append(
            "<li>"
            f'<a href="{root_prefix}{route}">{html.escape(entry.object_id)}</a>'
            f" <code>{html.escape(entry.kind.value)}</code>"
            f" <span>{html.escape(entry.status or 'not-declared')}</span>"
            "</li>"
        )
    content = "".join(rows) if rows else "<li>No cataloged objects.</li>"
    return f'<section><h2>{html.escape(title)}</h2><ul class="object-list">{content}</ul></section>'


def _record_page(catalog: CatalogIndex, entry: CatalogEntry) -> str:
    related = ""
    if entry.kind is CatalogObjectKind.MODEL_CONTRACT:
        related = _related_list(
            "Trained artifacts",
            _related_artifacts(catalog, entry.object_id),
            root_prefix="../../",
        )
    data = html.escape(_json(entry.data, pretty=True))
    body = (
        _source_block(catalog, entry, root_prefix="../../")
        + related
        + "<section><h2>Canonical record</h2>"
        + "<p>All displayed values below come directly from the "
        + "content-addressed catalog object.</p>"
        + f'<pre data-object-hash="{entry.object_hash}">{data}</pre></section>'
    )
    route = _entry_route(entry)
    if route is None:  # pragma: no cover - guarded by caller
        raise ValueError(f"entry has no record page: {entry.kind.value}")
    return _page(
        title=entry.object_id,
        eyebrow=entry.kind.value.replace("_", " "),
        body=body,
        root_prefix="../../",
        canonical_path=route,
    )


def _lineage_page(catalog: CatalogIndex, entry: CatalogEntry) -> str:
    edges = [
        edge
        for edge in catalog.lineage
        if edge.source.object_id == entry.object_id or edge.target.object_id == entry.object_id
    ]
    rows: list[str] = []
    for edge in edges:
        source = edge.source.object_id
        target = edge.target.object_id
        other = target if source == entry.object_id else source
        rows.append(
            "<li>"
            f"<code>{html.escape(source)}</code> "
            f"<strong>{html.escape(edge.relation.value)}</strong> "
            f"<code>{html.escape(target)}</code> "
            f'<a href="../../lineage/{_safe_segment(other)}/">inspect</a>'
            "</li>"
        )
    edge_list = "".join(rows) if rows else "<li>No typed lineage edges recorded.</li>"
    body = (
        _source_block(catalog, entry, root_prefix="../../")
        + f'<section><h2>Typed lineage</h2><ul class="edge-list">{edge_list}</ul></section>'
    )
    return _page(
        title=f"Lineage · {entry.object_id}",
        eyebrow="typed dependency graph",
        body=body,
        root_prefix="../../",
        canonical_path=_lineage_route(entry),
    )


def _index_page(
    *,
    catalog: CatalogIndex,
    slug: str,
    title: str,
    entries: tuple[CatalogEntry, ...],
    lineage_only: bool = False,
) -> str:
    cards: list[str] = []
    for entry in entries:
        route = (
            _lineage_route(entry)
            if lineage_only
            else (_entry_route(entry) or _lineage_route(entry))
        )
        cards.append(
            '<article class="object-card">'
            f"<p>{html.escape(entry.kind.value)}</p>"
            f'<h2><a href="../{route}">{html.escape(entry.object_id)}</a></h2>'
            f"<span>{html.escape(entry.status or 'not-declared')}</span>"
            f"<code>{entry.object_hash}</code>"
            "</article>"
        )
    empty = '<p class="empty-state">No canonical objects in this collection.</p>'
    body = (
        f'<p class="catalog-summary">Catalog <code>{catalog.catalog_hash}</code> · '
        f"{len(entries)} object(s).</p>"
        + (f'<section class="card-grid">{"".join(cards)}</section>' if cards else empty)
    )
    return _page(
        title=title,
        eyebrow="Git-native research catalog",
        body=body,
        root_prefix="../",
        canonical_path=f"{slug}/",
    )


def _projection_payload(catalog: CatalogIndex, files: dict[str, bytes]) -> dict[str, object]:
    routes = []
    for entry in catalog.entries:
        for route in filter(None, (_entry_route(entry), _lineage_route(entry))):
            routes.append(
                {
                    "kind": entry.kind.value,
                    "object_hash": entry.object_hash,
                    "object_id": entry.object_id,
                    "path": route,
                    "source_hash": entry.source_hash,
                    "source_path": entry.source_path,
                }
            )
    return {
        "schema_version": PROJECTION_SCHEMA,
        "catalog_hash": catalog.catalog_hash,
        "catalog_source_tree_hash": catalog.source_tree_hash,
        "catalog_source_revision": (
            catalog.source_revision.model_dump(mode="json")
            if catalog.source_revision is not None
            else None
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "routes": sorted(routes, key=lambda item: (str(item["path"]), str(item["object_id"]))),
        "files": {
            path: _sha256(data)
            for path, data in sorted(files.items())
        },
    }


def render_public_projection(catalog: CatalogIndex) -> dict[str, bytes]:
    """Render a complete projection in memory without mutating the source catalog."""

    files: dict[str, bytes] = {
        "catalog.json": _json(catalog.model_dump(mode="json")).encode(),
    }
    for slug, title, kinds in _COLLECTIONS:
        entries = tuple(entry for entry in catalog.entries if entry.kind in kinds)
        files[f"{slug}/index.html"] = _index_page(
            catalog=catalog,
            slug=slug,
            title=title,
            entries=entries,
        ).encode()
        for entry in entries:
            route = _entry_route(entry)
            if route is not None:
                files[f"{route}index.html"] = _record_page(catalog, entry).encode()

    files["lineage/index.html"] = _index_page(
        catalog=catalog,
        slug="lineage",
        title="Lineage",
        entries=catalog.entries,
        lineage_only=True,
    ).encode()
    for entry in catalog.entries:
        files[f"{_lineage_route(entry)}index.html"] = _lineage_page(catalog, entry).encode()

    projection = _projection_payload(catalog, files)
    files["research-projection.json"] = _json(projection, pretty=True).encode()
    return files


def _read_previous_files(public_root: Path) -> set[str]:
    manifest = public_root / "research-projection.json"
    if not manifest.is_file():
        return set()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PROJECTION_SCHEMA:
        raise ValueError("unsupported research projection manifest")
    return set(payload.get("files", {})) | {"research-projection.json"}


def write_public_projection(catalog: CatalogIndex, public_root: str | Path) -> tuple[Path, ...]:
    """Write generated files and remove only paths owned by the prior manifest."""

    root = Path(public_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = render_public_projection(catalog)
    stale = _read_previous_files(root) - set(files)
    for relative in sorted(stale):
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError(f"generated path escapes public root: {relative}")
        if candidate.is_file():
            candidate.unlink()
    written: list[Path] = []
    for relative, data in sorted(files.items()):
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ValueError(f"generated path escapes public root: {relative}")
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_bytes(data)
        written.append(candidate)
    return tuple(written)


def check_public_projection(catalog: CatalogIndex, public_root: str | Path) -> tuple[str, ...]:
    """Return deterministic projection drift without modifying checked-in files."""

    root = Path(public_root).resolve()
    expected = render_public_projection(catalog)
    issues: list[str] = []
    previous = _read_previous_files(root)
    expected_paths = set(expected)
    for relative in sorted(expected_paths):
        candidate = root / relative
        if not candidate.is_file():
            issues.append(f"missing generated public file: {relative}")
        elif candidate.read_bytes() != expected[relative]:
            issues.append(f"stale generated public file: {relative}")
    for relative in sorted(previous - expected_paths):
        if (root / relative).exists():
            issues.append(f"stale generated public route: {relative}")
    return tuple(issues)
