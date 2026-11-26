#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "site" / "public"
PUBLIC_PREFIX = "/tabu-lab/"
PAGES = {
    "index.html": {
        "version": 'data-site-version="tabu-lab-site-v20261024-retrospective"',
        "lang": '<html lang="en">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/"',
        "switch": 'href="zh/"',
    },
    "zh/index.html": {
        "version": 'data-page-revision="tabu-lab-zh-home-v20261126.2"',
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/"',
        "switch": 'href="../"',
    },
    "zh/experiments/index.html": {
        "version": 'data-page-revision="tabu-lab-zh-experiments-v20261121"',
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/experiments/"',
    },
    "zh/support/index.html": {
        "version": 'data-page-revision="tabu-lab-zh-support-v20261120"',
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/support/"',
    },
}
REQUIRED = {
    *PAGES,
    "styles.css",
    "app.js",
    "agent.json",
    "assets/tabu-mark.svg",
    "zh/research.css",
}


class SiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.refs: list[tuple[str, str]] = []
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.append(values["id"] or "")
        for attr in ("href", "src"):
            if values.get(attr):
                self.refs.append((attr, values[attr] or ""))
        if tag == "title":
            self._in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    missing = sorted(
        REQUIRED
        - {str(path.relative_to(PUBLIC)) for path in PUBLIC.rglob("*") if path.is_file()}
    )
    if missing:
        fail(f"missing required files: {', '.join(missing)}")

    for path in PUBLIC.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if path.suffix == ".png":
            if not data.startswith(b"\x89PNG\r\n\x1a\n"):
                fail(f"invalid PNG signature in {path.relative_to(ROOT)}")
            continue
        if b"\x00" in data:
            fail(f"NUL byte in {path.relative_to(ROOT)}")
        if path.suffix in {".html", ".css", ".js", ".json", ".svg"}:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                fail(f"non-UTF-8 text file {path.relative_to(ROOT)}: {exc}")

    pages = {}
    for page in sorted(PUBLIC.rglob("*.html")):
        page_ref = page.relative_to(PUBLIC).as_posix()
        html = page.read_text(encoding="utf-8")
        for field, expected in PAGES.get(page_ref, {}).items():
            if expected not in html:
                fail(f"missing or unexpected {field} in {page_ref}")
        if any(token in html for token in ("file://", "/Users/", "/home/cms/", "localhost:")):
            fail(f"local-only path leaked into {page_ref}")

        parser = SiteParser()
        parser.feed(html)
        if "TabU-lab" not in parser.title:
            fail(f"unexpected title in {page_ref}")
        if len(parser.ids) != len(set(parser.ids)):
            fail(f"duplicate HTML id in {page_ref}")
        pages[page] = parser

    for page, parser in pages.items():
        page_ref = page.relative_to(PUBLIC).as_posix()
        for attr, ref in parser.refs:
            parsed = urlparse(ref)
            if parsed.scheme in {"http", "https", "mailto", "data"} or ref.startswith("//"):
                continue
            path = unquote(parsed.path)
            if path.startswith(PUBLIC_PREFIX):
                target = (PUBLIC / path.removeprefix(PUBLIC_PREFIX)).resolve()
            elif path.startswith("/"):
                continue
            else:
                target = (page.parent / path).resolve() if path else page
            if not target.is_relative_to(PUBLIC):
                fail(f"path escapes site root in {page_ref}: {ref}")
            if target.is_dir():
                target /= "index.html"
            if not target.is_file():
                fail(f"missing local {attr} target in {page_ref}: {ref}")
            if parsed.fragment and target in pages and unquote(parsed.fragment) not in pages[target].ids:
                fail(f"broken anchor {ref} in {page_ref}")

    card = json.loads((PUBLIC / "agent.json").read_text(encoding="utf-8"))
    catalog = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    public_catalog = json.loads((PUBLIC / "catalog.json").read_text(encoding="utf-8"))
    if catalog != public_catalog:
        fail("public catalog does not match the repository catalog")
    status = card.get("status", {})
    for card_field, catalog_field in (
        ("formal_receipt_count", "formal_receipt_count"),
        ("accepted_model_claims", "accepted_claim_count"),
    ):
        count = catalog.get(catalog_field)
        if type(count) is not int or count < 0 or status.get(card_field) != count:
            fail(f"agent card {card_field} disagrees with the research catalog")
    for record in card.get("evidence", []):
        prefix = "https://github.com/1587causalai/TabU-lab/blob/main/"
        url = record.get("url", "")
        if not url.startswith(prefix):
            fail("evidence link must identify a public repository document")
        evidence = (ROOT / url.removeprefix(prefix)).resolve()
        if ROOT.resolve() not in evidence.parents or not evidence.is_file():
            fail(f"missing local source for evidence link: {url}")
    if card.get("project", {}).get("public_url") != "https://research.wehub.us/tabu-lab/":
        fail("agent card public URL mismatch")

    print(
        f"PASS: {len(REQUIRED)} required files, {len(pages)} HTML pages, "
        f"{sum(len(page.ids) for page in pages.values())} ids, "
        f"{sum(len(page.refs) for page in pages.values())} references"
    )
    print("PASS: agent-card counts match the catalog; evidence documents exist")


if __name__ == "__main__":
    main()
