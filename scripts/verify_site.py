#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "site" / "public"
REQUIRED = {
    "index.html",
    "styles.css",
    "app.js",
    "agent.json",
    "assets/tabu-mark.svg",
    "zh/index.html",
}
MARKER = "tabu-lab-site-v20260916-research"
PAGES = {
    "index.html": {
        "lang": '<html lang="en">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/"',
        "switch": 'href="zh/"',
    },
    "zh/index.html": {
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/"',
        "switch": 'href="../"',
    },
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
        if b"\x00" in data:
            fail(f"NUL byte in {path.relative_to(ROOT)}")
        if path.suffix in {".html", ".css", ".js", ".json", ".svg"}:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                fail(f"non-UTF-8 text file {path.relative_to(ROOT)}: {exc}")

    total_ids = 0
    total_refs = 0
    for page_ref, expected in PAGES.items():
        page = PUBLIC / page_ref
        html = page.read_text(encoding="utf-8")
        if MARKER not in html:
            fail(f"page-specific version marker missing in {page_ref}")
        if expected["lang"] not in html:
            fail(f"unexpected html lang in {page_ref}")
        if expected["canonical"] not in html:
            fail(f"canonical public URL missing in {page_ref}")
        if expected["switch"] not in html:
            fail(f"language switch missing in {page_ref}")
        if any(token in html for token in ("file://", "/Users/", "/home/cms/", "localhost:")):
            fail(f"local-only path leaked into {page_ref}")

        parser = SiteParser()
        parser.feed(html)
        if "TabU-lab" not in parser.title:
            fail(f"unexpected title in {page_ref}")
        if len(parser.ids) != len(set(parser.ids)):
            fail(f"duplicate HTML id in {page_ref}")

        ids = set(parser.ids)
        for attr, ref in parser.refs:
            parsed = urlparse(ref)
            if parsed.scheme in {"http", "https", "mailto", "data"} or ref.startswith("//"):
                continue
            if ref.startswith("#"):
                if ref[1:] not in ids:
                    fail(f"broken anchor {ref} in {page_ref}")
                continue
            target_ref = ref.split("#", 1)[0].split("?", 1)[0]
            if not target_ref or target_ref.startswith("/"):
                continue
            target = (page.parent / target_ref).resolve()
            if PUBLIC.resolve() not in target.parents and target != PUBLIC.resolve():
                fail(f"path escapes site root in {page_ref}: {ref}")
            if not target.exists():
                fail(f"missing local {attr} target in {page_ref}: {ref}")
        total_ids += len(parser.ids)
        total_refs += len(parser.refs)

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
        f"PASS: {len(REQUIRED)} required files, {len(PAGES)} language pages, "
        f"{total_ids} ids, {total_refs} references"
    )
    print(f"PASS: marker={MARKER}")
    print("PASS: agent-card counts match the catalog; evidence documents exist")


if __name__ == "__main__":
    main()
