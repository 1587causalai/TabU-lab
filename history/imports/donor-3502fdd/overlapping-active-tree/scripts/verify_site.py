#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "site" / "public"
PROJECTION_MANIFEST = ROOT / "site" / "projection-manifest.json"
REQUIRED = {
    "index.html",
    "styles.css",
    "app.js",
    "agent.json",
    "assets/tabu-mark.svg",
    "zh/index.html",
    "blog.css",
    "blog/index.html",
    "blog/introducing-tabu-lab/index.html",
    "blog/configure-tabu-model-with-yaml/index.html",
    "zh/blog/index.html",
    "zh/blog/introducing-tabu-lab/index.html",
    "zh/blog/configure-tabu-model-with-yaml/index.html",
    "catalog.json",
    "research.css",
    "research-projection.json",
    "models/index.html",
    "experiments/index.html",
    "runs/index.html",
    "evaluations/index.html",
    "lineage/index.html",
}
MARKER = "tabu-lab-site-v20260828-01"
RESEARCH_MARKER = "tabu-lab-research-index-v1"
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
    "blog/index.html": {
        "lang": '<html lang="en">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/blog/"',
        "switch": 'href="../zh/blog/"',
    },
    "blog/introducing-tabu-lab/index.html": {
        "lang": '<html lang="en">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/blog/introducing-tabu-lab/"',
        "switch": 'href="../../zh/blog/introducing-tabu-lab/"',
    },
    "blog/configure-tabu-model-with-yaml/index.html": {
        "lang": '<html lang="en">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/blog/configure-tabu-model-with-yaml/"',
        "switch": 'href="../../zh/blog/configure-tabu-model-with-yaml/"',
    },
    "zh/blog/index.html": {
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/blog/"',
        "switch": 'href="../../blog/"',
    },
    "zh/blog/introducing-tabu-lab/index.html": {
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/blog/introducing-tabu-lab/"',
        "switch": 'href="../../../blog/introducing-tabu-lab/"',
    },
    "zh/blog/configure-tabu-model-with-yaml/index.html": {
        "lang": '<html lang="zh-CN">',
        "canonical": 'href="https://research.wehub.us/tabu-lab/zh/blog/configure-tabu-model-with-yaml/"',
        "switch": 'href="../../../blog/configure-tabu-model-with-yaml/"',
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


def _required_research_route(kind: str, object_id: str) -> str | None:
    collection = {
        "model_contract": "models",
        "experiment": "experiments",
        "run": "runs",
        "eval_suite": "evaluations",
        "eval_result": "evaluations",
        "eval_comparison": "evaluations",
    }.get(kind)
    if collection is None:
        return None
    return f"{collection}/{object_id}/"


def verify_research_projection() -> tuple[int, int]:
    canonical = ROOT / "catalog.json"
    projected = PUBLIC / "catalog.json"
    if not canonical.is_file() or canonical.read_bytes() != projected.read_bytes():
        fail("public catalog must be byte-identical to canonical catalog.json")
    catalog = json.loads(canonical.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != "tabu.catalog-index.v1":
        fail("unsupported research catalog schema")
    projection = json.loads(
        (PUBLIC / "research-projection.json").read_text(encoding="utf-8")
    )
    if projection.get("schema_version") != "tabu.public-research-projection.v1":
        fail("unsupported research projection schema")
    if projection.get("catalog_source_tree_hash") != catalog.get("source_tree_hash"):
        fail("research projection is bound to another catalog source tree")
    if "not independent evidence" not in projection.get("claim_boundary", ""):
        fail("research projection claim boundary is missing")

    entries = catalog.get("entries", [])
    by_id = {entry["object_id"]: entry for entry in entries}
    if len(by_id) != len(entries):
        fail("research catalog has duplicate object ids")
    routes = projection.get("routes", [])
    route_by_key = {(item["kind"], item["object_id"], item["path"]): item for item in routes}
    expected_routes: list[tuple[str, str, str]] = []
    for entry in entries:
        object_id = entry["object_id"]
        if any(token in object_id for token in ("/", "\\", "..")):
            fail(f"unsafe catalog object id: {object_id}")
        lineage = f"lineage/{object_id}/"
        expected_routes.append((entry["kind"], object_id, lineage))
        detail = _required_research_route(entry["kind"], object_id)
        if detail is not None:
            expected_routes.append((entry["kind"], object_id, detail))
    if set(expected_routes) != set(route_by_key):
        fail("research projection route set does not match catalog objects")

    for key in expected_routes:
        item = route_by_key[key]
        entry = by_id[item["object_id"]]
        if item["object_hash"] != entry["object_hash"]:
            fail(f"route object hash drift: {item['path']}")
        if item["source_hash"] != entry["source_hash"]:
            fail(f"route source hash drift: {item['path']}")
        page = PUBLIC / item["path"] / "index.html"
        if not page.is_file():
            fail(f"missing research child route: {item['path']}")
        text = page.read_text(encoding="utf-8")
        if RESEARCH_MARKER not in text:
            fail(f"research marker missing: {item['path']}")
        for required in (entry["object_id"], entry["object_hash"], entry["source_hash"]):
            if required not in text:
                fail(f"research provenance missing from {item['path']}: {required}")
        forbidden = (
            "file://",
            "/Users/",
            "/home/",
            "/private/",
            "sys.executable",
            '"hostname"',
            '"username"',
        )
        if any(token in text for token in forbidden):
            fail(f"private environment detail leaked into {item['path']}")

    file_hashes = projection.get("files", {})
    for relative, expected in file_hashes.items():
        candidate = (PUBLIC / relative).resolve()
        if PUBLIC.resolve() not in candidate.parents:
            fail(f"research file escapes public root: {relative}")
        if not candidate.is_file():
            fail(f"missing research projection file: {relative}")
        if hashlib.sha256(candidate.read_bytes()).hexdigest() != expected:
            fail(f"stale research projection hash: {relative}")

    # Every published evaluation number must bind the exact successful formal
    # receipt selected by its typed producer, not merely share some lineage
    # edge with any receipted run.
    for entry in entries:
        if entry["kind"] != "eval_result":
            continue
        data = entry.get("data", {})
        producer = data.get("producer", {})
        if (
            producer.get("provenance") != "receipted_run"
            or producer.get("publication_eligible") is not True
        ):
            fail(f"evaluation result is not publication eligible: {entry['object_id']}")
        run_id = producer.get("run_id")
        receipt_hash = producer.get("receipt_sha256")
        receipt_pointer = producer.get("receipt_pointer")
        run_entry = by_id.get(run_id)
        if run_entry is None or run_entry.get("kind") != "run":
            fail(f"evaluation result producer run is missing: {entry['object_id']}")
        run_receipt = run_entry.get("data", {}).get("receipt")
        if not isinstance(run_receipt, dict) or (
            run_receipt.get("sha256") != receipt_hash
            or run_receipt.get("uri") != receipt_pointer
        ):
            fail(f"evaluation result producer receipt drift: {entry['object_id']}")
        matching_receipts = [
            candidate
            for candidate in entries
            if candidate.get("kind") == "receipt"
            and candidate.get("object_hash") == receipt_hash
            and candidate.get("source_path") == receipt_pointer
        ]
        if len(matching_receipts) != 1:
            fail(f"evaluation result receipt is not uniquely cataloged: {entry['object_id']}")
        receipt = matching_receipts[0].get("data", {})
        if (
            receipt.get("run_id") != run_id
            or receipt.get("status") != "succeeded"
            or receipt.get("metadata", {}).get("issuance_status") != "formal"
        ):
            fail(
                "evaluation result receipt is not successful formal evidence: "
                f"{entry['object_id']}"
            )
        artifact_id = data.get("adapter", {}).get("artifact_id")
        if artifact_id is not None:
            artifact = by_id.get(artifact_id)
            if artifact is None or artifact.get("kind") != "model_artifact":
                fail(f"evaluation result model artifact is missing: {entry['object_id']}")
            if entry["object_id"] not in artifact.get("data", {}).get(
                "evaluation_result_ids", []
            ):
                fail(f"model artifact does not bind evaluation result: {entry['object_id']}")
    return len(entries), len(expected_routes)


def main() -> None:
    if not PROJECTION_MANIFEST.exists():
        fail("missing site/projection-manifest.json")
    manifest = json.loads(PROJECTION_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "tabu-lab.site-projection.v1":
        fail("unsupported site projection manifest schema")
    hashed_paths: dict[str, str] = dict(manifest.get("public_files", {}))
    for binding in manifest.get("bindings", []):
        hashed_paths[binding["source"]] = binding["source_sha256"]
        hashed_paths[binding["projection"]] = binding["projection_sha256"]
    for relative_path, expected_sha256 in hashed_paths.items():
        candidate = (ROOT / relative_path).resolve()
        if ROOT.resolve() not in candidate.parents:
            fail(f"manifest path escapes repository: {relative_path}")
        if not candidate.is_file():
            fail(f"manifest path missing: {relative_path}")
        actual_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            fail(
                f"stale projection manifest for {relative_path}; "
                "run `python scripts/build_site_manifest.py`"
            )

    present = {
        str(path.relative_to(PUBLIC)) for path in PUBLIC.rglob("*") if path.is_file()
    }
    missing = sorted(REQUIRED - present)
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
    if card.get("project", {}).get("public_url") != "https://research.wehub.us/tabu-lab/":
        fail("agent card public URL mismatch")
    if card.get("narrative", {}).get("index") != "https://research.wehub.us/tabu-lab/blog/":
        fail("agent card narrative index mismatch")
    catalog_card = card.get("catalog", {})
    if catalog_card.get("index") != "https://research.wehub.us/tabu-lab/catalog.json":
        fail("agent card catalog URL mismatch")

    catalog_payload = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    formal_receipts = sum(
        entry.get("kind") == "receipt"
        and entry.get("data", {}).get("metadata", {}).get("issuance_status") == "formal"
        for entry in catalog_payload.get("entries", [])
    )
    accepted_claims = sum(
        entry.get("kind") == "claim"
        and entry.get("data", {}).get("status") == "accepted"
        for entry in catalog_payload.get("entries", [])
    )
    if card.get("status", {}).get("public_training_receipts") != formal_receipts:
        fail("agent card formal receipt count differs from canonical catalog")
    if card.get("status", {}).get("accepted_model_claims") != accepted_claims:
        fail("agent card accepted claim count differs from canonical catalog")

    research_entries, research_routes = verify_research_projection()

    print(
        f"PASS: {len(REQUIRED)} required files, {len(PAGES)} language pages, "
        f"{total_ids} ids, {total_refs} references"
    )
    print(f"PASS: marker={MARKER}")
    print(f"PASS: {len(hashed_paths)} canonical source/projection hashes")
    print(
        f"PASS: {research_entries} catalog objects, {research_routes} research child routes"
    )
    print("PASS: claim boundary and agent-card counts match the canonical catalog")


if __name__ == "__main__":
    main()
