"""Reuse the prior paired Puma-like data with the DGX2 H8 parent architecture."""

import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
parent = json.loads((root / "parent-manifest.json").read_text())
for arm in ("signal6", "full32"):
    source = json.loads((root.parent / "puma-synthetic-probe-20260927" / "receipts" /
                         f"{arm}-manifest.json").read_text())
    source["experiment_id"] = f"v6-sparse-relevance-h8-{arm}-20260927"
    source["description"] = "H8 paired V5.5/V6 sparse relevance 150-update screen"
    source["model"] = parent["model"]
    source["optimizer"] = parent["optimizer"]
    data = root / "data" / f"{arm}.json"
    source["tables"][0]["sha256"] = hashlib.sha256(data.read_bytes()).hexdigest()
    source["stages"][0]["max_updates"] = 150
    source["stages"][0]["max_seconds"] = 3600
    source["stages"][0]["evaluate_every"] = 150
    source["stages"][0]["checkpoint_every"] = 150
    target = root / "manifests" / f"{arm}.json"
    target.write_text(json.dumps(source, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(arm, source["tables"][0]["sha256"])
