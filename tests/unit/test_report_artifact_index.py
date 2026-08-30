from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs" / "reports" / "local-artifact-index.json"


def test_report_artifact_index_is_portable_and_content_addressed() -> None:
    raw = INDEX.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert payload["schema_version"] == "tabu.report-local-artifact-index.v1"
    assert payload["status"] == "local_unissued"
    assert payload["availability"] == "artifacts_not_bundled_with_repository"
    assert all(marker not in raw for marker in ("/Users/", "/home/", ".local-runs"))

    artifact_ids: list[str] = []
    for report in payload["reports"]:
        assert (ROOT / report["report_path"]).is_file()
        assert report["artifacts"]
        for artifact in report["artifacts"]:
            artifact_ids.append(artifact["artifact_id"])
            assert re.fullmatch(r"[a-z0-9][a-z0-9.-]+", artifact["artifact_id"])
            assert re.fullmatch(r"[0-9a-f]{64}", artifact["sha256"])

    assert len(artifact_ids) == len(set(artifact_ids))
