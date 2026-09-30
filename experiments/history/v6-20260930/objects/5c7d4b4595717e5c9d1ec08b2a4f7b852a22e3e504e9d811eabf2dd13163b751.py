"""Opt-in bridge to the existing V5.3 builder, without a second sampler."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from .catalog import BoundTable
from .episodes import EpisodeRequest


class V53EpisodeFactory:
    def __init__(self, *, device="cpu", epsilon=1e-6, codec_version="unit_gaussian_v2"):
        # Keep construction Torch-free; the builder also validates this identity.
        if codec_version not in (
            "unit_gaussian_v2", "constant_weight_v1", "unit_gaussian_v1", "legacy_v53",
        ):
            raise ValueError("unknown V5.3 codec_version")
        self.device = device
        self.epsilon = epsilon
        self.codec_version = codec_version

    def __call__(self, table: BoundTable, request: EpisodeRequest) -> tuple:
        if type(request) is not EpisodeRequest:
            raise ValueError("V5.3 adapter does not implement extended request parameters")
        if request.kind not in ("random_cell", "supervised_row") or request.fraction is None:
            raise ValueError("V5.3 adapter requires random_cell/supervised_row and a fraction")
        if request.partition != "train" and (
            request.kind != "supervised_row" or request.window_rows is not None
        ):
            raise ValueError("V5.3 reserved evaluation requires full-context supervised_row")
        # Keep even importing the model-specific factory cheap; Torch and model
        # contracts are loaded only when the caller constructs an episode.
        from tabu_lab.curriculum_v53.data import build_episode, load_table
        from tabu_lab.curriculum_v53.protocol import _source_digest

        table.verify()
        entry = {"id": table.table_id, "path": table.path.name,
                 "sha256": table.sha256, "cohort": table.cohort, "kind": table.kind,
                 "window_rows": request.window_rows}
        if request.target_column is not None:
            entry["target_column"] = request.target_column
        loaded = load_table(entry, table.path.parent)
        inputs, targets, truth, audit = build_episode(
            loaded, {"kind": request.kind, "fraction": request.fraction},
            request.index, asdict(request.seeds), self.device,
            evaluation=request.purpose != "training", partition=request.partition,
            epsilon=self.epsilon, codec_version=self.codec_version,
        )
        files = dict(_source_digest()["files"])
        for path in sorted(Path(__file__).parent.glob("*.py")):
            files[f"curriculum/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        source_digest = hashlib.sha256(json.dumps(
            files, sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        return inputs, targets, truth, {
            **audit, "selection_schema": "tabu.curriculum.selection.v1",
            "episode_schema": "tabu.curriculum.episode-audit.v1",
            "data_sha256": table.sha256, "split_sha256": table.split_sha256,
            "purpose": request.purpose,
            "episode_request": asdict(request),
            "episode_source": {"sha256": source_digest, "files": files},
        }
