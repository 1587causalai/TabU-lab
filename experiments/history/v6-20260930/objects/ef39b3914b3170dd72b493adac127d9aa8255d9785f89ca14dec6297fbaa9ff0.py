"""Artifact checks. All logic below runs with the Python standard library."""

import hashlib
import math
import json
import re
from collections import Counter
from pathlib import Path
from ..core.typed import dataset_features, validate_full_dataset
from ..core.validation import validate_table

FAMILIES = {
    "scm_mixed_v1": 48,
    "discoscm": 48,
    "scm_numeric_v0": 12,
    "sklearn_synthetic": 12,
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_path(root, name):
    p = Path(name)
    if p.is_absolute() or not p.parts or ".." in p.parts:
        raise ValueError(f"invalid relative path: {name}")
    result = (root / p).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"path escapes artifact: {name}")
    return result


def read_artifact(root):
    root = Path(root)
    members = {}

    def retain(name, expected=None):
        data = checked_path(root, name).read_bytes()
        if expected is not None and sha(data) != expected:
            raise ValueError(f"SHA-256 mismatch: {name}")
        members[name] = data
        return data

    manifest = json.loads(retain("manifest.json"))
    if manifest.get("schema") == "tfm-data.single-table.1":
        if manifest.get("outcome") != "table_frozen":
            raise ValueError("incomplete single table")
        result = validate_table(
            json.loads(retain("table.json", manifest["table_sha256"]))
        )
        return dict(outcome="table_checked", tables=1, **result), members
    if (
        manifest.get("schema") != "tfm-data.corpus.1"
        or manifest.get("outcome") != "corpus_frozen"
    ):
        raise ValueError("unknown or incomplete corpus")
    records = manifest["records"]
    count = manifest["count"]
    if count != len(records) or count < 1:
        raise ValueError("record count mismatch")
    families = dict(Counter(r["family"] for r in records))
    if manifest["recipe"] != "anchor120.1" or count != 120 or families != FAMILIES:
        raise ValueError("Anchor120 recipe composition mismatch")
    if manifest["family_counts"] != families:
        raise ValueError("family count mismatch")
    names = [r["dataset"] for r in records]
    if len(set(names)) != count or any(
        not re.fullmatch("[a-z0-9_]+", n) for n in names
    ):
        raise ValueError("invalid or duplicate dataset names")
    if not manifest["generator_files"]:
        raise ValueError("missing source identity")
    for name, digest in manifest["generator_files"].items():
        retain("generator-source/" + name, digest)
    aux = manifest["auxiliary_files"]
    if "requested.json" not in aux:
        raise ValueError("missing requested identity")
    for name, digest in aux.items():
        retain(name, digest)
    if (
        len(members.get("rejections.jsonl", b"").splitlines())
        != manifest["rejection_count"]
    ):
        raise ValueError("rejection count mismatch")
    requests = json.loads(members["requested.json"])
    if [e["dataset"] for e in requests] != names:
        raise ValueError("requested dataset mismatch")
    splits = Counter()
    for r in records:
        name = r["dataset"]
        raw = json.loads(retain(f"raw/{name}.json", r["raw_sha256"]))
        shape = validate_table(raw)
        data = json.loads(retain(f"data/{name}.json", r["data_sha256"]))
        if data["schema"] != "tabu.tar.typed-fit-table.1" or data["dataset"] != name:
            raise ValueError("typed identity mismatch")
        dataset_features(data)
        partition = validate_full_dataset(data, manifest["rows"])
        if shape["rows"] != manifest["rows"]:
            raise ValueError("raw row count mismatch")
        order = data["output_to_source_column"]
        if any(type(j) is not int for j in order) or sorted(order) != list(
            range(shape["columns"])
        ):
            raise ValueError("invalid column permutation")
        expected_features = [
            dict(
                kind=raw["table"]["column_types"][j]
                if raw["table"]["column_types"][j] in ("numeric", "ordinal")
                else "nominal",
                domain=[]
                if raw["table"]["column_types"][j] == "numeric"
                else [str(i) for i in range(raw["table"]["n_classes"][j])],
            )
            for j in order
        ]
        if data["features"] != expected_features:
            raise ValueError("raw/typed feature schema mismatch")
        if data["values"] != [
            [row[j] for j in order] for row in raw["table"]["values"]
        ]:
            raise ValueError("raw/typed values mismatch")
        digest = sha(
            json.dumps(
                data["values"], sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        )
        if digest != r["values_sha256"]:
            raise ValueError("values digest mismatch")
        if partition["test_rows"] != math.ceil(0.2 * manifest["rows"]):
            raise ValueError("reserved fraction mismatch")
        splits[f"{partition['train_rows']}/{partition['test_rows']}"] += 1
    return dict(
        outcome="corpus_checked",
        tables=count,
        rows_per_table=manifest["rows"],
        families=families,
        train_reserved_counts=dict(splits),
        rejected_attempts=manifest["rejection_count"],
        manifest_sha256=sha(members["manifest.json"]),
        evidence_status=manifest["status"],
    ), members


def check(path):
    root = Path(path)
    if root.is_file():
        return validate_table(json.loads(root.read_bytes()))
    if (root / "BUNDLE.json").exists():
        bundle = json.loads((root / "BUNDLE.json").read_bytes())
        if bundle["format"] != "tfm-data.bundle.1":
            raise ValueError("unknown bundle format")
        for name, digest in bundle["files"].items():
            if sha(checked_path(root, name).read_bytes()) != digest:
                raise ValueError(f"bundle SHA-256 mismatch: {name}")
        summary, members = read_artifact(root / "corpus")
        required = {"corpus/" + name for name in members} | {"check.py", "README.md"}
        if not required <= set(bundle["files"]):
            raise ValueError("bundle inventory incomplete")
        return dict(summary, bundle_files_checked=len(bundle["files"]))
    return read_artifact(root)[0]
