"""Immutable single-table outputs and deterministic, independently checked bundles."""

import gzip
import io
import json
import tarfile
from pathlib import Path
from .checker import check, read_artifact, sha
from .json import write_json
from ..core.validation import validate_table


def save_table(result, output):
    validate_table(result)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "table.json", result)
    write_json(
        root / "manifest.json",
        dict(
            schema="tfm-data.single-table.1",
            outcome="table_frozen",
            table_sha256=sha((root / "table.json").read_bytes()),
        ),
    )
    return check(root)


def portable_checker():
    """Assemble exact shared validation code, removing only package imports."""
    package = Path(__file__).resolve().parents[1]
    modules = ["core/types.py", "core/typed.py", "core/validation.py", "io/checker.py"]
    chunks = [
        "# TFM-Data portable checker; generated from shared validation modules.\n# Apache-2.0; derivative MIT sources have notices in the corpus snapshot.\n"
    ]
    for name in modules:
        lines = (package / name).read_text().splitlines()
        chunks.append(
            "\n".join(
                l
                for l in lines
                if not l.startswith("from .") and not l.startswith("from __future__")
            )
        )
    chunks.append(
        "if __name__ == '__main__':\n    import sys\n    print(json.dumps(check(sys.argv[1] if len(sys.argv)>1 else '.'),indent=2))\n"
    )
    return "\n\n".join(chunks).encode()


def pack(corpus, output):
    summary, payload = read_artifact(Path(corpus))
    members = {"corpus/" + name: data for name, data in payload.items()}
    members["check.py"] = portable_checker()
    members["README.md"] = b"""# TFM-Data portable data bundle

Run `python3 -I -S check.py .` using Python 3.11+; no installed packages are needed.
Read JSON with the standard library. A corpus has raw/*.json and data/*.json;
a single-table artifact has table.json. All paths are inside corpus/.
Typed data retains tabu.tar.typed-fit-table.1, with final target column and
complete train/test indices. Full target truth is present: consumers must apply
masks and hide query labels before model forward. Query and missing may overlap.
Hashes establish retained integrity, not external approval or model capability.
Corpus artifacts retain generator source and provenance in corpus/generator-source.
Single-table artifacts retain table.json and manifest.json; they do not include a
generator source snapshot. Both bundle kinds include license/provenance texts in licenses/.
"""
    # Also include licensing for single-table bundles, whose artifact has no source tree.
    resource = Path(__file__).resolve().parents[1] / "resources"
    for p in sorted(resource.iterdir()):
        if p.is_file() and p.suffix != ".py":
            members["licenses/" + p.name] = p.read_bytes()
    members["BUNDLE.json"] = (
        json.dumps(
            dict(
                format="tfm-data.bundle.1",
                files={n: sha(d) for n, d in sorted(members.items())},
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as stream,
        gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as gz,
    ):
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(members.items()):
                info = tarfile.TarInfo("tfm-data/" + name)
                info.size = len(data)
                info.mode = 0o644
                info.mtime = 0
                archive.addfile(info, io.BytesIO(data))
    return dict(
        summary,
        outcome="bundle_created",
        archive=str(output),
        archive_sha256=sha(output.read_bytes()),
        bytes=output.stat().st_size,
    )
