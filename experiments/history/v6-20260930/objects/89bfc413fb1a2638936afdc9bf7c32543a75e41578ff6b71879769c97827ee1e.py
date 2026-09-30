"""Command line routing; generation algorithms live in tfm_data.generators."""

import argparse
import json
import sys
from pathlib import Path
from . import generate_table, generate_corpus, check, pack, save_table
from .generators.registry import list_generators


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tfm-data")
    sub = parser.add_subparsers(dest="action", required=True)
    sub.add_parser("list")
    table = sub.add_parser("table")
    table.add_argument("source")
    table.add_argument("--seed", type=int, required=True)
    table.add_argument("--rows", type=int, default=256)
    table.add_argument("--options", default="{}", help="source-specific JSON object")
    table.add_argument("--output", required=True, type=Path)
    corpus = sub.add_parser("corpus")
    corpus.add_argument("recipe", choices=["anchor120", "anchor120.1"])
    corpus.add_argument("--seed", type=int, default=20260907)
    corpus.add_argument("--rows", type=int, default=256)
    corpus.add_argument("--output", required=True, type=Path)
    sub.add_parser("check").add_argument("path", type=Path)
    bundle = sub.add_parser("pack")
    bundle.add_argument("--corpus", required=True, type=Path)
    bundle.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "list":
            result = dict(generators=list_generators(), recipes=["anchor120.1"])
        elif args.action == "table":
            if args.output.exists():
                raise FileExistsError(f"output exists: {args.output}")
            result = save_table(
                generate_table(
                    args.source,
                    seed=args.seed,
                    n_rows=args.rows,
                    options=json.loads(args.options),
                ),
                args.output,
            )
        elif args.action == "corpus":
            result = generate_corpus(
                args.recipe, seed=args.seed, n_rows=args.rows, output=args.output
            )
        elif args.action == "check":
            result = check(args.path)
        else:
            result = pack(args.corpus, args.output)
        if args.action == "corpus":
            result = {
                k: v
                for k, v in result.items()
                if k not in ("records", "generator_files")
            }
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"tfm-data: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
