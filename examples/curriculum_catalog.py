"""Read-only catalog audit; no data generation, optimizer update or training.

PYTHONPATH=src python examples/curriculum_catalog.py --old-corpus PATH --new-corpus PATH
"""

import argparse
import json

from tabu_lab.curriculum import bind_legacy_corpus, reference_stage, require_disjoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-corpus", required=True)
    parser.add_argument("--new-corpus", required=True)
    parser.add_argument("--separation", choices=("table", "world"), default="table")
    args = parser.parse_args()
    old = bind_legacy_corpus(args.old_corpus, cohort="old120")
    new = bind_legacy_corpus(args.new_corpus, cohort="new120")
    shared = reference_stage("multi120", old)
    continual = reference_stage("continual120", new, previous=old,
                                separation_level=args.separation)
    print(json.dumps({
        "status": "verified_metadata_only", "training_started": False,
        "multi120_selection_sha256": shared.manifest()["sha256"],
        "continual120_selection_sha256": continual.manifest()["sha256"],
        "separation": require_disjoint(old, new, level=args.separation),
        "source_manifests": [str(old[0].provenance.manifest_path),
                             str(new[0].provenance.manifest_path)],
        "source_sha256": [old[0].provenance.manifest_sha256,
                          new[0].provenance.manifest_sha256],
    }, indent=2))


if __name__ == "__main__":
    main()
