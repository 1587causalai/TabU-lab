# Implementation and curriculum designs

[Pretraining curriculum specification](pretraining-curriculum.tex)
([PDF](pretraining-curriculum.pdf)) is an independent consumer-side design:
invariant semantics, revisable empirical probes, and replaceable adapters.
Its [minimal Python interface](../tutorials/model-independent-curriculum.md)
binds existing frozen data and checks selection/provenance without starting training.
It does not replace the model designs below or turn historical results into a fixed pipeline.

## V5.4 restoration implementation

[V5.4 implementation and episode choices](restoration-v54.md) describes the
versioned Python and curriculum entry points, Small/Nano comparisons, and the
current supervised-row default with random-cell as the first alternative.
The linked [V5.4 TeX snapshot](TabU_V5p4_Unified_Composition.tex) preserves the
owner design used for this code version; its source path and digest are recorded
in the implementation note. V5.3 remains a distinct historical entry point.

## TAR implementation design snapshot

[tar-model-design.tex](tar-model-design.tex) is an exact, read-only snapshot of the
owner-maintained `TabU/latex/model-factory/TabU-TAR/model-design.tex`.
Its figures are embedded TikZ. [tar-defaults.json](tar-defaults.json) captures the
same Standard configuration. These files make the implementation contract readable
from this repository without requiring the parent research workspace.

Edit the owner-maintained source first. Refresh the snapshot only together with
its source manifest and ModelSpec binding; the source-parity tests detect drift.
This copy is an implementation reference, not an independently maintained design.
To compile it locally: `latexmk -xelatex -synctex=1 tar-model-design.tex`.

The [packaged ModelSpec](../../specs/models/tabu.tar.yaml) records the original
entrypoint digest and recursive source-closure digest. The former experimental
manifest did not contain the required source-closure fields. Their addition here
changes TAR checkpoint source identity even though core operator source and
mathematics are unchanged. Older experimental checkpoints require their original
archived source and ModelSpec; this integration does not rewrite old receipts or
bypass strict checkpoint loading. New saves bind the integrated identity.

Small and Medium are runtime size presets; Standard remains the configuration
represented by this design. Neither a committed design nor a successful component
test is a pretrained-model or generalization claim.
