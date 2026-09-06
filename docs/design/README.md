# TAR implementation design snapshot

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
