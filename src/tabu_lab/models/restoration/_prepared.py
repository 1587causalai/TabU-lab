"""Guard fixed episode tensors without reading device values on every replay."""

from dataclasses import fields, is_dataclass

from torch import Tensor


class TensorVersions:
    """Owned snapshots are read-only; reject ordinary in-place edits before reuse.

    Version reads are host metadata, not CUDA scalar reads. As elsewhere in
    PyTorch, bypassing version counters with .data or external storage writes is
    unsupported. Prepare outside inference_mode so tensors have version counters.
    """

    def __init__(self, *values):
        tensors = {}

        def visit(value):
            if isinstance(value, Tensor):
                tensors[id(value)] = value
            elif is_dataclass(value):
                for field in fields(value):
                    visit(getattr(value, field.name))
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item)

        for value in values:
            visit(value)
        self.versions = tuple((tensor, tensor._version) for tensor in tensors.values())

    def validate(self):
        if any(tensor._version != version for tensor, version in self.versions):
            raise ValueError("prepared episode was mutated; prepare a new snapshot")
