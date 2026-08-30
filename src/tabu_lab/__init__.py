"""TabU-lab contract-first reference implementation."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tabu-lab")
except PackageNotFoundError:  # pragma: no cover - editable source tree fallback
    __version__ = "0.1.0.dev0"

__all__ = ["__version__"]
