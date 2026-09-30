"""TFM-Data: synthetic table laws, data validation and portable corpora."""

__version__ = "0.1.0"
from .api import generate_table
from .corpus.freeze import generate_corpus
from .core.validation import validate_table
from .io.artifacts import check, pack, save_table

__all__ = [
    "generate_table",
    "generate_corpus",
    "validate_table",
    "check",
    "pack",
    "save_table",
]
