"""Bounded Git-native catalog and public projection API."""

from .builder import (
    CatalogBuildError,
    build_catalog,
    check_catalog,
    load_catalog,
    render_catalog_json,
)
from .models import CatalogEntry, CatalogIndex, CatalogObjectKind
from .projection import render_catalog_html

__all__ = [
    "CatalogBuildError",
    "CatalogEntry",
    "CatalogIndex",
    "CatalogObjectKind",
    "build_catalog",
    "check_catalog",
    "load_catalog",
    "render_catalog_html",
    "render_catalog_json",
]
