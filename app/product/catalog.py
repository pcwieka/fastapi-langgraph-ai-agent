"""Canonical product catalog loader.

The catalog (data/products.json) is the system of record for product data.
Both the runtime app and the embedding indexer read from here, so there is
exactly one source of truth. In production this would be a real database.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

# data/products.json lives at the repo root, two levels up from app/product/.
_DEFAULT_CATALOG_PATH = Path(__file__).resolve().parents[2] / "data" / "products.json"


def catalog_path() -> Path:
    """Resolve the catalog file path, allowing an env override for deployments."""
    override = os.environ.get("PRODUCTS_CATALOG_PATH")
    return Path(override) if override else _DEFAULT_CATALOG_PATH


@lru_cache(maxsize=1)
def load_products() -> list[dict]:
    """Load all products from the catalog file (cached after first read)."""
    with catalog_path().open(encoding="utf-8") as f:
        return json.load(f)


def products_by_id() -> dict[str, dict]:
    """Index the catalog by product id for fast lookups."""
    return {p["id"]: p for p in load_products()}
