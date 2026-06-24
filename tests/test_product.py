from unittest.mock import patch

from app.product.catalog import load_products, products_by_id
from app.product.repository import ChromaProductRepository
from app.product.service import ProductService

EXPECTED_PRODUCT_COUNT = 20


class _FakeCollection:
    """Stand-in for a ChromaDB collection — returns canned query results."""

    def __init__(self, ids: list[str], distances: list[float]) -> None:
        self._ids = ids
        self._distances = distances
        self.last_query: dict | None = None

    def query(self, query_texts, n_results, include):
        self.last_query = {"query_texts": query_texts, "n_results": n_results, "include": include}
        return {
            "ids": [self._ids[:n_results]],
            "distances": [self._distances[:n_results]],
        }


def _repo(**kwargs) -> ChromaProductRepository:
    defaults = dict(host="x", port=1, collection_name="products")
    defaults.update(kwargs)
    return ChromaProductRepository(**defaults)


class TestCatalog:
    def test_load_products_returns_full_catalog(self) -> None:
        assert len(load_products()) == EXPECTED_PRODUCT_COUNT

    def test_every_product_has_required_fields(self) -> None:
        required = {"id", "name", "category", "brand", "price", "stock", "specs", "description"}
        for product in load_products():
            assert required <= product.keys()

    def test_products_by_id_indexes_catalog(self) -> None:
        by_id = products_by_id()
        assert by_id["probook-15"]["name"] == "ProBook 15"
        assert len(by_id) == EXPECTED_PRODUCT_COUNT


class TestChromaProductRepository:
    def test_search_maps_ids_to_products_in_order(self) -> None:
        repo = _repo(top_k=2)
        fake = _FakeCollection(
            ids=["probook-15", "ultrabook-13", "gamerbook-17"],
            distances=[0.10, 0.20, 0.30],
        )
        with patch.object(ChromaProductRepository, "_get_collection", return_value=fake):
            results = repo.search("cheap laptop")

        assert [p["id"] for p in results] == ["probook-15", "ultrabook-13"]
        # top_k is forwarded to ChromaDB as n_results — retrieval, not a full scan.
        assert fake.last_query["n_results"] == 2

    def test_search_applies_max_distance_threshold(self) -> None:
        repo = _repo(top_k=3, max_distance=0.25)
        fake = _FakeCollection(
            ids=["probook-15", "ultrabook-13", "gamerbook-17"],
            distances=[0.10, 0.20, 0.30],
        )
        with patch.object(ChromaProductRepository, "_get_collection", return_value=fake):
            results = repo.search("laptop")

        # The third hit (distance 0.30 > 0.25) is filtered out as not relevant enough.
        assert [p["id"] for p in results] == ["probook-15", "ultrabook-13"]

    def test_search_skips_unknown_ids(self) -> None:
        repo = _repo(top_k=2)
        fake = _FakeCollection(ids=["probook-15", "ghost-id"], distances=[0.1, 0.2])
        with patch.object(ChromaProductRepository, "_get_collection", return_value=fake):
            results = repo.search("laptop")

        assert [p["id"] for p in results] == ["probook-15"]

    def test_get_all_returns_full_catalog(self) -> None:
        assert len(_repo().get_all()) == EXPECTED_PRODUCT_COUNT


class TestProductService:
    def test_search_delegates_to_repo(self) -> None:
        repo = _repo(top_k=1)
        fake = _FakeCollection(ids=["wireless-headphones"], distances=[0.1])
        service = ProductService(repo)
        with patch.object(ChromaProductRepository, "_get_collection", return_value=fake):
            results = service.search("headphones")
        assert results[0]["name"] == "SoundMax Wireless"

    def test_get_all_delegates_to_repo(self) -> None:
        service = ProductService(_repo())
        assert len(service.get_all()) == EXPECTED_PRODUCT_COUNT
