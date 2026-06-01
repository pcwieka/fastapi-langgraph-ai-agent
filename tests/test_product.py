from app.product.repository import InMemoryProductRepository
from app.product.service import ProductService


class TestInMemoryProductRepository:
    def test_search_finds_matching_product(self) -> None:
        repo = InMemoryProductRepository()
        results = repo.search("laptops")
        assert len(results) > 0
        assert any(p["name"] == "ProBook 15" for p in results)

    def test_search_case_insensitive(self) -> None:
        repo = InMemoryProductRepository()
        results = repo.search("PROBOOK")
        assert any(p["name"] == "ProBook 15" for p in results)

    def test_search_no_match_returns_all(self) -> None:
        repo = InMemoryProductRepository()
        results = repo.search("xyznonexistent")
        assert len(results) == 4  # all products returned as fallback

    def test_search_by_brand(self) -> None:
        repo = InMemoryProductRepository()
        results = repo.search("SoundMax")
        assert len(results) > 0
        assert all(p["brand"] == "SoundMax" for p in results)

    def test_get_all_returns_all_products(self) -> None:
        repo = InMemoryProductRepository()
        results = repo.get_all()
        assert len(results) == 4


class TestProductService:
    def test_search_delegates_to_repo(self) -> None:
        repo = InMemoryProductRepository()
        service = ProductService(repo)
        results = service.search("headphones")
        assert any(p["name"] == "SoundMax Wireless" for p in results)

    def test_get_all_delegates_to_repo(self) -> None:
        repo = InMemoryProductRepository()
        service = ProductService(repo)
        assert len(service.get_all()) == 4
