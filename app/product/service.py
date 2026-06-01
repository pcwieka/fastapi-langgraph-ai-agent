from app.product.repository import ProductRepository


class ProductService:
    """Product search and retrieval.

    Depends on ProductRepository interface - never knows the concrete implementation.
    """

    def __init__(self, repo: ProductRepository) -> None:
        self._repo = repo

    def search(self, query: str) -> list[dict]:
        return self._repo.search(query)

    def get_all(self) -> list[dict]:
        return self._repo.get_all()
