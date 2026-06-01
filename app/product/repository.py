from abc import ABC, abstractmethod

MOCK_PRODUCTS: dict[str, dict[str, object]] = {
    "probook-15": {
        "id": "probook-15",
        "name": "ProBook 15",
        "category": "laptops",
        "brand": "TechCorp",
        "price": 1299.99,
        "stock": 5,
        "specs": '15.6" display, 16GB RAM, 512GB SSD, Intel i7',
        "description": "Business laptop with all-day battery life. Ideal for professionals.",
    },
    "budget-phone-x": {
        "id": "budget-phone-x",
        "name": "BudgetPhone X",
        "category": "phones",
        "brand": "PhoneCo",
        "price": 299.99,
        "stock": 12,
        "specs": '6.1" display, 8GB RAM, 128GB storage, dual camera',
        "description": "Affordable smartphone with great battery life.",
    },
    "ergo-mouse": {
        "id": "ergo-mouse",
        "name": "ErgoMouse Pro",
        "category": "accessories",
        "brand": "ComfortTech",
        "price": 79.99,
        "stock": 0,
        "specs": "Wireless, Bluetooth, ergonomic design, USB-C charging",
        "description": "Ergonomic wireless mouse designed for all-day comfort.",
    },
    "wireless-headphones": {
        "id": "wireless-headphones",
        "name": "SoundMax Wireless",
        "category": "accessories",
        "brand": "SoundMax",
        "price": 149.99,
        "stock": 8,
        "specs": "Active noise cancelling, 30h battery, Bluetooth 5.3",
        "description": "Premium wireless headphones with studio-quality sound.",
    },
}


class ProductRepository(ABC):
    """Interface for product data access.

    In production: Elasticsearch / vector DB.
    """

    @abstractmethod
    def search(self, query: str) -> list[dict[str, object]]: ...

    @abstractmethod
    def get_all(self) -> list[dict[str, object]]: ...


class InMemoryProductRepository(ProductRepository):
    """In-memory product catalog with keyword-based search."""

    def search(self, query: str) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        lower = query.lower()
        for product in MOCK_PRODUCTS.values():
            fields = [
                product["name"],
                product["category"],
                product["brand"],
                product.get("specs", ""),
            ]
            searchable = " ".join(str(f) for f in fields).lower()
            if any(word in searchable for word in lower.split()):
                results.append(product)
        return results if results else list(MOCK_PRODUCTS.values())

    def get_all(self) -> list[dict[str, object]]:
        return list(MOCK_PRODUCTS.values())
