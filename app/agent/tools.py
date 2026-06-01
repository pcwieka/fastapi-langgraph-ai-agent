MOCK_PRODUCTS: dict[str, dict[str, object]] = {
    "probook-15": {
        "id": "probook-15",
        "name": "ProBook 15",
        "category": "laptops",
        "brand": "TechCorp",
        "price": 1299.99,
        "stock": 5,
        "specs": "15.6\" display, 16GB RAM, 512GB SSD, Intel i7",
        "description": "Business laptop with all-day battery life. Ideal for professionals.",
    },
    "budget-phone-x": {
        "id": "budget-phone-x",
        "name": "BudgetPhone X",
        "category": "phones",
        "brand": "PhoneCo",
        "price": 299.99,
        "stock": 12,
        "specs": "6.1\" display, 8GB RAM, 128GB storage, dual camera",
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


def search_products(query: str) -> list[dict[str, object]]:
    """Mock product search — simulates product catalog search.

    In production this would be Elasticsearch / vector search over a product DB.
    """
    results: list[dict[str, object]] = []
    lower = query.lower()
    for product_id, product in MOCK_PRODUCTS.items():
        searchable = f"{product['name']} {product['category']} {product['brand']} {product.get('specs', '')}".lower()
        if any(word in searchable or word in lower for word in lower.split()):
            results.append(product)
    return results if results else list(MOCK_PRODUCTS.values())


def place_order(product_id: str, quantity: int) -> OrderDraft:
    """Mock order placement — returns draft, does NOT actually place the order.

    In production this would call an order management system via REST/gRPC.
    Requires HITL confirmation before execution.
    """
    product = MOCK_PRODUCTS[product_id]
    return {
        "product_id": str(product["id"]),
        "product_name": str(product["name"]),
        "quantity": quantity,
        "total_price": float(product["price"]) * quantity,
    }
