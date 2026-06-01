from app.order.repository import InMemoryOrderRepository
from app.order.service import OrderService


class TestInMemoryOrderRepository:
    def test_save_returns_order_id(self) -> None:
        repo = InMemoryOrderRepository()
        order_id = repo.save("s1", {"product_name": "ProBook 15", "quantity": 1, "total_price": 1299.99})
        assert order_id == "ORD-1000"

    def test_save_increments_ids(self) -> None:
        repo = InMemoryOrderRepository()
        id1 = repo.save("s1", {})
        id2 = repo.save("s1", {})
        assert id1 == "ORD-1000"
        assert id2 == "ORD-1001"

    def test_find_by_session_returns_matching_orders(self) -> None:
        repo = InMemoryOrderRepository()
        repo.save("s1", {"product_name": "A"})
        repo.save("s2", {"product_name": "B"})
        repo.save("s1", {"product_name": "C"})

        results = repo.find_by_session("s1")
        assert len(results) == 2
        assert results[0]["product_name"] == "A"
        assert results[1]["product_name"] == "C"

    def test_find_by_session_empty(self) -> None:
        repo = InMemoryOrderRepository()
        assert repo.find_by_session("nonexistent") == []

    def test_saved_order_has_default_status(self) -> None:
        repo = InMemoryOrderRepository()
        repo.save("s1", {"product_name": "X", "quantity": 2, "total_price": 50.0})
        results = repo.find_by_session("s1")
        assert results[0]["status"] == "processing"
        assert results[0]["eta"] == "2-3 business days"

    def test_reset_clears_data_and_counter(self) -> None:
        repo = InMemoryOrderRepository()
        repo.save("s1", {})
        repo.reset()
        assert repo.find_by_session("s1") == []
        assert repo.save("s1", {}) == "ORD-1000"


class TestOrderService:
    def test_create_order_delegates_to_repo(self) -> None:
        repo = InMemoryOrderRepository()
        service = OrderService(repo)
        order_id = service.create_order("s1", {"product_name": "Test", "quantity": 1, "total_price": 10.0})
        assert order_id.startswith("ORD-")

    def test_find_orders_delegates_to_repo(self) -> None:
        repo = InMemoryOrderRepository()
        service = OrderService(repo)
        service.create_order("s1", {"product_name": "A"})
        service.create_order("s1", {"product_name": "B"})
        assert len(service.find_orders("s1")) == 2
