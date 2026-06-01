from abc import ABC, abstractmethod


class OrderRepository(ABC):
    """Interface for order persistence.

    In production: relational DB or order management API.
    """

    @abstractmethod
    def save(self, session_id: str, order: dict) -> str:
        """Persist a confirmed order and return its ID."""
        ...

    @abstractmethod
    def find_by_session(self, session_id: str) -> list[dict]:
        """Return all orders for a given session."""
        ...


class InMemoryOrderRepository(OrderRepository):
    """In-memory order registry for development and testing."""

    def __init__(self) -> None:
        self._orders: dict[str, dict] = {}
        self._next_id = 1000

    def save(self, session_id: str, order: dict) -> str:
        order_id = f"ORD-{self._next_id}"
        self._next_id += 1
        self._orders[order_id] = {
            "order_id": order_id,
            "session_id": session_id,
            "product_name": order.get("product_name", ""),
            "quantity": order.get("quantity", 0),
            "total_price": order.get("total_price", 0),
            "status": "processing",
            "eta": "2-3 business days",
        }
        return order_id

    def find_by_session(self, session_id: str) -> list[dict]:
        return [o for o in self._orders.values() if o["session_id"] == session_id]

    def reset(self) -> None:
        """Clear all orders and reset the ID counter (for tests)."""
        self._orders.clear()
        self._next_id = 1000
