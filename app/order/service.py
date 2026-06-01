from app.order.repository import OrderRepository


class OrderService:
    """Order creation and tracking.

    Depends on OrderRepository interface - never knows the concrete implementation.
    """

    def __init__(self, repo: OrderRepository) -> None:
        self._repo = repo

    def create_order(self, session_id: str, order: dict) -> str:
        return self._repo.save(session_id, order)

    def find_orders(self, session_id: str) -> list[dict]:
        return self._repo.find_by_session(session_id)
