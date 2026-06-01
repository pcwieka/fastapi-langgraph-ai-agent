from langgraph.types import interrupt

from app.agent.state import AgentState
from app.llm.response_generator import OrderDraftGenerator, QaResponseGenerator
from app.llm.skill_router import SkillRouter
from app.order.repository import InMemoryOrderRepository
from app.order.service import OrderService
from app.product.repository import InMemoryProductRepository
from app.product.service import ProductService

# Module-level singletons — one instance shared across all requests.
skill_router = SkillRouter()
qa_generator = QaResponseGenerator()
order_generator = OrderDraftGenerator()

product_repo = InMemoryProductRepository()
order_repo = InMemoryOrderRepository()

product_service = ProductService(product_repo)
order_service = OrderService(order_repo)


async def route_skill(state: AgentState) -> dict:
    """Classify user intent: Q&A, Order, or Track.

    Only runs for new messages — on HITL resume, LangGraph skips
    the entry point and continues from the interrupt directly.
    """
    last_message: str = state["messages"][-1]["content"]
    result = await skill_router.classify(last_message)
    return {"skill": result.skill}


def search_products(state: AgentState) -> dict:
    """Search product catalog via ProductService."""
    query: str = state["messages"][-1]["content"]
    results: list[dict[str, object]] = product_service.search(query)
    return {"product_results": results}


async def generate_qa_answer(state: AgentState) -> dict:
    """Compose product Q&A response — LLM with search results as context (RAG generation)."""
    products = state.get("product_results", [])
    user_message: str = state["messages"][-1]["content"]
    answer: str = await qa_generator.generate(user_message, products)

    sources = [str(p["name"]) for p in products] if products else []
    return {
        "final_answer": answer,
        "messages": [*state["messages"], {"role": "assistant", "content": answer}],
        "product_results": sources,
    }


async def prepare_order(state: AgentState) -> dict:
    """Generate order draft via LLM and return it for HITL confirmation.

    The actual confirmation pause happens in await_confirmation node.
    """
    last_message: str = state["messages"][-1]["content"]
    all_products: list[dict[str, object]] = product_service.get_all()

    draft_result = await order_generator.generate(
        last_message, all_products, history=state.get("messages", [])
    )
    draft: dict = {
        "product_id": draft_result.product_id,
        "product_name": draft_result.product_name,
        "quantity": draft_result.quantity,
        "total_price": draft_result.total_price,
    }

    answer = (
        f"Here's your order summary:\n\n"
        f"Product: {draft['product_name']}\n"
        f"Quantity: {draft['quantity']}\n"
        f"Total: ${draft['total_price']:.2f}\n\n"
        f"Would you like to confirm this order? (yes/no)"
    )

    return {
        "order": draft,
        "order_confirmed": False,
        "final_answer": answer,
        "messages": [*state["messages"], {"role": "assistant", "content": answer}],
    }


def await_confirmation(state: AgentState) -> dict:
    """Pause execution and wait for user confirmation via LangGraph interrupt.

    On first pass: interrupt() pauses the graph, checkpointer persists state.
    On resume: interrupt() returns the value from Command(resume=...).
    """
    user_response: str = interrupt("Waiting for order confirmation")

    return {
        "user_response": user_response,
        "messages": [
            *state["messages"],
            {"role": "user", "content": user_response},
        ],
    }


def finalize_order(state: AgentState) -> dict:
    """Process HITL confirmation and create or cancel the order."""
    user_response: str = state.get("user_response", "")
    confirmed: bool = user_response.lower().strip() in (
        "yes",
        "yeah",
        "y",
        "confirm",
        "ok",
        "okay",
    )

    order = state.get("order", {})
    if confirmed:
        order_id = order_service.create_order(state["session_id"], order)
        answer = (
            f"Order {order_id} confirmed! Your {order.get('product_name', 'item')} "
            f"will be shipped within 2-3 business days. "
            f"Order total: ${order.get('total_price', 0):.2f}"
        )
    else:
        answer = "Order cancelled. Let me know if you need anything else."

    return {
        "order_confirmed": confirmed,
        "final_answer": answer,
        "messages": [*state["messages"], {"role": "assistant", "content": answer}],
    }


def track_order(state: AgentState) -> dict:
    """Look up orders for the current session and return status."""
    orders = order_service.find_orders(state["session_id"])

    if not orders:
        answer = "I couldn't find any orders for your session. Have you placed an order yet?"
    else:
        lines = []
        for o in orders:
            lines.append(
                f"Order {o['order_id']}: {o['product_name']} x{o['quantity']} — "
                f"${o['total_price']:.2f} | Status: {o['status']} | ETA: {o['eta']}"
            )
        answer = "Here are your orders:\n\n" + "\n".join(lines)

    return {
        "final_answer": answer,
        "messages": [*state["messages"], {"role": "assistant", "content": answer}],
    }
