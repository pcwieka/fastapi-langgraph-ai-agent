from app.agent.state import AgentState
from app.agent.tools import MOCK_PRODUCTS, search_products
from app.llm.response_generator import OrderDraftGenerator, QaResponseGenerator
from app.llm.skill_router import SkillRouter

# Module-level singletons — one instance shared across all requests.
# Each wraps get_llm().with_structured_output(), so creating them per request
# would re-wrap the LLM client unnecessarily.
skill_router = SkillRouter()
qa_generator = QaResponseGenerator()
order_generator = OrderDraftGenerator()


async def route_skill(state: AgentState) -> dict:
    """Classify user intent: Q&A or Order."""
    if state.get("needs_confirmation"):
        return {}

    last_message: str = state["messages"][-1]["content"]
    result = await skill_router.classify(last_message)
    return {"skill": result.skill}


def search_products_node(state: AgentState) -> dict:
    """Search product catalog — not an LLM call.

    In production this would be Elasticsearch / vector DB, not an LLM.
    """
    query: str = state["messages"][-1]["content"]
    results: list[dict[str, object]] = search_products(query)
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
    """Generate order draft and ask for confirmation (HITL entry point)."""
    last_message: str = state["messages"][-1]["content"]
    all_products: list[dict[str, object]] = list(MOCK_PRODUCTS.values())

    draft_result = await order_generator.generate(last_message, all_products)
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
        "needs_confirmation": True,
        "order_confirmed": False,
        "final_answer": answer,
        "messages": [*state["messages"], {"role": "assistant", "content": answer}],
    }


def confirm_order(state: AgentState) -> dict:
    """Process confirmation and finalize or cancel the order."""
    last_message: str = state["messages"][-1]["content"].lower().strip()
    confirmed: bool = last_message in ("yes", "yeah", "y", "confirm", "ok", "okay")

    order = state.get("order", {})
    if confirmed:
        # In production: POST to order management API (REST/gRPC)
        # Ex. order_api.place(order["product_id"], order["quantity"])
        answer = (
            f"Order confirmed! Your {order.get('product_name', 'item')} "
            f"will be shipped within 2-3 business days. "
            f"Order total: ${order.get('total_price', 0):.2f}"
        )
    else:
        answer = "Order cancelled. Let me know if you need anything else."

    return {
        "order_confirmed": confirmed,
        "needs_confirmation": False,
        "final_answer": answer,
        "messages": [*state["messages"], {"role": "assistant", "content": answer}],
    }
