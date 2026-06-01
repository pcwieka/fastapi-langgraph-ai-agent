from app.agent.state import AgentState
from app.agent.tools import place_order, search_products


def route_skill(state: AgentState) -> dict:
    """Classify user intent: Q&A (product questions) or Order (buy intent).

    In production this would be an LLM call for classification.
    We use keyword matching as a lightweight mock.
    """
    # If there's an unconfirmed order from a previous turn, stay in order flow
    if state.get("needs_confirmation"):
        return {}

    last_message: str = state["messages"][-1]["content"].lower()
    order_keywords: list[str] = [
        "buy", "order", "purchase", "i want", "i'd like", "get me",
        "place", "checkout", "cart",
    ]
    is_order: bool = any(kw in last_message for kw in order_keywords)
    return {"skill": "order" if is_order else "qa"}


def search_products_node(state: AgentState) -> dict:
    """Search product catalog for matching products (RAG retrieval step).

    Maps to the retrieval phase of agentic RAG — agent decides whether
    to search, then this node fetches context for the response.
    """
    query: str = state["messages"][-1]["content"]
    results: list[dict[str, object]] = search_products(query)
    return {"product_results": results}


def generate_qa_answer(state: AgentState) -> dict:
    """Compose a product Q&A response from search results.

    In production this node would feed product data as context to an LLM
    (the generation step of RAG). Here we format results directly.
    """
    products = state.get("product_results", [])
    if not products:
        answer = "I couldn't find any products matching your query."
        sources: list[str] = []
    else:
        lines: list[str] = []
        sources = []
        for p in products:
            lines.append(
                f"- {p['name']} ({p['brand']}): ${p['price']:.2f} — {p['description']} [stock: {p['stock']}]"
            )
            sources.append(str(p["name"]))
        answer = "Here's what I found:\n\n" + "\n".join(lines)

    return {
        "final_answer": answer,
        "messages": [
            *state["messages"],
            {"role": "assistant", "content": answer},
        ],
        "product_results": sources,  # repurpose for ChatResponse.sources
    }


def prepare_order(state: AgentState) -> dict:
    """Generate order draft and ask for user confirmation (HITL entry point).

    Does NOT execute the order — returns a draft and sets needs_confirmation=True.
    The next user message is routed to confirm_order.
    """
    last_message = state["messages"][-1]["content"]

    # Super-simple product detection from the message (mock)
    product_id = "probook-15"
    quantity = 1
    for pid in ["probook-15", "budget-phone-x", "ergo-mouse", "wireless-headphones"]:
        if pid.replace("-", " ") in last_message or any(
            word in last_message for word in pid.split("-")
        ):
            product_id = pid
            break

    draft = place_order(product_id, quantity)

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
        "messages": [
            *state["messages"],
            {"role": "assistant", "content": answer},
        ],
    }


def confirm_order(state: AgentState) -> dict:
    """Process user confirmation and finalize (or cancel) the order.

    This is the resume point after HITL — in production LangGraph this
    would use interrupt()/Command(resume=...) instead of parsing the message.
    """
    last_message = state["messages"][-1]["content"].lower().strip()
    confirmed: bool = last_message in ("yes", "yeah", "y", "confirm", "ok", "okay")

    if confirmed:
        answer = (
            f"Order confirmed! Your {state['order']['product_name']} "
            f"will be shipped within 2-3 business days. "
            f"Order total: ${state['order']['total_price']:.2f}"
        )
    else:
        answer = "Order cancelled. Let me know if you need anything else."

    return {
        "order_confirmed": confirmed,
        "needs_confirmation": False,
        "final_answer": answer,
        "messages": [
            *state["messages"],
            {"role": "assistant", "content": answer},
        ],
    }
