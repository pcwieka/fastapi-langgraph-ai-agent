from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    confirm_order,
    generate_qa_answer,
    prepare_order,
    route_skill,
    search_products_node,
)
from app.agent.state import AgentState


def route_after_skill(state: AgentState) -> str:
    """Conditional edge after skill routing.

    Q&A path: search products → generate answer
    Order path (no pending confirmation): prepare order draft → ask user to confirm
    Order path (pending confirmation): confirm or cancel the order
    """
    if state.get("needs_confirmation"):
        return "confirm_order"
    if state["skill"] == "order":
        return "prepare_order"
    return "search_products"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph.

    Graph topology:

    Q&A path:
        route_skill ──(qa)──> search_products ──> generate_qa_answer ──> END

    Order path (first turn):
        route_skill ──(order)──> prepare_order ──> END [needs_confirmation=True]

    Order path (second turn — user confirms/cancels):
        route_skill ──(pending)──> confirm_order ──> END

    The HITL loop is implemented via session state persistence outside
    the graph. When needs_confirmation=True, the next request resumes
    in the order flow without requiring LangGraph interrupt/resume.
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("route_skill", route_skill)
    graph.add_node("search_products", search_products_node)
    graph.add_node("generate_qa_answer", generate_qa_answer)
    graph.add_node("prepare_order", prepare_order)
    graph.add_node("confirm_order", confirm_order)

    graph.set_entry_point("route_skill")

    graph.add_conditional_edges(
        "route_skill",
        route_after_skill,
        {
            "search_products": "search_products",
            "prepare_order": "prepare_order",
            "confirm_order": "confirm_order",
        },
    )
    graph.add_edge("search_products", "generate_qa_answer")
    graph.add_edge("generate_qa_answer", END)
    graph.add_edge("prepare_order", END)
    graph.add_edge("confirm_order", END)

    return graph.compile()
