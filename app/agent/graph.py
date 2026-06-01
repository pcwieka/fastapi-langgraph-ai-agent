from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from app.agent.skills import (
    await_confirmation,
    finalize_order,
    generate_qa_answer,
    prepare_order,
    route_skill,
    search_products,
    track_order,
)
from app.agent.state import AgentState


def route_after_skill(state: AgentState) -> str:
    """Conditional edge after skill routing.

    Q&A path: search products → generate answer
    Order path: prepare order draft → interrupt for HITL confirmation
    Track path: look up order status in the registry
    """
    if state["skill"] == "order":
        return "prepare_order"
    if state["skill"] == "track":
        return "track_order"
    return "search_products"


def build_graph(checkpointer: BaseCheckpointSaver) -> StateGraph:
    """Build and compile the LangGraph agent graph.

    Requires a checkpointer for state persistence between turns.
    In production: SqliteSaver / PostgresSaver. In tests: InMemorySaver.

    Graph topology:

    Q&A path:
        route_skill ──(qa)──> search_products ──> generate_qa_answer ──> END

    Order path (HITL via interrupt):
        route_skill ──(order)──> prepare_order ──> await_confirmation ──> finalize_order ──> END
                                    (interrupt() pauses execution)     (resumes here)

    Track path:
        route_skill ──(track)──> track_order ──> END
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("route_skill", route_skill)
    graph.add_node("search_products", search_products)
    graph.add_node("generate_qa_answer", generate_qa_answer)
    graph.add_node("prepare_order", prepare_order)
    graph.add_node("await_confirmation", await_confirmation)
    graph.add_node("finalize_order", finalize_order)
    graph.add_node("track_order", track_order)

    graph.set_entry_point("route_skill")

    graph.add_conditional_edges(
        "route_skill",
        route_after_skill,
        {
            "search_products": "search_products",
            "prepare_order": "prepare_order",
            "track_order": "track_order",
        },
    )
    graph.add_edge("search_products", "generate_qa_answer")
    graph.add_edge("generate_qa_answer", END)
    graph.add_edge("prepare_order", "await_confirmation")
    graph.add_edge("await_confirmation", "finalize_order")
    graph.add_edge("finalize_order", END)
    graph.add_edge("track_order", END)

    return graph.compile(checkpointer=checkpointer)
