from langgraph.graph import END, StateGraph

from app.agent.nodes import classify_intent, generate_answer, search_kb
from app.agent.state import AgentState


def route_after_classify(state: AgentState) -> str:
    """Conditional edge: support questions go to KB search, general skips it.

    This is the "agentic" decision — the agent chooses whether to use RAG.
    In the production chatbot this is the LangGraph conditional edge
    after the router assistant's decision.
    """
    if state["intent"] == "support":
        return "search_kb"
    return "generate_answer"


def build_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph.

    Graph topology:
        classify_intent ──(support)──> search_kb ──> generate_answer ──> END
                        ──(general)────────────────> generate_answer ──> END
    """
    graph: StateGraph = StateGraph(AgentState)

    graph.add_node("classify_intent", classify_intent)
    graph.add_node("search_kb", search_kb)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "search_kb": "search_kb",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("search_kb", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph.compile()
