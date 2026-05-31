from app.agent.state import AgentState
from app.agent.tools import search_knowledgebase


def classify_intent(state: AgentState) -> dict:
    """Decide if user message needs KB lookup (support) or is general chat.

    In the production chatbot this is an LLM call (router assistant).
    We do simple keyword matching as a placeholder.
    """
    last_message: str = state["messages"][-1]["content"].lower()
    support_keywords: list[str] = ["hours", "price", "pricing", "return", "refund", "opening", "help"]
    is_support: bool = any(kw in last_message for kw in support_keywords)

    return {"intent": "support" if is_support else "general"}


def search_kb(state: AgentState) -> dict:
    """Call the KB tool with the user's message.

    Maps directly to the search_knowledgebase tool in the production chatbot.
    """
    query: str = state["messages"][-1]["content"]
    results: list[str] = search_knowledgebase(query)
    return {"kb_results": results}


def generate_answer(state: AgentState) -> dict:
    """Compose final answer from KB results (if support) or direct response.

    In production this node would call GPT-5 with KB articles as context
    (the RAG generation step). Here we just format the mock results.
    """
    if state["intent"] == "support" and state.get("kb_results"):
        sources = state["kb_results"]
        answer = " | ".join(sources)
    else:
        sources = []
        answer = "This is a general question. In production, GPT-5 would handle this."

    return {
        "final_answer": answer,
        "messages": [
            *state["messages"],
            {"role": "assistant", "content": answer},
        ],
    }
