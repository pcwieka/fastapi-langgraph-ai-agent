from typing import TypedDict


class AgentState(TypedDict):
    """State that flows through LangGraph nodes.

    TypedDict is a lightweight alternative to Pydantic for graph state.
    LangGraph copies this dict between nodes — each node returns a partial
    update that gets merged in (like Redux reducers).
    """

    session_id: str
    messages: list[dict[str, str]]
    intent: str
    kb_results: list[str]
    final_answer: str
