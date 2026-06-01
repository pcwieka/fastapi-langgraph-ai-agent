from typing import NotRequired, TypedDict


class OrderDraft(TypedDict):
    product_id: str
    product_name: str
    quantity: int
    total_price: float


class AgentState(TypedDict):
    """State that flows through LangGraph nodes.

    TypedDict is a lightweight alternative to Pydantic for graph state.
    LangGraph copies this dict between nodes - each node returns a partial
    update that gets merged into the accumulated state.
    """

    session_id: str
    messages: list[dict[str, str]]
    skill: str
    product_results: list[dict[str, object]]
    order: NotRequired[OrderDraft]
    order_confirmed: NotRequired[bool]
    user_response: NotRequired[str]
    final_answer: str
