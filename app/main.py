import uuid

from fastapi import FastAPI

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.guardrails import is_on_topic, validate_output
from app.models import ChatRequest, ChatResponse

app = FastAPI(title="E-commerce AI Agent")

agent_graph = build_graph()

# In-memory session store — simulates Redis/Aurora from the production chatbot.
# Keys are session_id, values are the last AgentState from that session.
# For multi-turn HITL: when a user confirms/cancels an order, we restore
# the previous graph state so confirm_order can access the order draft.
sessions: dict[str, dict] = {}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id: str = request.session_id or str(uuid.uuid4())

    # Input guardrail — rejects off-topic before hitting the graph.
    if not is_on_topic(request.message):
        return ChatResponse(
            answer="I can only help with product questions and orders in our store.",
            session_id=session_id,
            sources=[],
        )

    # Restore previous session state (for multi-turn HITL)
    previous: dict | None = sessions.get(session_id)

    if previous:
        # Resume existing session — append new message to conversation history
        messages = [*previous["messages"], {"role": "user", "content": request.message}]
    else:
        # New session — start fresh conversation
        messages = [{"role": "user", "content": request.message}]

    initial_state: AgentState = {
        "session_id": session_id,
        "messages": messages,
        "skill": "",
        "product_results": [],
        "final_answer": "",
    }

    # Carry over pending order draft so confirm_order can access it
    if previous and previous.get("needs_confirmation"):
        initial_state["needs_confirmation"] = True
        initial_state["order"] = previous.get("order", {})

    result: dict = await agent_graph.ainvoke(initial_state)

    # Output guardrail — validate the final answer before returning.
    if not validate_output(result.get("final_answer", "")):
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Output guardrail: invalid response")

    # Persist session state for multi-turn
    sessions[session_id] = result

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=result.get("product_results", []),
    )
