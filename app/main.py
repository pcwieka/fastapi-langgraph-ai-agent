import uuid

from fastapi import FastAPI

from app.agent.graph import build_graph
from app.guardrails import is_on_topic
from app.models import ChatRequest, ChatResponse

app = FastAPI(title="Mini AI Agent")

# Compiled once at startup — same graph instance handles all requests.
# Like a Spring singleton bean that orchestrates the agent flow.
agent_graph = build_graph()


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id: str = request.session_id or str(uuid.uuid4())

    # Input guardrail — rejects off-topic before hitting the graph.
    # In production this would be an LLM call (GPT-5 mini), not keywords.
    if not is_on_topic(request.message):
        return ChatResponse(
            answer="I can only answer questions about our services (support, pricing, returns).",
            session_id=session_id,
            sources=[],
        )

    initial_state = {
        "session_id": session_id,
        "messages": [{"role": "user", "content": request.message}],
        "intent": "",
        "kb_results": [],
        "final_answer": "",
    }

    result: dict = await agent_graph.ainvoke(initial_state)

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=result.get("kb_results", []),
    )
