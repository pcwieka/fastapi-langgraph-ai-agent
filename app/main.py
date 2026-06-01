import uuid

from dotenv import load_dotenv
from fastapi import FastAPI

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.guardrails import validate_output
from app.llm.guard import LlmGuardrail
from app.models import ChatRequest, ChatResponse

load_dotenv()

app = FastAPI(title="E-commerce AI Agent")

agent_graph = build_graph()

sessions: dict[str, dict] = {}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id: str = request.session_id or str(uuid.uuid4())

    # Input guardrail — LLM-based classification
    guard_result = await LlmGuardrail().check(request.message)
    if not guard_result.on_topic:
        return ChatResponse(
            answer="I can only help with product questions and orders in our store.",
            session_id=session_id,
            sources=[],
        )

    previous: dict | None = sessions.get(session_id)

    if previous:
        messages = [*previous["messages"], {"role": "user", "content": request.message}]
    else:
        messages = [{"role": "user", "content": request.message}]

    initial_state: AgentState = {
        "session_id": session_id,
        "messages": messages,
        "skill": "",
        "product_results": [],
        "final_answer": "",
    }

    if previous and previous.get("needs_confirmation"):
        initial_state["needs_confirmation"] = True
        initial_state["order"] = previous.get("order", {})

    result: dict = await agent_graph.ainvoke(initial_state)

    if not validate_output(result.get("final_answer", "")):
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail="Output guardrail: invalid response")

    sessions[session_id] = result

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=result.get("product_results", []),
    )
