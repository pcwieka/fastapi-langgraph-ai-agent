import uuid

from dotenv import load_dotenv
from fastapi import FastAPI

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.llm.guard import Guardrail
from app.logger import format_state, setup_logger
from app.models import ChatRequest, ChatResponse

load_dotenv()

app = FastAPI(title="E-commerce AI Agent")

agent_graph = build_graph()
guard = Guardrail()

sessions: dict[str, dict] = {}
logger = setup_logger("agent")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id: str = request.session_id or str(uuid.uuid4())

    logger.info("REQUEST | session=%s | message=%s", session_id, request.message[:100])

    # Input guardrail — LLM classifies if message is on-topic
    # Pass conversation history so short follow-ups (yes/no) are evaluated in context
    previous_messages = sessions.get(session_id, {}).get("messages")
    input_check = await guard.check_input(request.message, history=previous_messages)
    logger.info("GUARD INPUT | on_topic=%s | reason=%s", input_check.on_topic, input_check.reason)

    if not input_check.on_topic:
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

    logger.info("GRAPH INPUT%s", format_state(initial_state))

    result: dict = await agent_graph.ainvoke(initial_state)

    logger.info("GRAPH OUTPUT%s", format_state(result))

    # Output guardrail — LLM validates the response before returning to user
    output_check = await guard.check_output(result.get("final_answer", ""))
    logger.info("GUARD OUTPUT | valid=%s | reason=%s", output_check.valid, output_check.reason)

    if not output_check.valid:
        from fastapi import HTTPException

        raise HTTPException(status_code=500, detail="Output guardrail: invalid response")

    sessions[session_id] = result

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=result.get("product_results", []),
    )
