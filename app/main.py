import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from app.agent.graph import build_graph
from app.agent.state import AgentState
from app.llm.guardrail import Guardrail
from app.logger import format_state, setup_logger
from app.models import ChatRequest, ChatResponse

load_dotenv()

agent_graph: StateGraph | None = None
guard = Guardrail()
logger = setup_logger("agent")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown: manage AsyncSqliteSaver lifecycle."""
    global agent_graph
    async with AsyncSqliteSaver.from_conn_string("data/checkpoints.db") as checkpointer:
        agent_graph = build_graph(checkpointer=checkpointer)
        logger.info("Checkpointer ready (AsyncSqliteSaver: data/checkpoints.db)")
        yield
    logger.info("Checkpointer closed")


app = FastAPI(title="E-commerce AI Agent", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    session_id: str = request.session_id or str(uuid.uuid4())
    config: dict = {"configurable": {"thread_id": session_id}}

    t_start = time.perf_counter()
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("REQUEST | session=%s | message=%s", session_id, request.message[:100])

    # Check if there's a pending interrupt for this session
    snapshot = await agent_graph.aget_state(config)
    has_interrupt = bool(snapshot.next)
    history_messages = snapshot.values.get("messages") if snapshot.values else None

    # Input guardrail with conversation history for context
    t0 = time.perf_counter()
    input_check = await guard.check_input(request.message, history=history_messages)
    logger.info(
        "GUARD INPUT  [%.2fs] | on_topic=%s | reason=%s",
        time.perf_counter() - t0,
        input_check.on_topic,
        input_check.reason,
    )

    if not input_check.on_topic:
        logger.info("DONE [%.2fs] | rejected by input guardrail", time.perf_counter() - t_start)
        return ChatResponse(
            answer="I can only help with product questions and orders in our store.",
            session_id=session_id,
            sources=[],
        )

    # Invoke graph — resume from interrupt or start new run
    t_graph = time.perf_counter()
    if has_interrupt:
        logger.info("GRAPH RESUME | resuming from interrupt")
        result = await agent_graph.ainvoke(Command(resume=request.message), config)
    else:
        initial_state: AgentState = {
            "session_id": session_id,
            "messages": [{"role": "user", "content": request.message}],
            "skill": "",
            "product_results": [],
            "final_answer": "",
        }
        logger.info("GRAPH INPUT%s", format_state(initial_state))
        result = await agent_graph.ainvoke(initial_state, config)

    logger.info(
        "GRAPH OUTPUT [%.2fs]%s",
        time.perf_counter() - t_graph,
        format_state(result),
    )

    # Output guardrail — LLM validates the response before returning to user
    t1 = time.perf_counter()
    output_check = await guard.check_output(result.get("final_answer", ""))
    logger.info(
        "GUARD OUTPUT [%.2fs] | valid=%s | reason=%s",
        time.perf_counter() - t1,
        output_check.valid,
        output_check.reason,
    )

    if not output_check.valid:
        raise HTTPException(status_code=500, detail="Output guardrail: invalid response")

    logger.info("DONE [%.2fs] | total request time", time.perf_counter() - t_start)

    return ChatResponse(
        answer=result["final_answer"],
        session_id=session_id,
        sources=result.get("product_results", []),
    )
