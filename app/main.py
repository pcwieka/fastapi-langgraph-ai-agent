import uuid

from fastapi import FastAPI

from app.models import ChatRequest, ChatResponse

app = FastAPI(title="Mini AI Agent")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint — mock for now, LangGraph comes on Day 2."""
    return ChatResponse(
        answer=f"Echo: {request.message}",
        session_id=request.session_id or str(uuid.uuid4()),
    )
