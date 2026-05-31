# Mini AI Agent

FastAPI + LangGraph + Pydantic playground — learning Python AI stack by building a simplified version of an agentic RAG chatbot.

## Stack

| Tech | Role |
|------|------|
| FastAPI | Async HTTP API |
| Pydantic | Data models + validation |
| LangGraph | Agent orchestration (state graph) |

## Run

```bash
pip install fastapi "uvicorn[standard]" pydantic langgraph
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

## Test

```bash
pip install pytest httpx anyio
python -m pytest tests/ -v
```

## Test endpoint

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What are your opening hours?"}'
```
