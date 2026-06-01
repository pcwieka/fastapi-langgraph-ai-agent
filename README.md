# E-commerce AI Agent

FastAPI + LangGraph + Pydantic playground — an e-commerce shopping assistant with two skills:

- **Q&A** — product questions → mock KB search → structured response (agentic RAG)
- **Order (HITL)** — order draft → user confirmation → place or cancel

Inspired by agentic RAG, guardrails, and LangGraph orchestration patterns used in production AI chatbots.

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
