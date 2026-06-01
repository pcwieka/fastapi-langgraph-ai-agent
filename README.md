# E-commerce AI Agent

FastAPI + LangGraph + DeepSeek LLM — an e-commerce shopping assistant with two skills:

- **Q&A** — product questions → search → LLM response (agentic RAG)
- **Order (HITL)** — LLM extracts product → order draft → user confirmation → place/cancel

## Stack

| Tech | Role |
|------|------|
| FastAPI | Async HTTP API |
| Pydantic | Data models + validation |
| LangGraph | Agent orchestration (state graph) |
| DeepSeek (via langchain-openai) | LLM for routing, RAG generation, guardrails |

## Setup

```bash
pip install fastapi "uvicorn[standard]" pydantic langgraph langchain-openai python-dotenv
cp .env.example .env  # then edit with your DEEPSEEK_API_KEY
```

## Run

```bash
uvicorn app.main:app --reload
```

- API: `http://localhost:8000`
- Swagger docs: `http://localhost:8000/docs`

Works **with or without** `DEEPSEEK_API_KEY` — falls back to keyword-based mock when no key is set.

## Test

```bash
pip install pytest httpx anyio
python -m pytest tests/ -v
```

## API

```bash
# Q&A
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about laptops"}'

# Order (HITL — two turns with same session_id)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to buy a laptop", "session_id": "s1"}'

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "yes", "session_id": "s1"}'
```
