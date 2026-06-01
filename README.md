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

### With Docker (recommended)

```bash
cp .env.example .env   # then edit with your DEEPSEEK_API_KEY
make build
```

### Without Docker

```bash
pip install fastapi "uvicorn[standard]" pydantic langgraph langchain-openai python-dotenv
cp .env.example .env   # then edit with your DEEPSEEK_API_KEY
```

Requires `DEEPSEEK_API_KEY` (DeepSeek API key).

## Run

### Docker

```bash
make up       # starts with hot reload
make logs     # tail logs
make down     # stop
```

### Local

```bash
uvicorn app.main:app --reload
```

API: `http://localhost:8000` | Swagger: `http://localhost:8000/docs`

## Test

```bash
make test     # in running Docker container
# or locally:
python -m pytest tests/ -v
```

## API requests

Open `api-requests.http` in PyCharm/IntelliJ, select the `dev` environment, and click the green ▶ next to each request.

From CLI:

```bash
npx httpyac api-requests.http --env dev
```

### curl examples

```bash
# Q&A
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about laptops", "session_id": "s1"}'

# Order — step 1: draft
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to buy a ProBook 15", "session_id": "s2"}'

# Order — step 2: confirm
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "yes", "session_id": "s2"}'
```