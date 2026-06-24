# E-commerce AI Agent - Agentic RAG + HITL

A **multi-skill AI agent** built with **FastAPI**, **LangGraph**, **OpenAI ChatGPT**, and **ChromaDB**. The agent routes each request to the right skill - Q&A (RAG), Order (HITL), or Track - and orchestrates the full conversation flow. 7 nodes, 3 skills, 1 checkpointer, ChromaDB-backed semantic retrieval.

## Skills

Each **skill** is a dedicated path through the LangGraph state graph. An LLM classifier picks the right one on every request:

| Skill | Trigger | What happens |
|-------|---------|-------------|
| **Q&A (RAG)** | "Tell me about laptops" | Classify → search catalog → LLM composes answer from results |
| **Order (HITL)** | "I want to buy a ProBook 15" | LLM extracts product → draft → interrupt → confirm → place |
| **Track** | "Where is my order?" | Look up order registry by session → return status + ETA |

Adding a new skill means adding one node + one edge - zero changes to existing skills.

Every request passes through **input and output guardrails** (LLM-based validation) before reaching the agent and before returning to the user.

## Architecture highlights

- **Multi-skill agent** - LLM-powered skill router classifies intent, conditional edges route to the right skill path
- **Agentic RAG** - the agent decides when to run semantic retrieval (not every request goes through the vector store)
- **Real RAG with ChromaDB** - offline indexing (embed the catalog once) + online retrieval (embed the query, top-k by cosine distance); vectors live in a separate ChromaDB container
- **Human-in-the-Loop** - orders require explicit user confirmation via LangGraph's native `interrupt()` / `Command(resume=...)` pattern
- **Persistent state** - graph execution state survives server restarts (SQLite checkpointer)
- **Package-by-feature** - `product/` and `order/` domains each own their repository (ABC interface — ChromaDB for products, in-memory for orders) and service layer
- **Structured logging** - per-request timing breakdown: guardrail latency, graph execution time, total request duration; retrieval logs top-k hits with cosine distances

## RAG pipeline (index + retrieval)

Two separate phases share the same ChromaDB collection and the same embedding model. **Indexing** is an offline batch job (`make index`); **retrieval** runs online per request. Embeddings are computed client-side — the ChromaDB server only stores and compares vectors.

```mermaid
flowchart LR
    subgraph Offline["① Indexing — offline (make index, run once)"]
        Cat[data/products.json<br/>20 products] --> Idx[embedding/index.py]
        Idx -->|embed each product| EmbA[OpenAI<br/>text-embedding-3-small]
        EmbA -->|vectors| Coll[("ChromaDB<br/>collection 'products'<br/>cosine / HNSW")]
    end

    subgraph Online["② Retrieval — online (per /chat request)"]
        Q[User query] --> EmbB[OpenAI<br/>text-embedding-3-small]
        EmbB -->|query vector| Coll
        Coll -->|top-k by cosine distance| Hits[top-k product ids]
        Hits --> Map[map ids → catalog]
        Map -->|context| Gen[generate_qa_answer<br/>LLM composes answer]
        Gen --> Ans[Answer + sources]
    end

    style Offline fill:#eef2f7,stroke:#5b6b8c,color:#1a1a2e
    style Online fill:#eaf3ec,stroke:#3a7d54,color:#1a1a2e
    style Coll fill:#533483,color:#fff
```

## Agent graph (7 nodes, 3 paths)

```mermaid
flowchart LR
    Start(( )) --> Router[route_skill<br/>LLM: qa / order / track]

    Router -- "qa" --> Search[search_products<br/>ChromaDB top-k]
    Search --> GenQA[generate_qa_answer<br/>LLM RAG]
    GenQA --> End1(( ))

    Router -- "order" --> Prep[prepare_order<br/>LLM draft]
    Prep --> Await[await_confirmation<br/>⏸️ interrupt]
    Await -- "⏯️ resume" --> Final[finalize_order]
    Final --> End2(( ))

    Router -- "track" --> Track[track_order]
    Track --> End3(( ))

    style Router fill:#533483,color:#fff
    style Search fill:#1a1a2e,color:#fff,stroke:#0f3460
    style GenQA fill:#16213e,color:#fff
    style Prep fill:#16213e,color:#fff
    style Await fill:#e94560,color:#fff
    style Final fill:#16213e,color:#fff
    style Track fill:#1a1a2e,color:#fff,stroke:#0f3460
```

## Architecture

```mermaid
flowchart TB
    subgraph Client[" "]
        User[User]
    end

    subgraph FastAPI["FastAPI"]
        Main[main.py]
        GuardIn[Input Guardrail]
        GuardOut[Output Guardrail]
        Logger[Structured Logger]
    end

    subgraph LangGraph["LangGraph Agent"]
        Router[route_skill<br/>LLM classifier]
        QA1[search_products]
        QA2[generate_qa_answer<br/>LLM]
        Ord1[prepare_order<br/>LLM draft]
        Ord2[await_confirmation<br/>interrupt]
        Ord3[finalize_order]
        Track[track_order]
        CP[("Checkpointer<br/>SQLite")]
    end

    subgraph LLM["LLM Layer (OpenAI)"]
        OpenAI[ChatGPT<br/>gpt-4o-mini]
        Embed[Embeddings<br/>text-embedding-3-small]
    end

    subgraph Domains["Domain Packages"]
        Product[product/<br/>ChromaProductRepository + Service]
        Order[order/<br/>InMemory Repository + Service]
    end

    subgraph Vector["Vector store (separate container)"]
        Chroma[("ChromaDB<br/>collection 'products'")]
    end

    subgraph Offline["Offline indexing (make index)"]
        Catalog[data/products.json] --> Indexer[embedding/index.py]
    end

    User -->|"POST /chat"| Main
    Main --> GuardIn
    GuardIn -->|allowed| Router
    GuardIn -->|blocked| User
    Router --> QA1 --> QA2
    Router --> Ord1 --> Ord2 --> Ord3
    Router --> Track
    QA2 --> GuardOut --> User
    Ord3 --> GuardOut --> User
    Track --> GuardOut --> User
    QA2 -.->|LLM call| OpenAI
    Ord1 -.->|LLM call| OpenAI
    Router -.->|LLM call| OpenAI
    GuardIn -.->|LLM call| OpenAI
    GuardOut -.->|LLM call| OpenAI
    LangGraph --> CP
    QA1 --> Product
    Ord1 --> Product
    Ord3 --> Order
    Track --> Order
    Product -->|"semantic search (top-k)"| Chroma
    Product -.->|embed query| Embed
    Indexer -.->|embed catalog| Embed
    Indexer -->|store vectors| Chroma

    style Client fill:#f0f0f3,stroke:#888,color:#1a1a2e
    style FastAPI fill:#e8eef7,stroke:#5b6b8c,color:#1a1a2e
    style LangGraph fill:#efeaf8,stroke:#7a6aa8,color:#1a1a2e
    style LLM fill:#f7eef0,stroke:#b06a80,color:#1a1a2e
    style Domains fill:#eef5ee,stroke:#5b8c6b,color:#1a1a2e
    style Vector fill:#f7f1e0,stroke:#b09a4a,color:#1a1a2e
    style Offline fill:#f0f0f3,stroke:#888,color:#1a1a2e
```

## HITL flow (LangGraph interrupt / resume)

```mermaid
sequenceDiagram
    actor User
    box FastAPI + LangGraph
        participant Main as main.py
        participant Graph as Agent Graph
        participant CP as Checkpointer (SQLite)
    end

    User->>Main: POST "I want to buy a ProBook 15"
    Main->>CP: get_state(config)
    CP-->>Main: next=() → new run
    Main->>Graph: ainvoke(initial_state, config)
    activate Graph
    Graph->>Graph: route_skill → "order"
    Graph->>Graph: prepare_order → draft
    Graph->>Graph: await_confirmation → interrupt()
    Graph-->>CP: save state
    deactivate Graph
    CP-->>Main: next=("await_confirmation",)
    Main-->>User: "Confirm? ProBook 15 - $1299.99 (yes/no)"

    User->>Main: POST "yes"
    Main->>CP: get_state(config)
    CP-->>Main: next=("await_confirmation",) → resume
    Main->>Graph: ainvoke(Command(resume="yes"), config)
    activate Graph
    Graph->>Graph: ⏯️ resume await_confirmation
    Graph->>Graph: finalize_order → place order
    Graph-->>CP: save state
    deactivate Graph
    CP-->>Main: next=() → done
    Main-->>User: "Order ORD-1000 confirmed!"
```

## Project structure

```
app/
├── agent/          # LangGraph graph (AgentGraph), skills (AgentSkills), state
├── config/         # DI wiring — module-level singletons (di.py)
├── product/        # ProductRepository (ABC + ChromaDB) + ProductService + catalog loader
├── order/          # OrderRepository (ABC + InMemory) + OrderService
├── llm/            # LlmClient, prompts, guardrail, skill router, response generators
├── main.py         # FastAPI entry point + lifespan
├── models.py       # Pydantic request/response
└── logger.py       # Structured logging with timing
embedding/
├── index.py        # Offline RAG indexer — embeds the catalog into ChromaDB (`make index`)
└── inspect.py      # Inspect stored vectors/docs (`make inspect-documents` / `inspect-embeddings`)
data/
├── products.json   # Canonical product catalog (system of record, 20 products)
└── checkpoints.db  # SQLite checkpointer (gitignored, mounted in Docker)
tests/
├── test_main.py    # 10 integration tests (HTTP + mocked LLM and vector search)
├── test_product.py # 9 unit tests (catalog loader + ChromaDB repo with mocked collection)
└── test_order.py   # 8 unit tests
```

## Stack

| Component | Technology |
|-----------|-----------|
| API framework | FastAPI (async) |
| Agent orchestration | LangGraph (state graph with 7 nodes, 3 conditional paths) |
| LLM | OpenAI ChatGPT gpt-4o-mini (via langchain-openai) |
| RAG retrieval | ChromaDB 1.5.9 (separate container) + OpenAI embeddings (text-embedding-3-small) |
| Data validation | Pydantic |
| State persistence | LangGraph SQLite checkpointer (+ InMemorySaver for tests) |
| Linting & formatting | ruff |
| Containerization | Docker + docker-compose |
| Package management | uv |

## Quick start

```bash
# 1. Set your API key
cp .env.example .env   # edit with your OPENAI_API_KEY

# 2. Build and run (app on :8000, ChromaDB on :8001)
make build
make up                # starts app + chroma with hot reload

# 3. Build the vector index — embeds the catalog into ChromaDB (run once)
make index             # re-run whenever data/products.json changes

# 4. Test it
# Open api-requests.http in PyCharm/IntelliJ, select "dev" environment,
# and click the green ▶ next to each request.
```

> **Note:** Q&A requires the index. If you skip `make index`, product search has
> no collection to query. The indexer is a one-off batch job, kept out of the
> request path so startup stays fast and embeddings are computed exactly once.

API: `http://localhost:8000` | Swagger: `http://localhost:8000/docs`

## Commands

```
make up                 start app + chroma with hot reload
make index              build the vector index (embed catalog into ChromaDB)
make inspect-documents  list stored docs + metadata in ChromaDB
make inspect-embeddings same, plus a preview of each stored vector
make down               stop services
make logs               tail logs
make test               run tests locally
make lint               ruff check
make format             ruff format
make build              rebuild Docker image
```

## Skill router evaluation

The skill router is LLM-powered, so accuracy is measured with a frozen test set.

```bash
make eval   # runs evaluation/skill_router.eval.py — hits real OpenAI API
```

**Results (gpt-4o-mini, 56 test cases):**

| Skill | Cases | Correct | Accuracy |
|-------|-------|---------|----------|
| Q&A   | 17    | 17      | 100%     |
| Order | 26    | 20      | 76.9%    |
| Track | 13    | 13      | 100%     |
| **Total** | **56** | **50** | **88.89%** |

**Main failure pattern — Order → Track (6 cases):**

The router misclassifies order-management intent as tracking when the message contains order-status vocabulary:

```
"I want to cancel my order"                        expected=order  got=track
"I want to return my order"                        expected=order  got=track
"Return the headphones I bought"                   expected=order  got=track
"I want to cancel order #123"                      expected=order  got=track
"I want to return the item, what is the status?"   expected=order  got=track
"I am not happy with my order, I want a refund"    expected=order  got=track
```

Root cause: the prompt doesn't explicitly distinguish *managing* an order (cancel, return, refund) from *tracking* it (status, ETA, location). A prompt update or few-shot examples would fix this.

## API

```bash
# Product Q&A
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about laptops", "session_id": "s1"}'

# Place an order (step 1 - draft)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "I want to buy a ProBook 15", "session_id": "s2"}'

# Confirm order (step 2)
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "yes", "session_id": "s2"}'

# Track order
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Where is my order?", "session_id": "s2"}'
```

Or use `api-requests.http` with IntelliJ HTTP Client or `npx httpyac api-requests.http --env dev`.
