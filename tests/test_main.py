import os

# Must be set BEFORE importing app — LlmClient reads OPENAI_API_KEY at import time.
os.environ["OPENAI_API_KEY"] = "test-key"

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture(autouse=True)
def setup_graph_and_state() -> None:
    """Set up InMemorySaver checkpointer and clear order repo between tests."""
    from langgraph.checkpoint.memory import InMemorySaver

    import app.main as main_module
    from app.config.di import agent_graph_builder, order_repo

    main_module.agent = agent_graph_builder.build(InMemorySaver())
    order_repo.reset()


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _mock_guard(input_on_topic: bool, output_valid: bool = True):
    """Mock both input and output guardrail checks."""
    from app.llm.types import InputGuardResult, OutputGuardResult

    async def check_input(self, message: str, history=None) -> InputGuardResult:
        reason = "ok" if input_on_topic else "off-topic"
        return InputGuardResult(on_topic=input_on_topic, reason=reason)

    async def check_output(self, answer: str) -> OutputGuardResult:
        return OutputGuardResult(valid=output_valid, reason="ok" if output_valid else "invalid")

    return (
        patch("app.llm.guardrail.Guardrail.check_input", check_input),
        patch("app.llm.guardrail.Guardrail.check_output", check_output),
    )


def _mock_skill(skill: str):
    from app.llm.types import SkillResult

    return patch(
        "app.llm.skill_router.SkillRouter.classify",
        AsyncMock(return_value=SkillResult(skill=skill)),
    )


def _mock_qa_answer(response: str):
    return patch(
        "app.llm.response_generator.QaResponseGenerator.generate",
        AsyncMock(return_value=response),
    )


def _mock_product_search(products: list[dict] | None = None):
    """Stub the vector search so Q&A tests don't need a live ChromaDB server."""
    if products is None:
        products = [
            {
                "id": "probook-15",
                "name": "ProBook 15",
                "brand": "TechCorp",
                "price": 1299.99,
                "stock": 5,
                "description": "Business laptop.",
            }
        ]
    return patch("app.product.service.ProductService.search", return_value=products)


def _mock_order_draft():
    from app.llm.types import OrderDraftResult

    result = OrderDraftResult(
        product_id="probook-15", product_name="ProBook 15", quantity=1, total_price=1299.99, note=""
    )
    return patch(
        "app.llm.response_generator.OrderDraftGenerator.generate",
        AsyncMock(return_value=result),
    )


@pytest.mark.anyio
async def test_qa_product_search(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with (
        g1,
        g2,
        _mock_skill("qa"),
        _mock_product_search(),
        _mock_qa_answer("LLM: ProBook 15 is a great laptop"),
    ):
        response = await client.post("/chat", json={"message": "Tell me about laptops"})
    assert response.status_code == 200
    assert "ProBook" in response.json()["answer"]
    assert len(response.json()["sources"]) > 0


@pytest.mark.anyio
async def test_qa_no_match(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("qa"), _mock_product_search([]), _mock_qa_answer("LLM: no products found"):
        response = await client.post("/chat", json={"message": "Do you sell furniture?"})
    assert response.status_code == 200
    assert "no products" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_order_first_turn_prepare_draft(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("order"), _mock_order_draft():
        response = await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s1"})
    assert response.status_code == 200
    assert "order summary" in response.json()["answer"].lower()
    assert "confirm" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_order_second_turn_confirm(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("order"), _mock_order_draft():
        await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s2"})
    # On resume, route_skill is skipped - no _mock_skill needed
    with g1, g2:
        response = await client.post("/chat", json={"message": "yes", "session_id": "s2"})
    assert response.status_code == 200
    assert "confirmed" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_order_second_turn_cancel(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("order"), _mock_order_draft():
        await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s3"})
    # On resume, route_skill is skipped - no _mock_skill needed
    with g1, g2:
        response = await client.post("/chat", json={"message": "no", "session_id": "s3"})
    assert response.status_code == 200
    assert "cancelled" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_off_topic_rejected_by_guardrail(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=False)
    with g1, g2:
        response = await client.post("/chat", json={"message": "What is the weather today?"})
    assert response.status_code == 200
    assert "only help with product questions and orders" in response.json()["answer"]


@pytest.mark.anyio
async def test_empty_message_rejected_by_pydantic(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_session_id_preserved(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("qa"), _mock_product_search(), _mock_qa_answer("LLM: hello"):
        response = await client.post("/chat", json={"message": "Hi", "session_id": "my-session-123"})
    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session-123"


@pytest.mark.anyio
async def test_track_order_no_orders(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    with g1, g2, _mock_skill("track"):
        response = await client.post("/chat", json={"message": "Where is my order?", "session_id": "s9"})
    assert response.status_code == 200
    assert "couldn't find any orders" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_track_order_after_confirm(client: AsyncClient) -> None:
    g1, g2 = _mock_guard(input_on_topic=True)
    # Place an order and confirm it
    with g1, g2, _mock_skill("order"), _mock_order_draft():
        await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s10"})
    # On resume, route_skill is skipped - no _mock_skill needed
    with g1, g2:
        await client.post("/chat", json={"message": "yes", "session_id": "s10"})

    # Now track it
    with g1, g2, _mock_skill("track"):
        response = await client.post("/chat", json={"message": "Where is my order?", "session_id": "s10"})
    assert response.status_code == 200
    body = response.json()
    assert "ORD-1000" in body["answer"]
    assert "ProBook 15" in body["answer"]
    assert "processing" in body["answer"].lower()
