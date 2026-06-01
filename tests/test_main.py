from unittest.mock import AsyncMock, MagicMock, patch

import os
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app, sessions


@pytest.fixture(autouse=True)
def clear_sessions() -> None:
    sessions.clear()


@pytest.fixture(autouse=True)
def set_fake_api_key() -> None:
    """Set a fake API key so get_llm() constructor doesn't raise."""
    os.environ["DEEPSEEK_API_KEY"] = "test-key"


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


def _mock_guard(result: object) -> AsyncMock:
    return AsyncMock(return_value=result)


def _mock_skill(result: object) -> AsyncMock:
    return AsyncMock(return_value=result)


def _mock_qa_gen(response: str) -> AsyncMock:
    return AsyncMock(return_value=response)


def _mock_order_gen(result: object) -> AsyncMock:
    return AsyncMock(return_value=result)


@pytest.mark.anyio
async def test_qa_product_search(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult, SkillResult

    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="qa"))),
        patch("app.llm.response_generator.QaResponseGenerator.generate", _mock_qa_gen("LLM: ProBook 15 is a great laptop")),
    ):
        response = await client.post("/chat", json={"message": "Tell me about laptops"})

    assert response.status_code == 200
    body = response.json()
    assert "ProBook" in body["answer"]
    assert len(body["sources"]) > 0


@pytest.mark.anyio
async def test_qa_no_match(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult, SkillResult

    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="qa"))),
        patch("app.llm.response_generator.QaResponseGenerator.generate", _mock_qa_gen("LLM: no products found")),
    ):
        response = await client.post("/chat", json={"message": "Do you sell furniture?"})

    assert response.status_code == 200
    body = response.json()
    assert "no products" in body["answer"].lower()


@pytest.mark.anyio
async def test_order_first_turn_prepare_draft(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult, OrderDraftResult, SkillResult

    draft = OrderDraftResult(product_id="probook-15", product_name="ProBook 15", quantity=1, total_price=1299.99, note="")
    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="order"))),
        patch("app.llm.response_generator.OrderDraftGenerator.generate", _mock_order_gen(draft)),
    ):
        response = await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s1"})

    assert response.status_code == 200
    body = response.json()
    assert "order summary" in body["answer"].lower()
    assert "confirm" in body["answer"].lower()


@pytest.mark.anyio
async def test_order_second_turn_confirm(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult, OrderDraftResult, SkillResult

    draft = OrderDraftResult(product_id="probook-15", product_name="ProBook 15", quantity=1, total_price=1299.99, note="")
    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="order"))),
        patch("app.llm.response_generator.OrderDraftGenerator.generate", _mock_order_gen(draft)),
    ):
        await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s2"})

    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="order"))),
    ):
        response = await client.post("/chat", json={"message": "yes", "session_id": "s2"})

    assert response.status_code == 200
    assert "confirmed" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_order_second_turn_cancel(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult, OrderDraftResult, SkillResult

    draft = OrderDraftResult(product_id="probook-15", product_name="ProBook 15", quantity=1, total_price=1299.99, note="")
    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="order"))),
        patch("app.llm.response_generator.OrderDraftGenerator.generate", _mock_order_gen(draft)),
    ):
        await client.post("/chat", json={"message": "I want to buy a laptop", "session_id": "s3"})

    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="order"))),
    ):
        response = await client.post("/chat", json={"message": "no", "session_id": "s3"})

    assert response.status_code == 200
    assert "cancelled" in response.json()["answer"].lower()


@pytest.mark.anyio
async def test_off_topic_rejected_by_guardrail(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult

    with patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=False, reason="off-topic"))):
        response = await client.post("/chat", json={"message": "What is the weather today?"})

    assert response.status_code == 200
    assert "only help with product questions and orders" in response.json()["answer"]


@pytest.mark.anyio
async def test_empty_message_rejected_by_pydantic(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_session_id_preserved(client: AsyncClient) -> None:
    from app.llm.types import GuardrailResult, SkillResult

    with (
        patch("app.llm.guard.LlmGuardrail.check", _mock_guard(GuardrailResult(on_topic=True, reason="ok"))),
        patch("app.llm.skill_router.SkillRouter.classify", _mock_skill(SkillResult(skill="qa"))),
        patch("app.llm.response_generator.QaResponseGenerator.generate", _mock_qa_gen("LLM: hello")),
    ):
        response = await client.post("/chat", json={"message": "Hi", "session_id": "my-session-123"})

    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session-123"
