import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app, sessions


@pytest.fixture(autouse=True)
def clear_sessions() -> None:
    """Reset session store between tests to avoid cross-test state leaks."""
    sessions.clear()


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_qa_product_search(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"message": "Tell me about laptops"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "ProBook" in body["answer"]
    assert len(body["sources"]) > 0


@pytest.mark.anyio
async def test_qa_no_match(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"message": "Do you sell furniture?"}
    )
    assert response.status_code == 200
    body = response.json()
    # Returns all products when no match (fallback catalog)
    assert len(body["sources"]) > 0


@pytest.mark.anyio
async def test_order_first_turn_prepare_draft(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json={"message": "I want to buy a laptop", "session_id": "s1"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "order summary" in body["answer"].lower()
    assert "confirm" in body["answer"].lower()


@pytest.mark.anyio
async def test_order_second_turn_confirm(client: AsyncClient) -> None:
    # First turn — prepare order
    await client.post(
        "/chat", json={"message": "I want to buy a laptop", "session_id": "s2"}
    )
    # Second turn — confirm
    response = await client.post(
        "/chat", json={"message": "yes", "session_id": "s2"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "confirmed" in body["answer"].lower()


@pytest.mark.anyio
async def test_order_second_turn_cancel(client: AsyncClient) -> None:
    await client.post(
        "/chat", json={"message": "I want to buy a laptop", "session_id": "s3"}
    )
    response = await client.post(
        "/chat", json={"message": "no", "session_id": "s3"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "cancelled" in body["answer"].lower()


@pytest.mark.anyio
async def test_off_topic_rejected_by_guardrail(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"message": "What is the weather today?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "only help with product questions and orders" in body["answer"]
    assert len(body["sources"]) == 0


@pytest.mark.anyio
async def test_empty_message_rejected_by_pydantic(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422


@pytest.mark.anyio
async def test_session_id_preserved(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json={"message": "Hi", "session_id": "my-session-123"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session-123"
