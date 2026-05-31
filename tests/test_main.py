import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.fixture
async def client() -> AsyncClient:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.mark.anyio
async def test_support_kb_hit(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"message": "What are your opening hours?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "open" in body["answer"].lower()
    assert len(body["sources"]) > 0
    assert body["session_id"] is not None


@pytest.mark.anyio
async def test_general_skips_kb(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"message": "Hello! How are you?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["sources"]) == 0
    assert body["session_id"] is not None


@pytest.mark.anyio
async def test_off_topic_rejected_by_guardrail(client: AsyncClient) -> None:
    response = await client.post(
        "/chat", json={"message": "What is the weather today?"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "only answer questions about our services" in body["answer"]
    assert len(body["sources"]) == 0


@pytest.mark.anyio
async def test_empty_message_rejected_by_pydantic(client: AsyncClient) -> None:
    response = await client.post("/chat", json={"message": ""})
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.anyio
async def test_session_id_preserved(client: AsyncClient) -> None:
    response = await client.post(
        "/chat",
        json={"message": "Hi", "session_id": "my-session-123"},
    )
    assert response.status_code == 200
    assert response.json()["session_id"] == "my-session-123"
