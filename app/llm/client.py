import os
from functools import lru_cache
from typing import TypeVar

from langchain_openai import ChatOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@lru_cache
def get_llm() -> ChatOpenAI:
    """DeepSeek client factory — singleton via lru_cache.

    DeepSeek API is OpenAI-compatible. We use ChatOpenAI with a custom
    base_url. The API key comes from the DEEPSEEK_API_KEY env var.
    """
    api_key: str = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY environment variable is required for LLM calls. "
            "Set it in .env or export it in your shell."
        )

    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )


@lru_cache
def get_llm_json() -> ChatOpenAI:
    """LLM client for structured output via JSON prompting.

    DeepSeek does not support response_format (neither json_schema nor json_object).
    We use temperature=0 for deterministic output and rely on the prompt
    requesting JSON explicitly (e.g. "Reply with JSON: {...}").
    """
    api_key: str = os.environ.get("DEEPSEEK_API_KEY", "")
    return ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=api_key,
        base_url="https://api.deepseek.com",
        temperature=0.0,
    )


async def ainvoke_json(llm: ChatOpenAI, messages: list, model: type[T]) -> T:
    """Invoke LLM and parse the response as JSON into a Pydantic model."""
    response = await llm.ainvoke(messages)
    content = str(response.content).strip()
    # Strip markdown code fences if present
    if content.startswith("```"):
        content = content.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return model.model_validate_json(content)
