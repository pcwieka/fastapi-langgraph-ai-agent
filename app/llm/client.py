import os
from functools import lru_cache

from langchain_openai import ChatOpenAI


@lru_cache
def get_llm() -> ChatOpenAI:
    """DeepSeek client factory — singleton via lru_cache (like Spring @Service).

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
