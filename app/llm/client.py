import os
from functools import lru_cache

from langchain_openai import ChatOpenAI


@lru_cache
def get_llm() -> ChatOpenAI:
    """OpenAI ChatGPT client factory - singleton via lru_cache.

    Uses OPENAI_API_KEY and OPENAI_MODEL env vars. Defaults to gpt-4o-mini.
    """
    api_key: str = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is required. "
            "Set it in .env or export it in your shell."
        )

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.3,
    )
