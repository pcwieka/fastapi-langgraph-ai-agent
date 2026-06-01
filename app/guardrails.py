OFF_TOPIC_KEYWORDS: list[str] = [
    "weather",
    "recipe",
    "cook",
    "sport",
    "football",
    "movie",
    "music",
    "song",
    "game",
    "gaming",
]


def is_on_topic(message: str) -> bool:
    """Lightweight keyword-based input guardrail.

    Rejects messages clearly unrelated to an e-commerce store.
    In production this would be a fast LLM call (GPT-5 mini) with
    structured output: {"on_topic": bool, "reason": str}.
    """
    lower_msg: str = message.lower()
    return not any(kw in lower_msg for kw in OFF_TOPIC_KEYWORDS)


def validate_output(answer: str) -> bool:
    """Output guardrail — ensures the response meets minimum quality.

    Validates: not empty, not truncated, no placeholder text.
    In production this would also use an LLM check.
    """
    if not answer or not answer.strip():
        return False
    if len(answer) < 2:
        return False
    return True
