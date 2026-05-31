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
    """Lightweight keyword-based guardrail.

    In the production chatbot this is a GPT-5 mini call evaluating
    whether the message is within the agent's scope.
    """
    lower_msg: str = message.lower()
    return not any(kw in lower_msg for kw in OFF_TOPIC_KEYWORDS)
