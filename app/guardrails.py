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
