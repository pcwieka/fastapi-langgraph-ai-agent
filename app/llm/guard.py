"""LLM-based input guardrail — replaces keyword matching in guardrails.py."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_llm
from app.llm.prompts import GUARDRAIL_PROMPT
from app.llm.types import GuardrailResult


class LlmGuardrail:
    """Evaluates whether a user message is on-topic using an LLM.

    Uses a fast/cheap model (deepseek-chat with temperature=0) as a classifier.
    In production this would be an even cheaper model (GPT-5 mini equivalent).
    """

    def __init__(self) -> None:
        self._llm = get_llm().with_structured_output(GuardrailResult)

    async def check(self, message: str) -> GuardrailResult:
        messages = [
            SystemMessage(content=GUARDRAIL_PROMPT),
            HumanMessage(content=message),
        ]
        return await self._llm.ainvoke(messages)
