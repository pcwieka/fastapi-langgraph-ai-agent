"""LLM-based guardrails - both input (is this on-topic?) and output (is this response valid?)."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_llm
from app.llm.prompts import INPUT_GUARD_PROMPT, OUTPUT_GUARD_PROMPT
from app.llm.types import InputGuardResult, OutputGuardResult


class Guardrail:
    """Input and output guardrails using LLM classification.

    Uses OpenAI's native structured output for guaranteed JSON schema compliance.
    In production, guardrails use a cheaper/faster model than the main agent.
    Here we reuse the same model for both to keep things simple.
    """

    def __init__(self) -> None:
        self._input_llm = get_llm().with_structured_output(InputGuardResult)
        self._output_llm = get_llm().with_structured_output(OutputGuardResult)

    async def check_input(self, message: str, history: list[dict] | None = None) -> InputGuardResult:
        context_text = ""
        if history:
            last_exchanges = history[-4:]  # last 2 turns (user + assistant)
            context_text = (
                "Conversation so far:\n"
                + "\n".join(f"[{m['role']}] {m['content'][:200]}" for m in last_exchanges)
                + "\n\n"
            )

        messages = [
            SystemMessage(content=INPUT_GUARD_PROMPT),
            HumanMessage(content=f"{context_text}User message: {message}"),
        ]
        return await self._input_llm.ainvoke(messages)

    async def check_output(self, answer: str) -> OutputGuardResult:
        messages = [
            SystemMessage(content=OUTPUT_GUARD_PROMPT),
            HumanMessage(content=answer),
        ]
        return await self._output_llm.ainvoke(messages)
