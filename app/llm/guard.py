"""LLM-based guardrails — both input (is this on-topic?) and output (is this response valid?)."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_llm
from app.llm.prompts import INPUT_GUARD_PROMPT, OUTPUT_GUARD_PROMPT
from app.llm.types import InputGuardResult, OutputGuardResult


class Guardrail:
    """Input and output guardrails using LLM classification.

    In production, guardrails use a cheaper/faster model than the main agent
    (e.g., GPT-5 mini vs GPT-5). Here we reuse deepseek-v4-flash for both.
    """

    def __init__(self) -> None:
        self._input_llm = get_llm().with_structured_output(InputGuardResult)
        self._output_llm = get_llm().with_structured_output(OutputGuardResult)

    async def check_input(self, message: str) -> InputGuardResult:
        messages = [
            SystemMessage(content=INPUT_GUARD_PROMPT),
            HumanMessage(content=message),
        ]
        return await self._input_llm.ainvoke(messages)

    async def check_output(self, answer: str) -> OutputGuardResult:
        messages = [
            SystemMessage(content=OUTPUT_GUARD_PROMPT),
            HumanMessage(content=answer),
        ]
        return await self._output_llm.ainvoke(messages)
