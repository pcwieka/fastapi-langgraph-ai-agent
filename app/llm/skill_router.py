"""LLM-based skill router — replaces keyword matching in route_skill node."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_llm
from app.llm.prompts import SKILL_ROUTER_PROMPT
from app.llm.types import SkillResult


class SkillRouter:
    """Classifies user messages into 'qa' or 'order' skill using an LLM.

    Using a class (not a plain function) so the LLM client is initialized once.
    In production this would be a Spring-style singleton bean.
    """

    def __init__(self) -> None:
        self._llm = get_llm().with_structured_output(SkillResult)

    async def classify(self, message: str) -> SkillResult:
        messages = [
            SystemMessage(content=SKILL_ROUTER_PROMPT),
            HumanMessage(content=message),
        ]
        return await self._llm.ainvoke(messages)
