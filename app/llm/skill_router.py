"""LLM-based skill router — replaces keyword matching in route_skill node."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import ainvoke_json, get_llm_json
from app.llm.prompts import SKILL_ROUTER_PROMPT
from app.llm.types import SkillResult


class SkillRouter:
    """Classifies user messages into 'qa', 'order', or 'track' skill using an LLM.

    Using a class (not a plain function) so the JSON-mode LLM client
    is initialized once and reused across calls.
    """

    def __init__(self) -> None:
        self._llm = get_llm_json()

    async def classify(self, message: str) -> SkillResult:
        messages = [
            SystemMessage(content=SKILL_ROUTER_PROMPT),
            HumanMessage(content=message),
        ]
        return await ainvoke_json(self._llm, messages, SkillResult)
