"""LLM-based response generators - replace string formatting in agent nodes."""

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import ainvoke_json, get_llm, get_llm_json
from app.llm.prompts import ORDER_DRAFT_PROMPT, QA_ANSWER_PROMPT
from app.llm.types import OrderDraftResult


class QaResponseGenerator:
    """Generates product Q&A answers using LLM with search results as context.

    This is the RAG generation step - the LLM receives product data as context
    and composes a natural language answer.
    """

    def __init__(self) -> None:
        self._llm = get_llm()

    async def generate(self, user_message: str, products: list[dict]) -> str:
        if not products:
            return "I couldn't find any products matching your query."

        context = "\n".join(
            f"- {p['name']} ({p['brand']}): ${p['price']:.2f} | stock: {p['stock']} | {p['description']}"
            for p in products
        )
        prompt = QA_ANSWER_PROMPT.format(product_context=context)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]
        response = await self._llm.ainvoke(messages)
        return str(response.content)


class OrderDraftGenerator:
    """Extracts product and quantity from user's order request using LLM.

    Uses JSON mode (response_format json_object) + manual Pydantic parsing
    because DeepSeek does not support structured output (json_schema).
    """

    def __init__(self) -> None:
        self._llm = get_llm_json()

    async def generate(
        self, user_message: str, product_catalog: list[dict], history: list[dict] | None = None
    ) -> OrderDraftResult:
        catalog_text = "\n".join(
            f"- id: {p['id']} | {p['name']} | ${p['price']:.2f} | stock: {p['stock']}"
            for p in product_catalog
        )
        if history:
            conv_text = "Conversation history:\n" + "\n".join(
                f"[{m['role']}] {m['content'][:300]}" for m in history[-6:]
            )
        else:
            conv_text = "(no previous conversation)"
        prompt = ORDER_DRAFT_PROMPT.format(conversation_history=conv_text, product_catalog=catalog_text)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=user_message),
        ]
        return await ainvoke_json(self._llm, messages, OrderDraftResult)
