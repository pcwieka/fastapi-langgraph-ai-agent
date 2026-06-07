from langchain_openai import ChatOpenAI


class LlmClient:
    """OpenAI ChatGPT client — wraps ChatOpenAI creation.

    Created once in di.py and injected into LLM components.
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.chat_openai = ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=0.3,
        )
