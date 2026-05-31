MOCK_KB: dict[str, str] = {
    "opening hours": "We are open Mon-Fri 9:00-17:00. Weekend support is ilable via email.",
    "pricing": "Our pricing plans start at $29/month for the Basic tier and $99/month for Pro.",
    "returns": "You can return any item within 30 days. Contact support to initiate a return.",
}


def search_knowledgebase(query: str) -> list[str]:
    """Mock KB search — simulates Azure AI Search call from the production chatbot.

    In production this would call Azure AI Search with semantic ranking.
    Here we fake it with a dictionary lookup.
    """
    results: list[str] = []
    for keyword, article in MOCK_KB.items():
        if keyword.lower() in query.lower():
            results.append(article)
    if not results:
        results.append("I couldn't find relevant articles for your query.")
    return results
