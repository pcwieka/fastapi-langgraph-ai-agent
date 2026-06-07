"""Dependency injection — module-level singletons wired together.

One instance of each component, shared across all requests.
Tests import these directly and call order_repo.reset() between cases.
"""

import os

from app.agent.graph import AgentGraph
from app.agent.skills import AgentSkills
from app.llm.client import LlmClient
from app.llm.guardrail import Guardrail
from app.llm.response_generator import OrderDraftGenerator, QaResponseGenerator
from app.llm.skill_router import SkillRouter
from app.order.repository import InMemoryOrderRepository
from app.order.service import OrderService
from app.product.repository import InMemoryProductRepository
from app.product.service import ProductService

# Repositories (infrastructure)
product_repo = InMemoryProductRepository()
order_repo = InMemoryOrderRepository()

# Domain services
product_service = ProductService(product_repo)
order_service = OrderService(order_repo)

# LLM client — one ChatOpenAI shared across all LLM components
llm_client = LlmClient(
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
)

# LLM components — each receives the same ChatOpenAI instance
skill_router = SkillRouter(llm_client.chat_openai)
qa_generator = QaResponseGenerator(llm_client.chat_openai)
order_generator = OrderDraftGenerator(llm_client.chat_openai)
guardrail = Guardrail(llm_client.chat_openai)

# Agent skills — receives everything it needs via constructor
skills = AgentSkills(
    skill_router=skill_router,
    qa_generator=qa_generator,
    order_generator=order_generator,
    product_service=product_service,
    order_service=order_service,
)

# Agent graph builder — wires skills into the LangGraph graph
agent_graph_builder = AgentGraph(skills=skills)
