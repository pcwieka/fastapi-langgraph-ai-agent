"""Pydantic models for structured LLM output.

Using with_structured_output() instead of parsing raw JSON — this gives
type safety and automatic validation. Like @Valid on API responses.
"""

from pydantic import BaseModel, Field


class SkillResult(BaseModel):
    skill: str = Field(..., description="Either 'qa' or 'order'")


class OrderDraftResult(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    total_price: float
    note: str = ""


class GuardrailResult(BaseModel):
    on_topic: bool
    reason: str
