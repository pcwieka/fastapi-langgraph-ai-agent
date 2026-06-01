"""Pydantic models for structured LLM output.

Using with_structured_output() instead of parsing raw JSON — this gives
type safety and automatic validation.
"""

from pydantic import BaseModel, Field


class SkillResult(BaseModel):
    skill: str = Field(..., description="Either 'qa', 'order', or 'track'")


class OrderDraftResult(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    total_price: float
    note: str = ""


class InputGuardResult(BaseModel):
    on_topic: bool
    reason: str


class OutputGuardResult(BaseModel):
    valid: bool
    reason: str
