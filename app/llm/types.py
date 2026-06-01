"""Pydantic models for LLM JSON output - parsed via model_validate_json()."""

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
