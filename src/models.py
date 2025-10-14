"""Pydantic models for the Fitness RL Agent."""

import json
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Scenario(BaseModel):
    """Training scenario with user context and nutrition targets."""
    question: str
    split: Literal["train", "test"]
    id: str
    daily_cal_target: Optional[int] = None
    daily_prot_target: Optional[int] = None
    daily_carb_target: Optional[int] = None
    daily_fat_target: Optional[int] = None


class FinalAnswer(BaseModel):
    """Final answer from the agent containing the meal plan."""
    answer: Dict[str, Any]

    @field_validator("answer", mode="before")
    @classmethod
    def ensure_dict(cls, v):
        """Ensure answer is a dictionary, handling various input types."""
        # Unwrap nested FinalAnswer by mistake
        if isinstance(v, FinalAnswer):
            return v.answer
        # Parse JSON string
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"answer must be a JSON object string or dict; got invalid JSON: {e}"
                )
        # Already a dict
        if isinstance(v, dict):
            return v
        raise TypeError(f"Unsupported type for answer: {type(v).__name__}")


class FitnessScenario(BaseModel):
    """Wrapper for scenario with training step information."""
    step: int
    scenario: Scenario


class NutritionJudgeResponse(BaseModel):
    """Response from the nutrition quality judge."""
    reasoning: str = Field(description="Explanation of the reasoning process.")
    score: float = Field(description="Score between 0 and 1.")

