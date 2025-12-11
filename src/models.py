from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class IntakeInstruction(BaseModel):
    """One block of instructions within a sig.

    Example (from the brief):
    - "2x tabs/3d then 1x/2d" -> two instructions with doses and durations.
    """

    intakes: int = Field(..., description="Number of units taken per intake period (e.g. 2 capsules)")
    intake_period: str = Field(
        ...,
        description="ISO-8601 duration for how often the intakes occur, e.g. 'P1D' for once per day.",
    )
    intake_type: str = Field(..., description="Dosage form, e.g. 'tablet', 'capsule', 'ml'.")
    duration: str = Field(
        ...,
        description="ISO-8601 duration for how long this instruction is followed, e.g. 'P3D' for 3 days.",
    )


class StructuredSig(BaseModel):
    """Structured representation of a complete sig.

    This is the JSON schema we want the LLM to produce.
    """

    sigs: List[IntakeInstruction]


class TranslationResult(BaseModel):
    """Output of the sig translation stage for a single row."""

    english_instructions: str
    structured: StructuredSig


class ValidationResult(BaseModel):
    """Output of the validation stage for a single row."""

    decision: str  # "OK" or "NOT_OK"
    reason: str
    emoji: str  # "✅" or "❌"
