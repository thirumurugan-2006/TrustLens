from pydantic import BaseModel, Field


class ValidationResult(BaseModel):
    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    quality_level: str | None = None
    metadata: dict = Field(default_factory=dict)
