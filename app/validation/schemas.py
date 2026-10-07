from enum import Enum
from pydantic import BaseModel, Field, model_validator


class ValidationStatus(str, Enum):
    VALID = "VALID"
    INVALID = "INVALID"
    LOW_QUALITY = "LOW_QUALITY"
    UNSUPPORTED = "UNSUPPORTED"
    INSUFFICIENT = "INSUFFICIENT"


class ValidationResult(BaseModel):
    valid: bool
    status: ValidationStatus = ValidationStatus.VALID
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    quality_score: float | None = None
    quality_level: str | None = None
    metadata: dict = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def infer_status(cls, data: dict) -> dict:
        if not isinstance(data, dict):
            return data
        if "status" not in data or data["status"] is None:
            valid = data.get("valid", True)
            errors = data.get("errors", [])
            warnings = data.get("warnings", [])
            quality_level = data.get("quality_level")

            if not valid:
                err_text = " ".join(str(e) for e in errors).lower()
                if "unsupported" in err_text or "unknown" in err_text:
                    data["status"] = ValidationStatus.UNSUPPORTED
                elif "empty" in err_text or "insufficient" in err_text or "required" in err_text:
                    data["status"] = ValidationStatus.INSUFFICIENT
                else:
                    data["status"] = ValidationStatus.INVALID
            else:
                if quality_level in ("low", "failed") or any("low" in str(w).lower() for w in warnings):
                    data["status"] = ValidationStatus.LOW_QUALITY
                else:
                    data["status"] = ValidationStatus.VALID
        return data
