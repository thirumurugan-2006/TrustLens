from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

from app.training.schemas import ProvenanceMetadata, ProvenanceSourceType


class ProvenanceError(ValueError):
    """Raised when provenance attributes are missing, invalid, or unverifiable."""
    pass


class ProvenanceManager:
    """
    Manages provenance tracking, validation, and assignment for TrustLens datasets.
    Guarantees that every imported record carries unambiguous origin data without invention.
    """

    ALLOWED_SOURCE_TYPES = {st.value for st in ProvenanceSourceType}

    @classmethod
    def create_provenance(
        cls,
        source_type: Union[str, ProvenanceSourceType],
        source_name: str,
        source_id: Optional[str] = None,
        collection_method: str = "automated_import",
        license_str: Optional[str] = None,
        original_language: str = "en",
        original_location: Optional[str] = None,
        collection_date: Optional[str] = None,
    ) -> ProvenanceMetadata:
        """
        Creates and strictly validates a ProvenanceMetadata object.
        """
        if isinstance(source_type, ProvenanceSourceType):
            st_val = source_type.value
        else:
            st_val = str(source_type).upper().strip()

        if st_val not in cls.ALLOWED_SOURCE_TYPES:
            raise ProvenanceError(
                f"Invalid source_type '{source_type}'. Allowed types: {sorted(list(cls.ALLOWED_SOURCE_TYPES))}"
            )

        if not source_name or not str(source_name).strip():
            raise ProvenanceError("source_name is required and cannot be empty.")

        if not collection_method or not str(collection_method).strip():
            raise ProvenanceError("collection_method is required and cannot be empty.")

        date_val = collection_date or datetime.now(timezone.utc).isoformat()

        # Build provenance object
        return ProvenanceMetadata(
            source_type=ProvenanceSourceType(st_val),
            source_name=str(source_name).strip(),
            source_id=str(source_id).strip() if source_id else None,
            collection_method=str(collection_method).strip(),
            license=str(license_str).strip() if license_str else "UNSPECIFIED",
            collection_date=date_val,
        )

    @classmethod
    def validate_provenance(cls, provenance: Optional[ProvenanceMetadata]) -> None:
        """
        Verifies that provenance is present and valid.
        """
        if provenance is None:
            raise ProvenanceError("Record is missing required provenance metadata.")

        if not isinstance(provenance, ProvenanceMetadata):
            raise ProvenanceError(f"Expected ProvenanceMetadata instance, got {type(provenance)}")

        if provenance.source_type.value not in cls.ALLOWED_SOURCE_TYPES:
            raise ProvenanceError(f"Unsupported provenance source_type: {provenance.source_type}")

        if not provenance.source_name:
            raise ProvenanceError("Provenance source_name cannot be empty.")
