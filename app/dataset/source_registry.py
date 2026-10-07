import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from app.training.schemas import ProvenanceSourceType


class SourceEntry(BaseModel):
    """
    Source Registry record tracking permission, license, and provenance of data sources.
    """
    source_id: str
    source_name: str
    source_type: ProvenanceSourceType
    license: str
    permission_status: str = "APPROVED"  # APPROVED, PENDING_REVIEW, REJECTED, LICENSE_REVIEW_REQUIRED
    collection_method: str = "curated_archive"
    language: List[str] = Field(default_factory=lambda: ["en"])
    category: List[str] = Field(default_factory=lambda: ["general"])
    collection_date: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    notes: Optional[str] = None


class SourceRegistry:
    """
    Manages permitted data source registrations, enforcing license review
    and preventing unauthorized/unreviewed data from reaching training splits.
    """

    def __init__(self, registry_path: Union[str, Path] = "data/trustlens/source_registry.json"):
        self.registry_path = Path(registry_path)
        self.sources: Dict[str, SourceEntry] = {}
        if self.registry_path.is_file():
            self.load()

    def load(self) -> None:
        """Loads source records from registry file."""
        if not self.registry_path.is_file():
            return
        content = self.registry_path.read_text(encoding="utf-8")
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                entry = SourceEntry.model_validate(item)
                self.sources[entry.source_id] = entry
        elif isinstance(data, dict):
            # Keyed by source_id or sources array
            if "sources" in data and isinstance(data["sources"], list):
                for item in data["sources"]:
                    entry = SourceEntry.model_validate(item)
                    self.sources[entry.source_id] = entry
            else:
                for k, v in data.items():
                    entry = SourceEntry.model_validate(v)
                    self.sources[k] = entry

    def save(self) -> Path:
        """Persists source records to registry file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        serialized = [entry.model_dump() for entry in self.sources.values()]
        self.registry_path.write_text(json.dumps(serialized, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.registry_path

    def register_source(self, entry: SourceEntry) -> None:
        """Registers or updates a source entry."""
        self.sources[entry.source_id] = entry

    def get_source(self, source_id: str) -> Optional[SourceEntry]:
        """Retrieves source entry by identifier."""
        return self.sources.get(source_id)

    def is_permitted(self, source_id_or_name: str) -> bool:
        """
        Returns True only if the source has permission_status == 'APPROVED'.
        Sources marked PENDING_REVIEW or LICENSE_REVIEW_REQUIRED return False.
        """
        for entry in self.sources.values():
            if entry.source_id == source_id_or_name or entry.source_name == source_id_or_name:
                return entry.permission_status == "APPROVED"
        return False

    def validate_source_license(self, source_name: str) -> bool:
        """Verifies if the given source name has an approved license."""
        for entry in self.sources.values():
            if entry.source_name == source_name:
                return entry.permission_status == "APPROVED" and entry.license not in ("UNSPECIFIED", "UNKNOWN", "PENDING_REVIEW")
        return False
