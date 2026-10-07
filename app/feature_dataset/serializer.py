"""
TrustLens Feature Matrix Serializer (Phase 6D).
Serializes canonical FeatureRecord collections to CSV (tabular inspection)
and JSONL (rich provenance and metadata), and exports the formal feature_schema.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from app.feature_dataset.schemas import FeatureDefinition, FeatureRecord, FeatureRegistry


class FeatureSerializer:
    """
    Serializes structured feature matrices across CSV, JSONL, and schema JSON formats.
    Guarantees deterministic column ordering and complete schema registry export.
    """

    ID_COLUMNS: List[str] = [
        "feature_id",
        "post_id",
        "cluster_id",
        "split",
        "risk_label",
    ]

    @classmethod
    def records_to_dataframe(cls, records: List[FeatureRecord]) -> pd.DataFrame:
        """
        Converts canonical FeatureRecord objects into a pandas DataFrame.
        Columns: ID/target columns followed by strictly ordered features.
        """
        feature_names = FeatureRegistry.get_feature_names()
        rows: List[Dict[str, Any]] = []

        for rec in records:
            row: Dict[str, Any] = {
                "feature_id": rec.feature_id,
                "post_id": rec.post_id,
                "cluster_id": rec.cluster_id,
                "split": rec.split.value if hasattr(rec.split, "value") else str(rec.split),
                "risk_label": rec.risk_label.value if hasattr(rec.risk_label, "value") else str(rec.risk_label),
            }
            # Append features in registry order
            for name in feature_names:
                row[name] = rec.features.get(name)
            rows.append(row)

        df = pd.DataFrame(rows)
        # Ensure ordered columns
        expected_cols = cls.ID_COLUMNS + feature_names
        # Keep only existing columns if any were missed
        ordered_cols = [c for c in expected_cols if c in df.columns]
        return df[ordered_cols]

    @classmethod
    def save_csv(cls, records: List[FeatureRecord], output_path: Path) -> Path:
        """Saves records to CSV with exact column ordering."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = cls.records_to_dataframe(records)
        df.to_csv(output_path, index=False, encoding="utf-8")
        return output_path

    @classmethod
    def save_jsonl(cls, records: List[FeatureRecord], output_path: Path) -> Path:
        """Saves canonical records to JSONL preserving rich metadata and provenance."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                line = json.dumps(rec.model_dump(mode="json"), ensure_ascii=False)
                f.write(line + "\n")
        return output_path

    @classmethod
    def export_feature_schema(cls, output_path: Path) -> Path:
        """
        Exports the master feature schema registry as a structured JSON catalog.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        definitions: List[Dict[str, Any]] = [
            d.model_dump(mode="json") for d in FeatureRegistry.DEFINITIONS
        ]

        schema_payload = {
            "schema_version": "1.0.0",
            "total_features": len(definitions),
            "feature_groups": list({d.group.value for d in FeatureRegistry.DEFINITIONS}),
            "features": definitions,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(schema_payload, f, indent=2, ensure_ascii=False)

        return output_path
