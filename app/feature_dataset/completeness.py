"""
TrustLens Feature Completeness and Missing Data Policy (Phase 6D).
Tracks, reports, and validates missingness rates across all feature columns.
Enforces the explicit missing data policy: unknown/unavailable data is never coerced to "safe".
"""

import math
from typing import Any, Dict, List, Optional, Set
import pandas as pd
from pydantic import BaseModel, Field

from app.feature_dataset.schemas import FeatureDefinition, FeatureRecord, FeatureRegistry


class FeatureMissingnessStats(BaseModel):
    """Missingness statistics for a single feature column."""
    feature_name: str
    feature_group: str
    missing_count: int
    total_count: int
    missing_rate: float
    missing_policy: str


class RowMissingnessStats(BaseModel):
    """Missingness statistics for a single instance row."""
    feature_id: str
    post_id: str
    split: str
    missing_feature_count: int
    total_features: int
    missing_rate: float
    is_excessive: bool


class CompletenessReport(BaseModel):
    """Comprehensive completeness audit report across all records."""
    total_rows: int
    total_features: int
    total_cells: int
    total_missing_cells: int
    overall_missing_rate: float
    features_with_missingness: List[FeatureMissingnessStats]
    excessive_missing_rows: List[RowMissingnessStats]
    summary_by_group: Dict[str, Dict[str, float]] = Field(default_factory=dict)


class FeatureCompletenessAnalyzer:
    """
    Analyzes missing data according to TrustLens Phase 6D missingness policy.
    Identifies true missing values (NaN/None/null) as well as explicit sentinel
    encodings representing unavailable observations (e.g., -1.0 for missing OCR confidence).
    """

    # Features where negative values explicitly represent legitimately unavailable observations
    SENTINEL_NUMERIC_FEATURES: Set[str] = {
        "domain_age_days",
        "top_retrieval_score",
        "mean_retrieval_score",
        "max_retrieval_score",
        "ocr_confidence",
    }

    # Features where UNKNOWN represents missing category
    SENTINEL_CATEGORICAL_VALUES: Set[str] = {"UNKNOWN", "unknown", "NONE", "none"}

    def __init__(self, excessive_threshold: float = 0.50):
        self.excessive_threshold = excessive_threshold
        self.feature_map = FeatureRegistry.get_feature_map()

    def is_missing_value(self, feature_name: str, val: Any) -> bool:
        """Determines if a feature value is missing or sentinel-unavailable."""
        if val is None:
            return True
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return True
        if feature_name in self.SENTINEL_NUMERIC_FEATURES and isinstance(val, (int, float)):
            if val < 0:
                return True
        if isinstance(val, str) and val in self.SENTINEL_CATEGORICAL_VALUES:
            return True
        return False

    def analyze_records(self, records: List[FeatureRecord]) -> CompletenessReport:
        """Analyzes a list of canonical FeatureRecord objects."""
        if not records:
            return CompletenessReport(
                total_rows=0,
                total_features=len(self.feature_map),
                total_cells=0,
                total_missing_cells=0,
                overall_missing_rate=0.0,
                features_with_missingness=[],
                excessive_missing_rows=[],
            )

        feature_names = FeatureRegistry.get_feature_names()
        total_rows = len(records)
        total_features = len(feature_names)
        total_cells = total_rows * total_features

        # Track per-feature missing counts
        feat_missing_counts = {name: 0 for name in feature_names}
        excessive_rows: List[RowMissingnessStats] = []
        total_missing_cells = 0

        for rec in records:
            row_missing = 0
            for name in feature_names:
                val = rec.features.get(name)
                if self.is_missing_value(name, val):
                    feat_missing_counts[name] += 1
                    row_missing += 1
                    total_missing_cells += 1

            row_rate = row_missing / total_features if total_features > 0 else 0.0
            is_excessive = row_rate > self.excessive_threshold
            if is_excessive:
                excessive_rows.append(
                    RowMissingnessStats(
                        feature_id=rec.feature_id,
                        post_id=rec.post_id,
                        split=str(rec.split.value if hasattr(rec.split, "value") else rec.split),
                        missing_feature_count=row_missing,
                        total_features=total_features,
                        missing_rate=round(row_rate, 4),
                        is_excessive=is_excessive,
                    )
                )

        # Compile feature stats
        feature_stats: List[FeatureMissingnessStats] = []
        group_missing: Dict[str, Dict[str, int]] = {}

        for name in feature_names:
            cnt = feat_missing_counts[name]
            rate = cnt / total_rows if total_rows > 0 else 0.0
            defn = self.feature_map.get(name)
            group_name = defn.group.value if defn else "OTHER"
            policy = defn.missing_policy if defn else "unspecified"

            feature_stats.append(
                FeatureMissingnessStats(
                    feature_name=name,
                    feature_group=group_name,
                    missing_count=cnt,
                    total_count=total_rows,
                    missing_rate=round(rate, 4),
                    missing_policy=policy,
                )
            )

            if group_name not in group_missing:
                group_missing[group_name] = {"missing": 0, "total": 0}
            group_missing[group_name]["missing"] += cnt
            group_missing[group_name]["total"] += total_rows

        group_summary = {}
        for grp, counts in group_missing.items():
            tot = counts["total"]
            grp_rate = counts["missing"] / tot if tot > 0 else 0.0
            group_summary[grp] = {
                "missing_cells": counts["missing"],
                "total_cells": tot,
                "missing_rate": round(grp_rate, 4),
            }

        overall_rate = total_missing_cells / total_cells if total_cells > 0 else 0.0

        return CompletenessReport(
            total_rows=total_rows,
            total_features=total_features,
            total_cells=total_cells,
            total_missing_cells=total_missing_cells,
            overall_missing_rate=round(overall_rate, 4),
            features_with_missingness=feature_stats,
            excessive_missing_rows=excessive_rows,
            summary_by_group=group_summary,
        )

    def analyze_dataframe(self, df: pd.DataFrame) -> CompletenessReport:
        """Analyzes a pandas DataFrame containing feature columns."""
        feature_names = [f for f in FeatureRegistry.get_feature_names() if f in df.columns]
        total_rows = len(df)
        total_features = len(feature_names)
        total_cells = total_rows * total_features

        feat_missing_counts = {}
        total_missing_cells = 0

        for name in feature_names:
            series = df[name]
            if name in self.SENTINEL_NUMERIC_FEATURES:
                missing_mask = series.isna() | (series < 0)
            elif series.dtype == "object":
                missing_mask = series.isna() | series.isin(self.SENTINEL_CATEGORICAL_VALUES)
            else:
                missing_mask = series.isna()

            cnt = int(missing_mask.sum())
            feat_missing_counts[name] = cnt
            total_missing_cells += cnt

        feature_stats = []
        for name in feature_names:
            cnt = feat_missing_counts[name]
            rate = cnt / total_rows if total_rows > 0 else 0.0
            defn = self.feature_map.get(name)
            group_name = defn.group.value if defn else "OTHER"
            policy = defn.missing_policy if defn else "unspecified"
            feature_stats.append(
                FeatureMissingnessStats(
                    feature_name=name,
                    feature_group=group_name,
                    missing_count=cnt,
                    total_count=total_rows,
                    missing_rate=round(rate, 4),
                    missing_policy=policy,
                )
            )

        overall_rate = total_missing_cells / total_cells if total_cells > 0 else 0.0

        return CompletenessReport(
            total_rows=total_rows,
            total_features=total_features,
            total_cells=total_cells,
            total_missing_cells=total_missing_cells,
            overall_missing_rate=round(overall_rate, 4),
            features_with_missingness=feature_stats,
            excessive_missing_rows=[],
            summary_by_group={},
        )
