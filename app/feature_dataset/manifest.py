"""
TrustLens Feature Manifest Builder (Phase 6D).
Computes descriptive distributions, group breakdown, feature correlations,
constant/near-constant detection, and class distributions for feature_manifest.json.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.feature_dataset.schemas import FeatureDType, FeatureGroup, FeatureRegistry
from app.training.schemas import RiskLevel


class FeatureManifest(BaseModel):
    """Formal summary manifest for Phase 6D feature matrices."""
    manifest_version: str = "1.0.0"
    created_at: str
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    numerical_feature_count: int
    categorical_feature_count: int
    boolean_feature_count: int
    feature_group_counts: Dict[str, int]
    target_class_distribution: Dict[str, int]
    target_class_proportions: Dict[str, float]
    missing_value_count: int
    missing_value_rate: float
    constant_features: List[str]
    near_constant_features: List[str]
    high_correlation_pairs: List[Dict[str, Any]]
    numerical_feature_distributions: Dict[str, Dict[str, float]]
    categorical_feature_distributions: Dict[str, Dict[str, int]]
    boolean_feature_distributions: Dict[str, Dict[str, int]]


class FeatureManifestBuilder:
    """
    Builds the authoritative Phase 6D manifest and statistical distribution audit.
    """

    @classmethod
    def build_manifest(
        cls,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> FeatureManifest:
        """
        Analyzes combined dataset matrices and builds the complete manifest.
        """
        from datetime import datetime

        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
        total_rows = len(df_all)
        train_rows = len(df_train)
        val_rows = len(df_val)
        test_rows = len(df_test)

        feature_names = FeatureRegistry.get_feature_names()
        feature_map = FeatureRegistry.get_feature_map()

        num_features = [f for f in feature_names if feature_map[f].dtype in [FeatureDType.FLOAT, FeatureDType.INT]]
        cat_features = [f for f in feature_names if feature_map[f].dtype == FeatureDType.CATEGORICAL]
        bool_features = [f for f in feature_names if feature_map[f].dtype == FeatureDType.BOOLEAN]

        # Group counts
        group_counts: Dict[str, int] = {}
        for defn in FeatureRegistry.DEFINITIONS:
            grp = defn.group.value
            group_counts[grp] = group_counts.get(grp, 0) + 1

        # Target distribution
        class_dist: Dict[str, int] = {}
        class_prop: Dict[str, float] = {}
        for r in RiskLevel:
            cnt = int((df_all["risk_label"] == r.value).sum())
            class_dist[r.value] = cnt
            class_prop[r.value] = round(cnt / total_rows, 4) if total_rows > 0 else 0.0

        # Missingness
        total_cells = total_rows * len(feature_names)
        null_cells = int(df_all[feature_names].isna().sum().sum())
        missing_rate = round(null_cells / total_cells, 4) if total_cells > 0 else 0.0

        # Numerical distributions
        num_dist: Dict[str, Dict[str, float]] = {}
        constant_features: List[str] = []
        near_constant_features: List[str] = []

        for f in num_features:
            series = pd.to_numeric(df_all[f], errors="coerce").dropna()
            if series.empty:
                continue
            f_min = float(series.min())
            f_max = float(series.max())
            f_mean = float(series.mean())
            f_median = float(series.median())
            f_std = float(series.std(ddof=0))

            num_dist[f] = {
                "min": round(f_min, 4),
                "max": round(f_max, 4),
                "mean": round(f_mean, 4),
                "median": round(f_median, 4),
                "std": round(f_std, 4),
            }

            if f_min == f_max:
                constant_features.append(f)
            elif f_std < 1e-4:
                near_constant_features.append(f)

        # Categorical distributions
        cat_dist: Dict[str, Dict[str, int]] = {}
        for f in cat_features:
            val_counts = df_all[f].value_counts().to_dict()
            cat_dist[f] = {str(k): int(v) for k, v in val_counts.items()}
            # Check constant
            if len(val_counts) <= 1:
                constant_features.append(f)

        # Boolean distributions
        bool_dist: Dict[str, Dict[str, int]] = {}
        for f in bool_features:
            val_counts = df_all[f].value_counts().to_dict()
            bool_dist[f] = {
                "true": int(val_counts.get(True, 0)),
                "false": int(val_counts.get(False, 0)),
            }
            # Check constant
            if bool_dist[f]["true"] == 0 or bool_dist[f]["false"] == 0:
                constant_features.append(f)

        # Pairwise correlations for numerical features
        high_corr_pairs: List[Dict[str, Any]] = []
        if len(num_features) > 1:
            corr_df = df_train[num_features].corr().abs()
            cols = corr_df.columns
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    c1 = cols[i]
                    c2 = cols[j]
                    val = corr_df.iloc[i, j]
                    if not np.isnan(val) and val > 0.95:
                        high_corr_pairs.append({
                            "feature_1": c1,
                            "feature_2": c2,
                            "correlation": round(float(val), 4),
                        })

        manifest = FeatureManifest(
            created_at=datetime.utcnow().isoformat() + "Z",
            total_rows=total_rows,
            train_rows=train_rows,
            validation_rows=val_rows,
            test_rows=test_rows,
            feature_count=len(feature_names),
            numerical_feature_count=len(num_features),
            categorical_feature_count=len(cat_features),
            boolean_feature_count=len(bool_features),
            feature_group_counts=group_counts,
            target_class_distribution=class_dist,
            target_class_proportions=class_prop,
            missing_value_count=null_cells,
            missing_value_rate=missing_rate,
            constant_features=sorted(list(set(constant_features))),
            near_constant_features=sorted(list(set(near_constant_features))),
            high_correlation_pairs=high_corr_pairs,
            numerical_feature_distributions=num_dist,
            categorical_feature_distributions=cat_dist,
            boolean_feature_distributions=bool_dist,
        )

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(manifest.model_dump(mode="json"), f, indent=2)

        return manifest
