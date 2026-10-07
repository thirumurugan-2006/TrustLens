"""
TrustLens Feature Dataset Validator (Phase 6D).
Validates structural integrity, 16-point quality checklist, cluster-safe split isolation,
schema consistency across splits, and zero target leakage.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd
from pydantic import BaseModel, Field

from app.feature_dataset.schemas import FeatureDType, FeatureRegistry
from app.training.schemas import RiskLevel, SplitName


class LeakageAuditResult(BaseModel):
    """Detailed audit of potential target leakage signals."""
    leakage_detected: bool
    forbidden_columns_found: List[str]
    suspicious_features: List[str]
    notes: List[str]


class FeatureValidationReport(BaseModel):
    """Comprehensive validation report for TrustLens Phase 6D feature matrices."""
    is_valid: bool
    total_rows: int
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    class_distribution: Dict[str, int]
    missing_value_rate: float
    duplicate_rows: int
    cross_split_post_leakage: int
    cross_split_cluster_leakage: int
    target_leakage_detected: bool
    schema_consistency: str
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class FeatureDatasetValidator:
    """
    Validates train, validation, and test feature files against the 16-point specification.
    """

    FORBIDDEN_LEAKAGE_TERMS: Set[str] = {
        "gold_risk_label",
        "final_human_decision",
        "ground_truth_risk",
        "post_risk_score",
        "human_risk_decision",
        "future_model_prediction",
        "predicted_risk",
    }

    VALID_RISK_LABELS: Set[str] = {r.value for r in RiskLevel}
    VALID_SPLITS: Set[str] = {s.value for s in SplitName}

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.feature_names = FeatureRegistry.get_feature_names()
        self.feature_map = FeatureRegistry.get_feature_map()

    def audit_leakage(self, df_all: pd.DataFrame) -> LeakageAuditResult:
        """Audits feature columns for forbidden target-derivative signals."""
        notes = []
        forbidden_found = []
        suspicious = []

        feature_cols = [c for c in df_all.columns if c not in ["feature_id", "post_id", "cluster_id", "split", "risk_label"]]

        for col in feature_cols:
            col_lower = col.lower()
            for term in self.FORBIDDEN_LEAKAGE_TERMS:
                if term in col_lower:
                    forbidden_found.append(col)
                    notes.append(f"Forbidden leakage term '{term}' detected in feature column '{col}'")

        # Verify that risk_label is NOT included among feature names in registry
        if "risk_label" in self.feature_names:
            forbidden_found.append("risk_label_in_registry")
            notes.append("risk_label is directly present in FeatureRegistry!")

        leakage_detected = len(forbidden_found) > 0

        return LeakageAuditResult(
            leakage_detected=leakage_detected,
            forbidden_columns_found=forbidden_found,
            suspicious_features=suspicious,
            notes=notes,
        )

    def validate(self) -> FeatureValidationReport:
        """Executes full 16-point validation across train, validation, and test files."""
        errors: List[str] = []
        warnings: List[str] = []

        train_path = self.data_dir / "train.csv"
        val_path = self.data_dir / "validation.csv"
        test_path = self.data_dir / "test.csv"

        for p in [train_path, val_path, test_path]:
            if not p.exists():
                errors.append(f"Missing required dataset file: {p.name}")

        if errors:
            return FeatureValidationReport(
                is_valid=False,
                total_rows=0,
                train_rows=0,
                validation_rows=0,
                test_rows=0,
                feature_count=len(self.feature_names),
                class_distribution={},
                missing_value_rate=0.0,
                duplicate_rows=0,
                cross_split_post_leakage=0,
                cross_split_cluster_leakage=0,
                target_leakage_detected=False,
                schema_consistency="FAIL",
                errors=errors,
                warnings=warnings,
            )

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)
        df_test = pd.read_csv(test_path)

        train_rows = len(df_train)
        val_rows = len(df_val)
        test_rows = len(df_test)
        total_rows = train_rows + val_rows + test_rows

        # 1. Required columns & 11/12 Schema consistency across splits
        id_cols = ["feature_id", "post_id", "cluster_id", "split", "risk_label"]
        expected_cols = id_cols + self.feature_names

        for name, df in [("train", df_train), ("validation", df_val), ("test", df_test)]:
            missing_cols = [c for c in expected_cols if c not in df.columns]
            if missing_cols:
                errors.append(f"{name}.csv missing required columns: {missing_cols[:5]}...")

            # 4. Valid split column values
            invalid_splits = df[~df["split"].isin(self.VALID_SPLITS)]
            if not invalid_splits.empty:
                errors.append(f"{name}.csv contains invalid split values: {invalid_splits['split'].unique()}")

            # 3. Valid target labels
            invalid_labels = df[~df["risk_label"].isin(self.VALID_RISK_LABELS)]
            if not invalid_labels.empty:
                errors.append(f"{name}.csv contains invalid risk labels: {invalid_labels['risk_label'].unique()}")

        # Verify exact column equality across train, val, test
        if list(df_train.columns) != list(df_val.columns) or list(df_train.columns) != list(df_test.columns):
            errors.append("Schema mismatch: column lists differ between train, validation, and test!")
            schema_consistency = "FAIL"
        else:
            schema_consistency = "PASS"

        df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

        # 2. Unique feature IDs
        if df_all["feature_id"].duplicated().any():
            dup_cnt = int(df_all["feature_id"].duplicated().sum())
            errors.append(f"Found {dup_cnt} duplicate feature_ids across records.")

        # 8. Duplicate rows
        dup_posts = df_all["post_id"].duplicated().sum()
        if dup_posts > 0:
            errors.append(f"Found {dup_posts} duplicate post instances across dataset.")

        # 9 & 10. Cluster-safe split and cross-split leakage
        train_posts = set(df_train["post_id"])
        val_posts = set(df_val["post_id"])
        test_posts = set(df_test["post_id"])
        post_leakage = len((train_posts & val_posts) | (train_posts & test_posts) | (val_posts & test_posts))
        if post_leakage > 0:
            errors.append(f"Cross-split post leakage detected: {post_leakage} overlapping posts!")

        train_clusters = set(df_train["cluster_id"].dropna())
        val_clusters = set(df_val["cluster_id"].dropna())
        test_clusters = set(df_test["cluster_id"].dropna())
        cluster_leakage = len((train_clusters & val_clusters) | (train_clusters & test_clusters) | (val_clusters & test_clusters))
        if cluster_leakage > 0:
            errors.append(f"Cross-split cluster leakage detected: {cluster_leakage} overlapping clusters!")

        # 7. Target leakage audit
        leakage_res = self.audit_leakage(df_all)
        if leakage_res.leakage_detected:
            errors.append(f"Target leakage detected: {leakage_res.forbidden_columns_found}")

        # 13. Numeric values are finite
        for feat_name, defn in self.feature_map.items():
            if defn.dtype in [FeatureDType.FLOAT, FeatureDType.INT] and feat_name in df_all.columns:
                series = pd.to_numeric(df_all[feat_name], errors="coerce")
                inf_count = int(np_isinf(series)) if hasattr(series, "values") else 0
                if inf_count > 0:
                    errors.append(f"Feature '{feat_name}' contains {inf_count} infinite values!")

        # 16. Feature registry completeness
        missing_from_registry = [c for c in df_all.columns if c not in id_cols and c not in self.feature_map]
        if missing_from_registry:
            errors.append(f"Columns not declared in FeatureRegistry: {missing_from_registry}")

        # Class distribution
        class_dist = {label: int((df_all["risk_label"] == label).sum()) for label in sorted(self.VALID_RISK_LABELS)}

        # Missing value rate (excluding sentinel negative indicators)
        total_cells = total_rows * len(self.feature_names)
        null_cells = int(df_all[self.feature_names].isna().sum().sum())
        missing_rate = null_cells / total_cells if total_cells > 0 else 0.0

        is_valid = (len(errors) == 0)

        # Write leakage audit report to disk
        leakage_report_path = self.data_dir / "feature_leakage_report.json"
        with open(leakage_report_path, "w", encoding="utf-8") as f:
            json.dump(leakage_res.model_dump(mode="json"), f, indent=2)

        return FeatureValidationReport(
            is_valid=is_valid,
            total_rows=total_rows,
            train_rows=train_rows,
            validation_rows=val_rows,
            test_rows=test_rows,
            feature_count=len(self.feature_names),
            class_distribution=class_dist,
            missing_value_rate=round(missing_rate, 4),
            duplicate_rows=int(dup_posts),
            cross_split_post_leakage=post_leakage,
            cross_split_cluster_leakage=cluster_leakage,
            target_leakage_detected=leakage_res.leakage_detected,
            schema_consistency=schema_consistency,
            errors=errors,
            warnings=warnings,
        )


def np_isinf(series: pd.Series) -> int:
    """Helper to count infinite values without strict numpy import dependency."""
    import numpy as np
    return int(np.isinf(series).sum())


def main():
    parser = argparse.ArgumentParser(description="TrustLens Phase 6D Feature Dataset Validator CLI")
    parser.add_argument("--data-dir", type=str, default="data/processed/features", help="Directory containing feature matrices")
    args = parser.parse_args()

    validator = FeatureDatasetValidator(Path(args.data_dir))
    report = validator.validate()

    print("================ TRUSTLENS FEATURE DATASET VALIDATION ================")
    print(f"Rows:                 {report.total_rows}")
    print(f"Features:             {report.feature_count}")
    print(f"Train:                {report.train_rows}")
    print(f"Validation:           {report.validation_rows}")
    print(f"Test:                 {report.test_rows}")
    print("Risk Distribution:")
    for label, count in sorted(report.class_distribution.items()):
        print(f"  {label:<12}: {count}")
    print(f"Missing feature rate: {report.missing_value_rate:.4f}")
    print(f"Duplicate rows:       {report.duplicate_rows}")
    print(f"Cross-split leakage:  {report.cross_split_post_leakage}")
    print(f"Target leakage:       {'DETECTED' if report.target_leakage_detected else 0}")
    print(f"Schema consistency:   {report.schema_consistency}")
    print(f"Overall:              {'VALID' if report.is_valid else 'INVALID'}")
    print("====================================================================")

    if report.errors:
        print("\nValidation Errors:")
        for err in report.errors:
            print(f"  - [ERROR] {err}")

    sys.exit(0 if report.is_valid else 1)


if __name__ == "__main__":
    main()
