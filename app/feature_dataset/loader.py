"""
TrustLens LightGBM Dataset Loader (Phase 6D).
Provides isolated, deterministic loaders for train, validation, and test matrices.
Enforces feature schema ordering, type casting for categorical features,
and strict exclusion of target labels and metadata from the feature matrix X.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd

from app.feature_dataset.schemas import FeatureDType, FeatureRegistry


class FeatureDatasetLoader:
    """
    Loads train, validation, and test datasets in LightGBM-compatible format.
    Guarantees:
    1. Zero target leakage: target risk column is strictly separated into y.
    2. Strict feature ordering according to FeatureRegistry.
    3. Native pandas categorical dtypes for LightGBM compatibility.
    """

    LABEL_MAPPING: Dict[str, int] = {
        "INSUFFICIENT": 0,
        "LOW": 1,
        "MEDIUM": 2,
        "HIGH": 3,
    }

    DERIVED_BINARY_MAPPING: Dict[str, int] = {
        "INSUFFICIENT": 0,
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 1,
    }

    def __init__(self, data_dir: Union[str, Path] = "data/processed/features"):
        self.data_dir = Path(data_dir)
        self.feature_names = FeatureRegistry.get_feature_names()
        self.feature_map = FeatureRegistry.get_feature_map()

    def _prepare_matrix(
        self,
        df: pd.DataFrame,
        encode_labels: bool = True,
        binary_target: bool = False,
        include_metadata: bool = False,
    ) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.DataFrame]]:
        """
        Extracts ordered feature matrix X and target vector y from tabular DataFrame.
        """
        # 1. Verify all required features exist in df
        missing_features = [f for f in self.feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(f"DataFrame is missing required feature columns: {missing_features}")

        # 2. Extract feature matrix X in strict registry order
        X = df[self.feature_names].copy()

        # 3. Cast categorical columns to pandas 'category' dtype for LightGBM
        for name in self.feature_names:
            defn = self.feature_map.get(name)
            if defn and defn.dtype == FeatureDType.CATEGORICAL:
                X[name] = X[name].astype("category")
            elif defn and defn.dtype == FeatureDType.BOOLEAN:
                X[name] = X[name].astype("bool")

        # 4. Extract target vector y
        if "risk_label" not in df.columns:
            raise ValueError("Target column 'risk_label' missing from dataset.")

        y_raw = df["risk_label"]

        if binary_target:
            y = y_raw.map(self.DERIVED_BINARY_MAPPING).astype(int)
        elif encode_labels:
            y = y_raw.map(self.LABEL_MAPPING).astype(int)
        else:
            y = y_raw.copy()

        # 5. Extract metadata if requested
        metadata = None
        if include_metadata:
            meta_cols = [c for c in ["feature_id", "post_id", "cluster_id", "split"] if c in df.columns]
            metadata = df[meta_cols].copy()

        return X, y, metadata

    def load_train(
        self,
        encode_labels: bool = True,
        binary_target: bool = False,
        include_metadata: bool = False,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
        """Loads only the training matrix."""
        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Training matrix not found at: {train_path}")

        df = pd.read_csv(train_path)
        X, y, meta = self._prepare_matrix(df, encode_labels, binary_target, include_metadata)
        if include_metadata:
            return X, y, meta
        return X, y

    def load_validation(
        self,
        encode_labels: bool = True,
        binary_target: bool = False,
        include_metadata: bool = False,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
        """Loads only the validation matrix."""
        val_path = self.data_dir / "validation.csv"
        if not val_path.exists():
            raise FileNotFoundError(f"Validation matrix not found at: {val_path}")

        df = pd.read_csv(val_path)
        X, y, meta = self._prepare_matrix(df, encode_labels, binary_target, include_metadata)
        if include_metadata:
            return X, y, meta
        return X, y

    def load_test(
        self,
        encode_labels: bool = True,
        binary_target: bool = False,
        include_metadata: bool = False,
    ) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
        """Loads only the test matrix."""
        test_path = self.data_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Test matrix not found at: {test_path}")

        df = pd.read_csv(test_path)
        X, y, meta = self._prepare_matrix(df, encode_labels, binary_target, include_metadata)
        if include_metadata:
            return X, y, meta
        return X, y


# Module-level convenience functions
def load_train(data_dir: Union[str, Path] = "data/processed/features", **kwargs):
    return FeatureDatasetLoader(data_dir).load_train(**kwargs)


def load_validation(data_dir: Union[str, Path] = "data/processed/features", **kwargs):
    return FeatureDatasetLoader(data_dir).load_validation(**kwargs)


def load_test(data_dir: Union[str, Path] = "data/processed/features", **kwargs):
    return FeatureDatasetLoader(data_dir).load_test(**kwargs)
