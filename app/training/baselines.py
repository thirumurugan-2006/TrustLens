from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


class MajorityClassBaseline:
    """
    Trivial baseline predicting the most frequent class observed in training data.
    Provides lower bound performance benchmark.
    """

    def __init__(self):
        self.majority_label: Optional[Any] = None
        self.classes_: List[Any] = []
        self.class_priors_: Dict[Any, float] = {}

    def fit(self, y_train: List[Any]) -> "MajorityClassBaseline":
        if not y_train:
            raise ValueError("Training labels cannot be empty.")
        counts = Counter(y_train)
        self.majority_label = counts.most_common(1)[0][0]
        self.classes_ = sorted(list(counts.keys()))
        total = len(y_train)
        self.class_priors_ = {cls: counts[cls] / total for cls in self.classes_}
        return self

    def predict(self, X: List[Any]) -> List[Any]:
        if self.majority_label is None:
            raise RuntimeError("Model has not been fitted.")
        return [self.majority_label] * len(X)

    def predict_proba(self, X: List[Any]) -> np.ndarray:
        if not self.classes_:
            raise RuntimeError("Model has not been fitted.")
        prior_row = [self.class_priors_[c] for c in self.classes_]
        return np.tile(prior_row, (len(X), 1))


class TfidfLogisticRegressionBaseline:
    """
    Standard linear n-gram baseline using TF-IDF feature extraction and Logistic Regression.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)),
            ("clf", LogisticRegression(C=C, max_iter=1000, random_state=random_state, class_weight="balanced")),
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], y_train: List[Any]) -> "TfidfLogisticRegressionBaseline":
        if not texts or not y_train:
            raise ValueError("Texts and labels cannot be empty.")
        self.pipeline.fit(texts, y_train)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[Any]:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return list(self.pipeline.predict(texts))

    def predict_proba(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return self.pipeline.predict_proba(texts)


class TfidfLinearSVMBaseline:
    """
    Max-margin linear baseline using TF-IDF feature extraction and Linear Support Vector Classifier.
    """

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        C: float = 1.0,
        random_state: int = 42,
    ):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, sublinear_tf=True)),
            ("clf", LinearSVC(C=C, max_iter=2000, random_state=random_state, class_weight="balanced")),
        ])
        self.is_fitted = False

    def fit(self, texts: List[str], y_train: List[Any]) -> "TfidfLinearSVMBaseline":
        if not texts or not y_train:
            raise ValueError("Texts and labels cannot be empty.")
        self.pipeline.fit(texts, y_train)
        self.is_fitted = True
        return self

    def predict(self, texts: List[str]) -> List[Any]:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return list(self.pipeline.predict(texts))

    def decision_function(self, texts: List[str]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model has not been fitted.")
        return self.pipeline.decision_function(texts)


class BaselineFramework:
    """
    Factory and runner for standard reproducibility baselines.
    """

    @staticmethod
    def get_baseline(baseline_type: str, **kwargs) -> Any:
        b_type = baseline_type.lower().strip()
        if b_type in ("majority", "majority_class"):
            return MajorityClassBaseline()
        elif b_type in ("tfidf_lr", "logistic_regression", "tfidf_logistic"):
            return TfidfLogisticRegressionBaseline(**kwargs)
        elif b_type in ("tfidf_svm", "linear_svm", "svm"):
            return TfidfLinearSVMBaseline(**kwargs)
        else:
            raise ValueError(f"Unknown baseline type: {baseline_type}. Available: 'majority', 'tfidf_lr', 'tfidf_svm'")
