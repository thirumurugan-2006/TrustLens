import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


class EvaluationFramework:
    """
    Unified evaluation framework for TrustLens experiments.
    Provides standard evaluation functions for:
    - Classification (Accuracy, Precision, Recall, Macro-F1, Weighted-F1, Confusion Matrix)
    - Probability Calibration (Expected Calibration Error, Brier Score)
    - Retrieval & Ranking (Recall@K, Precision@K, MRR, nDCG@K)
    - Selective Classification & Abstention (Coverage, Selective Accuracy, Risk-Coverage Curve)
    - Multimodal Text / OCR (Character Error Rate, Word Error Rate)
    """

    # ---------------------------------------------------------------------------
    # 1. Classification Metrics
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_classification_metrics(
        y_true: List[Any],
        y_pred: List[Any],
        labels: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Computes comprehensive multiclass classification metrics."""
        if not y_true or not y_pred:
            return {"error": "Empty input predictions or ground truth."}
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: len(y_true)={len(y_true)}, len(y_pred)={len(y_pred)}")

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
        )

        unique_labels = labels if labels is not None else sorted(list(set(y_true) | set(y_pred)))

        acc = float(accuracy_score(y_true, y_pred))
        macro_prec = float(precision_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0))
        weighted_prec = float(precision_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0))
        weighted_rec = float(recall_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()

        # Detailed per-class report
        report_dict = classification_report(
            y_true, y_pred, labels=unique_labels, output_dict=True, zero_division=0
        )

        return {
            "accuracy": acc,
            "macro_precision": macro_prec,
            "weighted_precision": weighted_prec,
            "macro_recall": macro_rec,
            "weighted_recall": weighted_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "labels": unique_labels,
            "confusion_matrix": cm,
            "per_class_report": report_dict,
        }

    # ---------------------------------------------------------------------------
    # 2. Probability & Ranking AUC Metrics
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_auc_metrics(
        y_true: List[Any],
        y_score: List[Any],
        multi_class: str = "ovr",
    ) -> Dict[str, Any]:
        """Computes ROC-AUC and PR-AUC for binary or multiclass probability predictions."""
        if not y_true or not y_score:
            return {"error": "Empty probability predictions or labels."}

        from sklearn.metrics import average_precision_score, roc_auc_score

        y_true_arr = np.array(y_true)
        y_score_arr = np.array(y_score)

        results = {}
        unique_classes = np.unique(y_true_arr)

        if len(unique_classes) == 2:
            try:
                # If 2D probability matrix passed for binary, take second column
                if y_score_arr.ndim == 2 and y_score_arr.shape[1] == 2:
                    y_score_arr = y_score_arr[:, 1]
                roc = float(roc_auc_score(y_true_arr, y_score_arr))
                pr = float(average_precision_score(y_true_arr, y_score_arr))
                results["roc_auc"] = roc
                results["pr_auc"] = pr
            except Exception as e:
                results["error"] = str(e)
        elif len(unique_classes) > 2:
            try:
                roc = float(roc_auc_score(y_true_arr, y_score_arr, multi_class=multi_class))
                results["roc_auc"] = roc
            except Exception as e:
                results["roc_auc_error"] = str(e)
        else:
            results["error"] = "Only 1 unique class present in y_true; AUC cannot be computed."

        return results

    # ---------------------------------------------------------------------------
    # 3. Retrieval & Reranking Metrics
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_retrieval_metrics(
        retrieved_ids_per_query: Optional[List[List[str]]] = None,
        relevant_ids_per_query: Optional[List[Any]] = None,
        k_values: Optional[List[int]] = None,
        *,
        ranked_evidence_ids: Optional[List[List[str]]] = None,
        ground_truth_evidence_ids: Optional[List[Any]] = None,
    ) -> Dict[str, float]:
        """
        Computes Recall@K, Precision@K, Mean Reciprocal Rank (MRR), and nDCG@K.
        Supports K in [1, 5, 10, 20]. Accepts either retrieved_ids_per_query / relevant_ids_per_query
        or ranked_evidence_ids / ground_truth_evidence_ids.
        """
        retrieved_list = retrieved_ids_per_query if retrieved_ids_per_query is not None else ranked_evidence_ids
        relevant_list = relevant_ids_per_query if relevant_ids_per_query is not None else ground_truth_evidence_ids

        if not retrieved_list or not relevant_list:
            return {}

        # Normalize relevant items to sets
        normalized_relevant: List[Set[str]] = [
            set(r) if not isinstance(r, set) else r for r in relevant_list
        ]

        if k_values is None:
            k_values = [1, 5, 10, 20]

        num_queries = len(retrieved_list)
        recalls = {k: 0.0 for k in k_values}
        precisions = {k: 0.0 for k in k_values}
        ndcgs = {k: 0.0 for k in k_values}
        mrr_sum = 0.0

        for retrieved, relevant in zip(retrieved_list, normalized_relevant):
            if not relevant:
                continue

            # MRR
            first_rank = 0
            for rank, doc_id in enumerate(retrieved, start=1):
                if doc_id in relevant:
                    first_rank = rank
                    break
            if first_rank > 0:
                mrr_sum += 1.0 / first_rank

            # Metrics @ K
            for k in k_values:
                top_k = retrieved[:k]
                hits = sum(1 for doc_id in top_k if doc_id in relevant)
                recalls[k] += hits / len(relevant)
                precisions[k] += hits / k

                # nDCG@K
                dcg = 0.0
                for r_idx, doc_id in enumerate(top_k, start=1):
                    if doc_id in relevant:
                        dcg += 1.0 / math.log2(r_idx + 1)
                ideal_hits = min(len(relevant), k)
                idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
                if idcg > 0:
                    ndcgs[k] += dcg / idcg

        results: Dict[str, float] = {
            "mrr": round(mrr_sum / num_queries, 4),
        }
        for k in k_values:
            r_val = round(recalls[k] / num_queries, 4)
            p_val = round(precisions[k] / num_queries, 4)
            n_val = round(ndcgs[k] / num_queries, 4)
            results[f"recall@{k}"] = r_val
            results[f"recall_at_{k}"] = r_val
            results[f"precision@{k}"] = p_val
            results[f"precision_at_{k}"] = p_val
            results[f"ndcg@{k}"] = n_val
            results[f"ndcg_at_{k}"] = n_val

        return results

    # ---------------------------------------------------------------------------
    # 4. Calibration Metrics
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_calibration_metrics(
        y_true: List[int],
        y_prob: List[float],
        num_bins: int = 10,
    ) -> Dict[str, float]:
        """
        Computes Expected Calibration Error (ECE) and Brier Score for probability estimates.
        """
        if not y_true or not y_prob:
            return {}

        y_true_arr = np.array(y_true)
        y_prob_arr = np.array(y_prob)

        # Brier Score = Mean Squared Error of probability
        brier = float(np.mean((y_prob_arr - y_true_arr) ** 2))

        # ECE calculation
        bin_limits = np.linspace(0.0, 1.0, num_bins + 1)
        ece = 0.0
        n_samples = len(y_true)

        for i in range(num_bins):
            bin_lower = bin_limits[i]
            bin_upper = bin_limits[i + 1]

            in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr < bin_upper if i < num_bins - 1 else y_prob_arr <= bin_upper)
            bin_size = np.sum(in_bin)

            if bin_size > 0:
                bin_acc = float(np.mean(y_true_arr[in_bin]))
                bin_conf = float(np.mean(y_prob_arr[in_bin]))
                ece += (bin_size / n_samples) * abs(bin_acc - bin_conf)

        return {
            "ece": round(float(ece), 4),
            "brier_score": round(brier, 4),
        }

    # ---------------------------------------------------------------------------
    # 5. Abstention & Selective Classification Metrics
    # ---------------------------------------------------------------------------
    @staticmethod
    def calculate_abstention_metrics(
        y_true: List[Any],
        y_pred: List[Any],
        confidences: List[float],
        threshold: float = 0.5,
        abstain_label: str = "INSUFFICIENT",
    ) -> Dict[str, Any]:
        """
        Computes Coverage, Selective Accuracy, and Risk-Coverage curve.
        """
        if not y_true or not y_pred or not confidences:
            return {}

        total_samples = len(y_true)
        # Decision rule: if confidence < threshold or pred == abstain_label, abstain
        covered_indices = [
            i for i in range(total_samples)
            if confidences[i] >= threshold and y_pred[i] != abstain_label
        ]

        coverage = len(covered_indices) / total_samples if total_samples > 0 else 0.0

        if covered_indices:
            covered_true = [y_true[i] for i in covered_indices]
            covered_pred = [y_pred[i] for i in covered_indices]
            correct = sum(1 for t, p in zip(covered_true, covered_pred) if t == p)
            selective_acc = correct / len(covered_indices)
            selective_risk = 1.0 - selective_acc
        else:
            selective_acc = 0.0
            selective_risk = 0.0

        # Generate Risk-Coverage curve over threshold sweep [0.1 .. 0.9]
        curve = []
        for t_step in np.linspace(0.1, 0.9, 9):
            subset = [
                i for i in range(total_samples)
                if confidences[i] >= t_step and y_pred[i] != abstain_label
            ]
            c_cov = len(subset) / total_samples
            if subset:
                c_acc = sum(1 for i in subset if y_true[i] == y_pred[i]) / len(subset)
                c_risk = 1.0 - c_acc
            else:
                c_acc = 0.0
                c_risk = 0.0
            curve.append({
                "threshold": round(float(t_step), 2),
                "coverage": round(c_cov, 4),
                "selective_accuracy": round(c_acc, 4),
                "selective_risk": round(c_risk, 4),
            })

        return {
            "threshold": threshold,
            "coverage": round(coverage, 4),
            "selective_accuracy": round(selective_acc, 4),
            "selective_risk": round(selective_risk, 4),
            "risk_coverage_curve": curve,
        }

    # ---------------------------------------------------------------------------
    # 6. OCR Text Error Rate Metrics (CER / WER)
    # ---------------------------------------------------------------------------
    @staticmethod
    def _levenshtein_distance(ref: List[Any], hyp: List[Any]) -> int:
        dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            dp[i][0] = i
        for j in range(len(hyp) + 1):
            dp[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[len(ref)][len(hyp)]

    @classmethod
    def calculate_ocr_metrics(
        cls,
        references: List[str],
        hypotheses: List[str],
    ) -> Dict[str, float]:
        """Computes Character Error Rate (CER) and Word Error Rate (WER)."""
        if not references or not hypotheses:
            return {}

        total_char_dist = 0
        total_ref_chars = 0
        total_word_dist = 0
        total_ref_words = 0

        for ref, hyp in zip(references, hypotheses):
            # CER
            ref_chars = list(ref)
            hyp_chars = list(hyp)
            total_char_dist += cls._levenshtein_distance(ref_chars, hyp_chars)
            total_ref_chars += max(len(ref_chars), 1)

            # WER
            ref_words = ref.split()
            hyp_words = hyp.split()
            total_word_dist += cls._levenshtein_distance(ref_words, hyp_words)
            total_ref_words += max(len(ref_words), 1)

        cer = total_char_dist / total_ref_chars if total_ref_chars > 0 else 0.0
        wer = total_word_dist / total_ref_words if total_ref_words > 0 else 0.0

        return {
            "cer": round(cer, 4),
            "wer": round(wer, 4),
        }
