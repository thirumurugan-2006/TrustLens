import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from app.retrieval_dataset.schemas import NegativeType, RetrievalExample, RetrievalPair
from app.training.schemas import EvidenceRelationLabel, ProvenanceSourceType, SplitName


class RetrievalDatasetValidator:
    """
    Validates structural integrity, provenance, negative safety, and cluster leakage
    for the TrustLens Retrieval and Reranking Dataset.
    """

    def __init__(self, data_dir: str = "data/processed/retrieval"):
        self.data_dir = Path(data_dir)
        self.parse_errors: List[str] = []

    def load_examples(self) -> List[RetrievalExample]:
        """Loads all RetrievalExample records from train, validation, and test jsonl files."""
        self.parse_errors = []
        examples: List[RetrievalExample] = []
        for sname in ["train.jsonl", "validation.jsonl", "test.jsonl"]:
            file_path = self.data_dir / sname
            if not file_path.is_file():
                continue
            with file_path.open("r", encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line.strip())
                        examples.append(RetrievalExample.model_validate(data))
                    except Exception as e:
                        self.parse_errors.append(f"PARSE_ERROR in {sname} line {line_no}: {e}")
        return examples

    def validate(self) -> Dict[str, Any]:
        """Executes full validation checks across the dataset files."""
        examples = self.load_examples()
        errors: List[str] = list(self.parse_errors)
        warnings: List[str] = []

        if not examples:
            if not errors:
                errors.append("NO_RECORDS_FOUND: No retrieval examples found in data directory.")
            return {
                "valid": False,
                "is_valid": False,
                "errors": errors,
                "warnings": warnings,
                "counts": {},
                "duplicate_pairs": 0,
                "cross_split_leakage": 0,
                "test_count": 0,
            }

        # 1. Traceability & Required Fields
        seen_pairs: Set[Tuple[str, str]] = set()
        duplicate_pairs_count = 0
        invalid_positives_count = 0
        invalid_negatives_count = 0
        missing_provenance_count = 0
        synthetic_in_test_count = 0

        cluster_splits: Dict[str, Set[str]] = {}
        claim_splits: Dict[str, Set[str]] = {}
        evidence_splits: Dict[str, Set[str]] = {}

        split_counts = {"train": 0, "validation": 0, "test": 0}
        cross_lang_count = 0
        unique_queries: Set[str] = set()
        unique_positives: Set[str] = set()
        total_negatives = 0

        for ex in examples:
            s_val = ex.split.value if hasattr(ex.split, "value") else str(ex.split)
            split_counts[s_val] = split_counts.get(s_val, 0) + 1
            unique_queries.add(ex.query_id)
            unique_positives.add(ex.positive_evidence_id)

            if ex.cross_language:
                cross_lang_count += 1

            # Check provenance
            if not ex.provenance or not ex.provenance.source_name:
                missing_provenance_count += 1

            # Synthetic in test
            is_synth = (ex.provenance and ex.provenance.source_type == ProvenanceSourceType.SYNTHETIC)
            if is_synth and s_val == "test":
                synthetic_in_test_count += 1

            # Positive relation safety
            if ex.positive_relation not in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
                invalid_positives_count += 1

            # Duplicate pair check (query_id, positive_id)
            pair_key_pos = (ex.query_id, ex.positive_evidence_id)
            if pair_key_pos in seen_pairs:
                duplicate_pairs_count += 1
            else:
                seen_pairs.add(pair_key_pos)

            # Negative safety checks
            for neg in ex.negatives:
                total_negatives += 1
                pair_key_neg = (ex.query_id, neg.evidence_id)
                if pair_key_neg in seen_pairs:
                    duplicate_pairs_count += 1
                else:
                    seen_pairs.add(pair_key_neg)

                # Critical check: Negative must NOT support or contradict the claim
                if neg.evidence_relation in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
                    invalid_negatives_count += 1

            # Leakage grouping
            cluster_splits.setdefault(ex.cluster_id, set()).add(s_val)
            claim_splits.setdefault(ex.claim_id, set()).add(s_val)
            evidence_splits.setdefault(ex.positive_evidence_id, set()).add(s_val)

        # 2. Leakage violations
        leaking_clusters = {c: splits for c, splits in cluster_splits.items() if len(splits) > 1}
        leaking_claims = {c: splits for c, splits in claim_splits.items() if len(splits) > 1}
        leaking_evidence = {e: splits for e, splits in evidence_splits.items() if len(splits) > 1}
        cross_split_leakage = len(leaking_clusters) + len(leaking_claims) + len(leaking_evidence)

        if cross_split_leakage > 0:
            errors.append(f"CROSS_SPLIT_LEAKAGE: {cross_split_leakage} clusters or entities span multiple splits.")
        if invalid_negatives_count > 0:
            errors.append(f"INVALID_NEGATIVES: {invalid_negatives_count} negatives have SUPPORTS or CONTRADICTS relations.")
        if invalid_positives_count > 0:
            errors.append(f"INVALID_POSITIVES: {invalid_positives_count} positives lack SUPPORTS or CONTRADICTS relation.")
        if synthetic_in_test_count > 0:
            errors.append(f"SYNTHETIC_IN_TEST: {synthetic_in_test_count} synthetic records found in test split.")
        if missing_provenance_count > 0:
            errors.append(f"MISSING_PROVENANCE: {missing_provenance_count} examples lack valid provenance metadata.")
        if duplicate_pairs_count > 0:
            errors.append(f"DUPLICATE_PAIRS: {duplicate_pairs_count} duplicate query-evidence pairs detected.")

        is_valid = len(errors) == 0

        return {
            "valid": is_valid,
            "is_valid": is_valid,
            "cross_split_leakage": cross_split_leakage,
            "duplicate_pairs": duplicate_pairs_count,
            "test_count": split_counts["test"],
            "errors": errors,
            "warnings": warnings,
            "counts": {
                "total_examples": len(examples),
                "queries": len(unique_queries),
                "positive_evidence": len(unique_positives),
                "hard_negatives": total_negatives,
                "cross_language_pairs": cross_lang_count,
                "train": split_counts["train"],
                "validation": split_counts["validation"],
                "test": split_counts["test"],
                "cross_split_leakage": cross_split_leakage,
                "invalid_negatives": invalid_negatives_count,
            },
        }


def main():
    parser = argparse.ArgumentParser(description="TrustLens Retrieval Dataset Validator")
    parser.add_argument("--data-dir", default="data/processed/retrieval", help="Retrieval data directory")
    args = parser.parse_args()

    validator = RetrievalDatasetValidator(data_dir=args.data_dir)
    res = validator.validate()
    counts = res.get("counts", {})

    print("\n================ TRUSTLENS RETRIEVAL DATASET VALIDATION ================")
    print(f"Queries:              {counts.get('queries', 0)}")
    print(f"Positive evidence:    {counts.get('positive_evidence', 0)}")
    print(f"Hard negatives:       {counts.get('hard_negatives', 0)}")
    print(f"Cross-language pairs: {counts.get('cross_language_pairs', 0)}")
    print(f"Train:                {counts.get('train', 0)}")
    print(f"Validation:           {counts.get('validation', 0)}")
    print(f"Test:                 {counts.get('test', 0)}")
    print(f"Cross-split leakage:  {counts.get('cross_split_leakage', 0)}")
    print(f"Invalid negatives:    {counts.get('invalid_negatives', 0)}")
    print(f"Overall:              {'VALID' if res.get('valid') else 'INVALID'}")
    print("=======================================================================\n")

    if not res.get("valid"):
        print("Validation Failures:")
        for err in res.get("errors", []):
            print(f"  [ERROR] {err}")
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
