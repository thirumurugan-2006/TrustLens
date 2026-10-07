import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.retrieval_dataset.manifest import RetrievalManifestBuilder
from app.retrieval_dataset.pair_builder import RetrievalPairBuilder
from app.retrieval_dataset.schemas import (
    RetrievalEvaluationItem,
    RetrievalExample,
    RetrievalPair,
)
from app.retrieval_dataset.splitter import RetrievalDatasetSplitter
from app.retrieval_dataset.validator import RetrievalDatasetValidator
from app.training.schemas import (
    EvidenceRelationLabel,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
)


class RetrievalDatasetBuilder:
    """
    Main builder for the TrustLens Retrieval and Reranking Dataset (v0.1.0).
    Constructs train, validation, and test splits with cluster-isolated hard negatives
    from the v0.2.0 TrustLens benchmark dataset.
    """

    def __init__(
        self,
        data_dir: str = "data/trustlens",
        output_dir: str = "data/processed/retrieval",
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.pair_builder = RetrievalPairBuilder()
        self.splitter = RetrievalDatasetSplitter()

    def load_source_data(self) -> Tuple[List[TrainingClaim], List[TrainingEvidence]]:
        """Loads claims and evidence from the source TrustLens dataset."""
        claims: List[TrainingClaim] = []
        evidence_list: List[TrainingEvidence] = []

        # 1. Claims
        claims_file = self.data_dir / "annotated" / "claims.jsonl"
        if not claims_file.is_file():
            claims_file = self.data_dir / "claims.jsonl"
        if claims_file.is_file():
            with claims_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        claims.append(TrainingClaim.model_validate(json.loads(line.strip())))

        # 2. Evidence
        ev_file = self.data_dir / "annotated" / "evidence.jsonl"
        if not ev_file.is_file():
            ev_file = self.data_dir / "evidence.jsonl"
        if ev_file.is_file():
            with ev_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        evidence_list.append(TrainingEvidence.model_validate(json.loads(line.strip())))

        return claims, evidence_list

    def build_dataset(
        self,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes query synthesis, positive linking, hard-negative mining,
        cluster isolation, splitting, and persistence.
        """
        claims, evidence_list = self.load_source_data()

        # Build records
        examples, pairs, eval_items = self.pair_builder.build_dataset_from_records(
            claims=claims,
            evidence_list=evidence_list,
            human_review_ratio=0.25,
            max_negatives_per_query=3,
        )

        # Partition examples and pairs
        split_examples = self.splitter.partition_examples(examples)
        split_pairs = self.splitter.partition_pairs(pairs)

        # Manifest
        manifest = RetrievalManifestBuilder.build_manifest(
            examples=examples,
            pairs=pairs,
            dataset_version="v0.1.0",
            source_dataset_version="v0.2.0",
        )

        clusters = {ex.cluster_id for ex in examples}
        lang_dist = manifest.language_distribution

        result_summary = {
            "claims_discovered": len(claims),
            "evidence_discovered": len(evidence_list),
            "queries_discovered": manifest.query_count,
            "accepted_positives": manifest.positive_count,
            "accepted_negatives": manifest.negative_count,
            "total_examples": len(examples),
            "total_pairs": len(pairs),
            "train_examples": manifest.train_count,
            "validation_examples": manifest.validation_count,
            "test_examples": manifest.test_count,
            "clusters": len(clusters),
            "cross_language_pairs": manifest.cross_language_count,
            "language_distribution": lang_dist,
            "negative_types": manifest.negative_type_distribution,
            "human_reviewed_count": manifest.human_reviewed_count,
            "dry_run": dry_run,
        }

        if dry_run:
            return result_summary

        # Write files
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._write_jsonl(self.output_dir / "train.jsonl", split_examples["train"])
        self._write_jsonl(self.output_dir / "validation.jsonl", split_examples["validation"])
        self._write_jsonl(self.output_dir / "test.jsonl", split_examples["test"])

        self._write_jsonl(self.output_dir / "pairs_train.jsonl", split_pairs["train"])
        self._write_jsonl(self.output_dir / "pairs_validation.jsonl", split_pairs["validation"])
        self._write_jsonl(self.output_dir / "pairs_test.jsonl", split_pairs["test"])

        self._write_jsonl(self.output_dir / "eval_test_pool.jsonl", eval_items)

        RetrievalManifestBuilder.save_manifest(manifest, self.output_dir / "dataset_manifest.json")

        # Run validation
        validator = RetrievalDatasetValidator(data_dir=str(self.output_dir))
        val_res = validator.validate()
        result_summary["validation"] = val_res

        return result_summary

    @staticmethod
    def _write_jsonl(path: Path, items: List[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                data = item.model_dump() if hasattr(item, "model_dump") else item
                f.write(json.dumps(data, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="TrustLens Retrieval Dataset Builder")
    parser.add_argument("--data-dir", default="data/trustlens", help="Source TrustLens dataset directory")
    parser.add_argument("--output-dir", default="data/processed/retrieval", help="Destination output directory")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without writing files")

    args = parser.parse_args()

    builder = RetrievalDatasetBuilder(data_dir=args.data_dir, output_dir=args.output_dir)
    res = builder.build_dataset(dry_run=args.dry_run)

    print("\n================ TRUSTLENS RETRIEVAL DATASET BUILDER ================")
    print(f"Mode:                 {'DRY RUN (No files written)' if res.get('dry_run') else 'LIVE EXECUTION'}")
    print(f"Claims Discovered:    {res.get('claims_discovered')}")
    print(f"Evidence Discovered:  {res.get('evidence_discovered')}")
    print(f"Queries Generated:    {res.get('queries_discovered')}")
    print(f"Accepted Positives:   {res.get('accepted_positives')}")
    print(f"Accepted Negatives:   {res.get('accepted_negatives')}")
    print(f"Total Examples:       {res.get('total_examples')}")
    print(f"Total Pairs:          {res.get('total_pairs')}")
    print(f"Splits:               Train={res.get('train_examples')}, Val={res.get('validation_examples')}, Test={res.get('test_examples')}")
    print(f"Clusters Formed:      {res.get('clusters')}")
    print(f"Cross-Language:       {res.get('cross_language_pairs')}")
    print(f"Human-Reviewed:       {res.get('human_reviewed_count')}")
    print("\nLanguage Distribution:")
    for lang, count in res.get("language_distribution", {}).items():
        print(f"  - {lang}: {count}")

    print("\nNegative Types Distribution:")
    for nt, count in res.get("negative_types", {}).items():
        print(f"  - {nt}: {count}")

    if not res.get("dry_run"):
        v_res = res.get("validation", {})
        print(f"\nValidation Result:    {'PASSED (VALID)' if v_res.get('valid') else 'FAILED (INVALID)'}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
