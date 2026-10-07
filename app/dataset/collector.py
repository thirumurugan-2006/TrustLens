import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.claims.decomposer import RuleBasedDecomposer
from app.claims.extractor import RuleBasedClaimExtractor
from app.dataset.annotation_queue import AnnotationQueue
from app.dataset.clusterer import DatasetClusterer
from app.dataset.deduplicator import DatasetDeduplicator
from app.dataset.normalizer import DatasetNormalizer
from app.dataset.provenance import ProvenanceError, ProvenanceManager
from app.dataset.schemas import (
    AnnotationTaskStatus,
    ClusterInfo,
    DuplicateInfo,
    RawDataRecord,
    SemanticAutoSuggestions,
)
from app.query_synthesis.query_generator import RuleBasedQuerySynthesizer
from app.training.schemas import ProvenanceMetadata, ProvenanceSourceType, TrainingPost
from app.validation.post_validator import validate_post


class DatasetCollector:
    """
    Controlled dataset ingestion and collection pipeline.
    Executes:
    1. Import & raw persistence
    2. Provenance assignment & validation
    3. Canonical normalization
    4. L2 validation routing (VALID -> annotation queue, INVALID -> rejected/quarantine)
    5. Deterministic deduplication
    6. Composite leakage clustering
    7. Semantic understanding suggestions (Rule-based L4 claims, decomposition, queries)
    8. Human annotation queue generation (strict separation from model outputs)
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        rejected_dir: str = "data/rejected",
        queue_dir: str = "data/annotation",
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.rejected_dir = Path(rejected_dir)
        self.queue_dir = Path(queue_dir)

        self.normalizer = DatasetNormalizer()
        self.deduplicator = DatasetDeduplicator()
        self.clusterer = DatasetClusterer()
        self.queue = AnnotationQueue()

        self.claim_extractor = RuleBasedClaimExtractor()
        self.claim_decomposer = RuleBasedDecomposer()
        self.query_synthesizer = RuleBasedQuerySynthesizer()

    def process_file(
        self,
        input_path: str,
        source_type: str,
        source_name: str,
        source_id: Optional[str] = None,
        license_str: Optional[str] = None,
        collection_method: str = "automated_collector",
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Runs the end-to-end ingestion pipeline on an input file.
        Returns a detailed execution summary.
        """
        path = Path(input_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        # 1. Parse raw records
        raw_records: List[RawDataRecord] = self.normalizer.parse_file(path)
        total_discovered = len(raw_records)

        # 2. Build & validate provenance
        provenance = ProvenanceManager.create_provenance(
            source_type=source_type,
            source_name=source_name,
            source_id=source_id,
            license_str=license_str,
            collection_method=collection_method,
        )

        valid_posts: List[TrainingPost] = []
        invalid_records: List[Dict[str, Any]] = []
        duplicate_records: List[Dict[str, Any]] = []
        enqueued_tasks = []

        language_dist: Dict[str, int] = {}
        category_dist: Dict[str, int] = {}

        for raw_rec in raw_records:
            # Canonical normalization
            try:
                post = self.normalizer.normalize_record(raw_rec, provenance)
            except Exception as e:
                invalid_records.append({
                    "raw_id": raw_rec.raw_id,
                    "reason": f"NORMALIZATION_FAILURE: {str(e)}",
                })
                continue

            # L2 Validation
            val_result = validate_post(post)
            if not val_result.valid:
                invalid_records.append({
                    "post_id": post.post_id,
                    "raw_id": raw_rec.raw_id,
                    "status": val_result.status.value,
                    "errors": val_result.errors,
                })
                continue

            # Deduplication
            dup_info = self.deduplicator.check_and_register(post)
            if dup_info.is_duplicate:
                post.metadata["duplicate_info"] = dup_info.model_dump()
                duplicate_records.append({
                    "post_id": post.post_id,
                    "duplicate_of": dup_info.duplicate_of,
                    "duplicate_type": dup_info.duplicate_type.value,
                    "duplicate_score": dup_info.duplicate_score,
                })

            # Clustering
            cluster_info = self.clusterer.register_post(post)
            post.metadata["cluster_info"] = cluster_info.model_dump()

            # Record stats
            lang = post.language_info.primary if post.language_info else "unknown"
            language_dist[lang] = language_dist.get(lang, 0) + 1

            # Semantic Processing (Auto-suggestions only, never ground-truth labels)
            extracted_claims = self.claim_extractor.extract_from_post(post)
            all_atomic = []
            for claim in extracted_claims:
                decomposed = self.claim_decomposer.decompose(claim)
                all_atomic.extend(decomposed.atomic_claims)
                ctype = claim.claim_type.upper() if claim.claim_type else "UNCLASSIFIED"
                category_dist[ctype] = category_dist.get(ctype, 0) + 1

            all_queries = self.query_synthesizer.synthesize(all_atomic) if all_atomic else []

            suggestions = SemanticAutoSuggestions(
                claims=extracted_claims,
                atomic_claims=all_atomic,
                search_queries=all_queries,
                is_auto_generated=True,
                generation_pipeline="TrustLens_L4_SemanticSuggestions",
            )

            # Enqueue into Annotation Queue
            task = self.queue.enqueue(post, auto_suggestions=suggestions, priority=1)
            enqueued_tasks.append(task)
            valid_posts.append(post)

        clusters = self.clusterer.get_all_clusters()

        summary = {
            "records_discovered": total_discovered,
            "records_parsed": len(raw_records),
            "records_valid": len(valid_posts),
            "records_invalid": len(invalid_records),
            "duplicates": len(duplicate_records),
            "clusters": len(clusters),
            "languages": language_dist,
            "categories": category_dist,
            "dry_run": dry_run,
        }

        if dry_run:
            return summary

        # -------------------------------------------------------------
        # Non-Dry Run: Persist artifacts separately
        # -------------------------------------------------------------
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        # 1. Preserve Raw Data
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        raw_out = self.raw_dir / f"{source_name}_{timestamp}.jsonl"
        with raw_out.open("w", encoding="utf-8") as f:
            for r in raw_records:
                f.write(json.dumps(r.model_dump(), ensure_ascii=False) + "\n")

        # 2. Persist Valid Normalized Posts
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        posts_out = self.processed_dir / f"posts_{source_name}_{timestamp}.jsonl"
        with posts_out.open("w", encoding="utf-8") as f:
            for p in valid_posts:
                f.write(json.dumps(p.model_dump(), ensure_ascii=False) + "\n")

        # 3. Persist Rejected / Quarantine Records
        if invalid_records:
            self.rejected_dir.mkdir(parents=True, exist_ok=True)
            rej_out = self.rejected_dir / f"rejected_{source_name}_{timestamp}.jsonl"
            with rej_out.open("w", encoding="utf-8") as f:
                for inv in invalid_records:
                    f.write(json.dumps(inv, ensure_ascii=False) + "\n")

        # 4. Persist Annotation Tasks
        self.queue_dir.mkdir(parents=True, exist_ok=True)
        queue_out = self.queue_dir / f"tasks_{source_name}_{timestamp}.jsonl"
        with queue_out.open("w", encoding="utf-8") as f:
            for task in enqueued_tasks:
                f.write(json.dumps(task.model_dump(), ensure_ascii=False) + "\n")

        summary["artifacts"] = {
            "raw": str(raw_out),
            "processed": str(posts_out),
            "queue": str(queue_out),
            "rejected": str(rej_out) if invalid_records else None,
        }

        return summary


def main():
    parser = argparse.ArgumentParser(description="TrustLens Data Collector & Normalization Pipeline")
    parser.add_argument("--input", required=True, help="Path to input raw data file (JSON, JSONL, CSV, TXT)")
    parser.add_argument(
        "--source-type",
        required=True,
        choices=[st.value for st in ProvenanceSourceType],
        help="Provenance source type (e.g. PUBLIC_DATASET, USER_PROVIDED, etc.)",
    )
    parser.add_argument("--source-name", required=True, help="Name of source provider or campaign")
    parser.add_argument("--source-id", default=None, help="Identifier within source dataset")
    parser.add_argument("--license", default=None, help="License identifier or declaration")
    parser.add_argument("--raw-dir", default="data/raw", help="Directory for preserved raw records")
    parser.add_argument("--processed-dir", default="data/processed", help="Directory for normalized records")
    parser.add_argument("--rejected-dir", default="data/rejected", help="Directory for rejected records")
    parser.add_argument("--queue-dir", default="data/annotation", help="Directory for annotation queue")
    parser.add_argument("--dry-run", action="store_true", help="Execute without writing output files")

    args = parser.parse_args()

    collector = DatasetCollector(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        rejected_dir=args.rejected_dir,
        queue_dir=args.queue_dir,
    )

    try:
        summary = collector.process_file(
            input_path=args.input,
            source_type=args.source_type,
            source_name=args.source_name,
            source_id=args.source_id,
            license_str=args.license,
            dry_run=args.dry_run,
        )

        print("\n================ TRUSTLENS DATA COLLECTION REPORT ================")
        print(f"Mode:                 {'DRY RUN' if summary['dry_run'] else 'WRITE MODE'}")
        print(f"Records Discovered:   {summary['records_discovered']}")
        print(f"Records Parsed:       {summary['records_parsed']}")
        print(f"Records Valid (L2):   {summary['records_valid']}")
        print(f"Records Invalid (L2): {summary['records_invalid']}")
        print(f"Duplicates Detected:  {summary['duplicates']}")
        print(f"Clusters Formed:      {summary['clusters']}")
        print(f"Languages:            {summary['languages']}")
        print(f"Categories:           {summary['categories']}")
        if "artifacts" in summary:
            print("Persisted Artifacts:")
            for k, v in summary["artifacts"].items():
                if v:
                    print(f"  - {k.capitalize()}: {v}")
        print("===================================================================\n")
        sys.exit(0)
    except Exception as e:
        print(f"Error executing data collection: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
