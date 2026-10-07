import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.dataset.quality_control import DatasetQualityControl
from app.dataset.schemas import DatasetValidationReport, QCResult
from app.training.schemas import (
    ProvenanceSourceType,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


class DatasetValidator:
    """
    Validates dataset integrity, schema conformity, provenance preservation,
    leakage prevention across splits, and strict quarantine of synthetic data.
    """

    def __init__(self, data_dir: str = "data/trustlens"):
        self.data_dir = Path(data_dir)

    def load_posts_from_dir(self, directory: Optional[Path] = None) -> List[TrainingPost]:
        """Loads all TrainingPost records from JSONL files in directory or its split folders."""
        dir_to_scan = directory or self.data_dir
        posts: List[TrainingPost] = []
        if not dir_to_scan.exists():
            return posts

        seen_ids = set()

        # Check if root has posts.jsonl
        root_posts = dir_to_scan / "posts.jsonl"
        if root_posts.is_file():
            candidates = [root_posts]
        else:
            candidates = list(dir_to_scan.glob("*.jsonl")) + list(dir_to_scan.glob("*/*.jsonl"))

        for jf in candidates:
            # Skip non-post files
            if any(skip_word in jf.name for skip_word in ["task", "queue", "rejected", "evidence", "claim", "risk"]):
                continue
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            data = json.loads(line_s)
                            post = TrainingPost.model_validate(data)
                            if post.post_id not in seen_ids:
                                seen_ids.add(post.post_id)
                                posts.append(post)
                        except Exception:
                            pass
        return posts

    def load_claims_from_dir(self, directory: Optional[Path] = None) -> List[TrainingClaim]:
        """Loads all TrainingClaim records from JSONL files in directory."""
        dir_to_scan = directory or self.data_dir
        claims: List[TrainingClaim] = []
        if not dir_to_scan.exists():
            return claims
        candidates = list(dir_to_scan.glob("*claim*.jsonl")) + list(dir_to_scan.glob("*/*claim*.jsonl"))
        seen_ids = set()
        for jf in candidates:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            data = json.loads(line_s)
                            c = TrainingClaim.model_validate(data)
                            if c.claim_id not in seen_ids:
                                seen_ids.add(c.claim_id)
                                claims.append(c)
                        except Exception:
                            pass
        return claims

    def load_evidence_from_dir(self, directory: Optional[Path] = None) -> List[TrainingEvidence]:
        """Loads all TrainingEvidence records from JSONL files in directory."""
        dir_to_scan = directory or self.data_dir
        evidence: List[TrainingEvidence] = []
        if not dir_to_scan.exists():
            return evidence
        candidates = list(dir_to_scan.glob("*evidence*.jsonl")) + list(dir_to_scan.glob("*/*evidence*.jsonl"))
        seen_ids = set()
        for jf in candidates:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            data = json.loads(line_s)
                            e = TrainingEvidence.model_validate(data)
                            if e.evidence_id not in seen_ids:
                                seen_ids.add(e.evidence_id)
                                evidence.append(e)
                        except Exception:
                            pass
        return evidence

    def load_risks_from_dir(self, directory: Optional[Path] = None) -> List[TrainingRisk]:
        """Loads all TrainingRisk records from JSONL files in directory."""
        dir_to_scan = directory or self.data_dir
        risks: List[TrainingRisk] = []
        if not dir_to_scan.exists():
            return risks
        candidates = list(dir_to_scan.glob("*risk*.jsonl")) + list(dir_to_scan.glob("*/*risk*.jsonl"))
        seen_ids = set()
        for jf in candidates:
            with jf.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        try:
                            data = json.loads(line_s)
                            r = TrainingRisk.model_validate(data)
                            if r.risk_id not in seen_ids:
                                seen_ids.add(r.risk_id)
                                risks.append(r)
                        except Exception:
                            pass
        return risks

    def load_posts_from_file(self, file_path: Path) -> List[TrainingPost]:
        """Loads posts directly from a single JSONL file."""
        posts: List[TrainingPost] = []
        if not file_path.is_file():
            return posts
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_s = line.strip()
                if line_s:
                    data = json.loads(line_s)
                    posts.append(TrainingPost.model_validate(data))
        return posts

    def validate_dataset(
        self,
        posts: List[TrainingPost],
        claims: Optional[List[TrainingClaim]] = None,
        evidence: Optional[List[TrainingEvidence]] = None,
        risks: Optional[List[TrainingRisk]] = None,
    ) -> DatasetValidationReport:
        """
        Runs full comprehensive dataset validation suite.
        """
        errors: List[str] = []
        warnings: List[str] = []

        total_records = len(posts)
        if total_records == 0:
            return DatasetValidationReport(
                valid=False,
                total_records=0,
                schema_valid=False,
                provenance_valid=False,
                duplicates_detected=0,
                clusters_count=0,
                train_count=0,
                validation_count=0,
                test_count=0,
                synthetic_in_test_violations=0,
                cross_split_leakage_violations=0,
                language_distribution={},
                risk_distribution={},
                claim_distribution={},
                errors=["Dataset is empty. No training posts found."],
            )

        post_ids: Set[str] = set()
        text_hashes: Set[str] = set()
        duplicates_detected = 0

        lang_dist: Dict[str, int] = {}
        split_counts = {"train": 0, "validation": 0, "test": 0, "unassigned": 0}

        # Cluster to splits mapping: cluster_id -> set of split names
        cluster_splits: Dict[str, Set[str]] = {}
        synthetic_in_test_violations = 0

        schema_valid = True
        provenance_valid = True

        for post in posts:
            # 1. Post Schema QC
            qc = DatasetQualityControl.validate_post(post)
            if not qc.passed:
                schema_valid = False
                for r in qc.reason_codes:
                    errors.append(f"Post {post.post_id} failed QC: {r}")

            # 2. Provenance Validity
            if not post.provenance or not post.provenance.source_name:
                provenance_valid = False
                errors.append(f"Post {post.post_id} missing valid provenance.")

            # 3. Post ID uniqueness
            if post.post_id in post_ids:
                duplicates_detected += 1
                errors.append(f"Duplicate post_id detected: {post.post_id}")
            post_ids.add(post.post_id)

            # 4. Text duplicate tracking
            norm_text = " ".join((post.content.text or "").lower().split())
            if norm_text in text_hashes:
                duplicates_detected += 1
            else:
                text_hashes.add(norm_text)

            # 5. Language distribution
            lang = post.language_info.primary if post.language_info else "unknown"
            lang_dist[lang] = lang_dist.get(lang, 0) + 1

            # 6. Split and Cluster tracking
            split_name = post.split_info.split.value if post.split_info and post.split_info.split else "unassigned"
            split_counts[split_name] = split_counts.get(split_name, 0) + 1

            # Synthetic in test check
            is_synthetic = post.is_example or (
                post.provenance and post.provenance.source_type == ProvenanceSourceType.SYNTHETIC
            )
            if is_synthetic and split_name == "test":
                synthetic_in_test_violations += 1
                errors.append(
                    f"SYNTHETIC_TEST_VIOLATION: Synthetic post {post.post_id} found in test split!"
                )

            # Cluster leakage tracking
            if post.split_info:
                # Track cluster keys
                cluster_keys = []
                if post.split_info.campaign_group_id:
                    cluster_keys.append(f"camp_{post.split_info.campaign_group_id}")
                if post.split_info.translation_group_id:
                    cluster_keys.append(f"trans_{post.split_info.translation_group_id}")
                if post.split_info.post_family_id:
                    cluster_keys.append(f"postfam_{post.split_info.post_family_id}")

                for ck in cluster_keys:
                    cluster_splits.setdefault(ck, set()).add(split_name)

        # Cross-split leakage violations
        cross_split_leakage_violations = 0
        for ck, assigned_splits in cluster_splits.items():
            valid_splits = {s for s in assigned_splits if s != "unassigned"}
            if len(valid_splits) > 1:
                cross_split_leakage_violations += 1
                errors.append(
                    f"CROSS_SPLIT_LEAKAGE: Cluster {ck} spans multiple splits: {sorted(list(valid_splits))}"
                )

        # Claim distribution
        claim_dist: Dict[str, int] = {}
        if claims:
            for c in claims:
                qc_c = DatasetQualityControl.validate_claim(c, existing_post_ids=post_ids)
                if not qc_c.passed:
                    for r in qc_c.reason_codes:
                        errors.append(f"Claim {c.claim_id} failed QC: {r}")
                c_type = c.claim_type.value if hasattr(c.claim_type, "value") else str(c.claim_type)
                claim_dist[c_type] = claim_dist.get(c_type, 0) + 1

        # Evidence distribution
        if evidence:
            claim_ids = {c.claim_id for c in claims} if claims else None
            for e in evidence:
                qc_e = DatasetQualityControl.validate_evidence(e, existing_claim_ids=claim_ids)
                if not qc_e.passed:
                    for r in qc_e.reason_codes:
                        errors.append(f"Evidence {e.evidence_id} failed QC: {r}")

        # Risk distribution
        risk_dist: Dict[str, int] = {}
        if risks:
            for r in risks:
                qc_r = DatasetQualityControl.validate_risk(r, existing_post_ids=post_ids)
                if not qc_r.passed:
                    for err in qc_r.reason_codes:
                        errors.append(f"Risk {r.risk_id} failed QC: {err}")
                r_lvl = r.risk_level.value if hasattr(r.risk_level, "value") else str(r.risk_level)
                risk_dist[r_lvl] = risk_dist.get(r_lvl, 0) + 1

        is_valid = (
            schema_valid
            and provenance_valid
            and synthetic_in_test_violations == 0
            and cross_split_leakage_violations == 0
            and len(errors) == 0
        )

        return DatasetValidationReport(
            valid=is_valid,
            total_records=total_records,
            schema_valid=schema_valid,
            provenance_valid=provenance_valid,
            duplicates_detected=duplicates_detected,
            clusters_count=len(cluster_splits),
            train_count=split_counts.get("train", 0),
            validation_count=split_counts.get("validation", 0),
            test_count=split_counts.get("test", 0),
            synthetic_in_test_violations=synthetic_in_test_violations,
            cross_split_leakage_violations=cross_split_leakage_violations,
            language_distribution=lang_dist,
            risk_distribution=risk_dist,
            claim_distribution=claim_dist,
            errors=errors,
            warnings=warnings,
        )


def main():
    parser = argparse.ArgumentParser(description="TrustLens Dataset Integrity Validator")
    parser.add_argument("--data-dir", default="data/trustlens", help="Directory containing dataset records")
    parser.add_argument("--file", default=None, help="Validate a specific JSONL file")

    args = parser.parse_args()

    validator = DatasetValidator(data_dir=args.data_dir)

    if args.file:
        posts = validator.load_posts_from_file(Path(args.file))
        report = validator.validate_dataset(posts)
    else:
        posts = validator.load_posts_from_dir()
        claims = validator.load_claims_from_dir()
        evidence = validator.load_evidence_from_dir()
        risks = validator.load_risks_from_dir()
        report = validator.validate_dataset(posts, claims=claims, evidence=evidence, risks=risks)

    readiness = "DATASET_READY" if report.valid else "DATASET_NOT_READY"

    print("\n================ TRUSTLENS DATASET VALIDATION REPORT ================")
    print(f"Overall Status:               {'VALID (PASSED)' if report.valid else 'INVALID (FAILED)'}")
    print(f"Readiness Status:             {readiness}")
    print(f"Total Records:                {report.total_records}")
    print(f"Schema Validity:              {'PASS' if report.schema_valid else 'FAIL'}")
    print(f"Provenance Validity:          {'PASS' if report.provenance_valid else 'FAIL'}")
    print(f"Duplicates Detected:          {report.duplicates_detected}")
    print(f"Clusters Evaluated:           {report.clusters_count}")
    print(f"Splits Distribution:          Train={report.train_count}, Val={report.validation_count}, Test={report.test_count}")
    print(f"Synthetic In Test Violations: {report.synthetic_in_test_violations}")
    print(f"Cross-Split Leakage:          {report.cross_split_leakage_violations}")
    print(f"Languages:                    {report.language_distribution}")
    print(f"Claims:                       {report.claim_distribution}")
    print(f"Risks:                        {report.risk_distribution}")

    if report.errors:
        print("\nIntegrity Errors:")
        for err in report.errors[:20]:
            print(f"  [ERROR] {err}")
        if len(report.errors) > 20:
            print(f"  ... and {len(report.errors) - 20} more errors")

    if report.warnings:
        print("\nWarnings:")
        for w in report.warnings[:10]:
            print(f"  [WARN] {w}")

    print("=====================================================================\n")

    sys.exit(0 if report.valid else 1)


if __name__ == "__main__":
    main()
