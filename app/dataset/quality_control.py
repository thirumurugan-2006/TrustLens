from typing import Any, Dict, List, Optional, Set

from app.dataset.schemas import HumanAnnotationSubmission, QCResult
from app.training.schemas import (
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    ProvenanceMetadata,
    ProvenanceSourceType,
    RiskLevel,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


class DatasetQualityControl:
    """
    Automated Quality Control (QC) validator enforcing dataset integrity,
    label constraints, anti-leakage rules, and multi-annotator agreement.
    """

    VALID_LANGUAGES = {"en", "ta", "hi", "te", "kn", "ml", "bn", "mr", "gu", "pa"}
    VALID_SCRIPTS = {"Latin", "Tamil", "Devanagari", "Telugu", "Kannada", "Malayalam"}

    @classmethod
    def validate_post(cls, post: TrainingPost) -> QCResult:
        """Validates a TrainingPost for core schema and provenance integrity."""
        reasons = []
        warnings = []

        if not post.post_id:
            reasons.append("MISSING_REQUIRED_FIELDS: post_id is empty")

        if not post.content.text or not post.content.text.strip():
            reasons.append("MISSING_CANONICAL_TEXT: content.text is empty")

        if not post.provenance:
            reasons.append("MISSING_PROVENANCE: provenance metadata is missing")
        elif not post.provenance.source_name:
            reasons.append("MISSING_PROVENANCE: source_name is empty")

        if not post.language_info or not post.language_info.primary:
            reasons.append("INVALID_LANGUAGE: primary language is missing")
        elif post.language_info.primary.lower() not in cls.VALID_LANGUAGES:
            warnings.append(f"Uncommon language tag: {post.language_info.primary}")

        passed = len(reasons) == 0
        return QCResult(
            passed=passed,
            status="ACCEPTED" if passed else "REJECTED",
            reason_codes=reasons,
            warnings=warnings,
            details={"post_id": post.post_id},
        )

    @classmethod
    def validate_claim(cls, claim: TrainingClaim, existing_post_ids: Optional[Set[str]] = None) -> QCResult:
        """Validates a TrainingClaim for label adherence and reference integrity."""
        reasons = []
        warnings = []

        if not claim.claim_id:
            reasons.append("MISSING_REQUIRED_FIELDS: claim_id is empty")

        if not claim.claim_text or not claim.claim_text.strip():
            reasons.append("MISSING_CANONICAL_TEXT: claim_text is empty")

        if existing_post_ids and claim.post_id not in existing_post_ids:
            reasons.append(f"BROKEN_REFERENCES: post_id {claim.post_id} not found in post index")

        if not isinstance(claim.claim_type, ClaimType):
            reasons.append(f"INVALID_LABELS: invalid claim_type '{claim.claim_type}'")

        if not isinstance(claim.detection_label, ClaimDetectionLabel):
            reasons.append(f"INVALID_LABELS: invalid detection_label '{claim.detection_label}'")

        if not (0.0 <= claim.check_worthiness <= 1.0):
            reasons.append("INVALID_LABELS: check_worthiness out of range [0.0, 1.0]")

        if not claim.provenance:
            reasons.append("MISSING_PROVENANCE: provenance is missing")

        passed = len(reasons) == 0
        return QCResult(
            passed=passed,
            status="ACCEPTED" if passed else "REJECTED",
            reason_codes=reasons,
            warnings=warnings,
            details={"claim_id": claim.claim_id},
        )

    @classmethod
    def validate_evidence(cls, evidence: TrainingEvidence, existing_claim_ids: Optional[Set[str]] = None) -> QCResult:
        """Validates a TrainingEvidence record for source reliability and relation labels."""
        reasons = []
        warnings = []

        if not evidence.evidence_id:
            reasons.append("MISSING_REQUIRED_FIELDS: evidence_id is empty")

        if not evidence.evidence_text or not evidence.evidence_text.strip():
            reasons.append("MISSING_CANONICAL_TEXT: evidence_text is empty")

        if existing_claim_ids and evidence.claim_id not in existing_claim_ids:
            reasons.append(f"BROKEN_REFERENCES: claim_id {evidence.claim_id} not found in claim index")

        if not isinstance(evidence.relation_label, EvidenceRelationLabel):
            reasons.append(f"INVALID_EVIDENCE_RELATION: invalid relation '{evidence.relation_label}'")

        if not evidence.provenance:
            reasons.append("MISSING_PROVENANCE: provenance is missing")

        passed = len(reasons) == 0
        return QCResult(
            passed=passed,
            status="ACCEPTED" if passed else "REJECTED",
            reason_codes=reasons,
            warnings=warnings,
            details={"evidence_id": evidence.evidence_id},
        )

    @classmethod
    def validate_risk(cls, risk: TrainingRisk, existing_post_ids: Optional[Set[str]] = None) -> QCResult:
        """Validates a TrainingRisk record for objective ground truth constraints."""
        reasons = []
        warnings = []

        if not risk.risk_id:
            reasons.append("MISSING_REQUIRED_FIELDS: risk_id is empty")

        if existing_post_ids and risk.post_id not in existing_post_ids:
            reasons.append(f"BROKEN_REFERENCES: post_id {risk.post_id} not found in post index")

        if not isinstance(risk.risk_level, RiskLevel):
            reasons.append(f"INVALID_RISK_LABEL: invalid risk_level '{risk.risk_level}'")

        if risk.risk_level == RiskLevel.HIGH and not risk.risk_factors:
            warnings.append("HIGH risk record contains empty risk_factors list")

        if not risk.provenance:
            reasons.append("MISSING_PROVENANCE: provenance is missing")

        passed = len(reasons) == 0
        return QCResult(
            passed=passed,
            status="ACCEPTED" if passed else "REJECTED",
            reason_codes=reasons,
            warnings=warnings,
            details={"risk_id": risk.risk_id},
        )

    @classmethod
    def check_split_integrity(
        cls,
        post: TrainingPost,
        split: SplitName,
        assigned_splits: Dict[str, SplitName],
        composite_cluster_id: str,
    ) -> QCResult:
        """
        Enforces partition constraints:
        1. Synthetic data is prohibited from test.
        2. All members of composite_cluster_id must belong to the exact same split.
        """
        reasons = []
        warnings = []

        # 1. Synthetic data in test
        if split == SplitName.test:
            if post.is_example or post.provenance.source_type == ProvenanceSourceType.SYNTHETIC:
                reasons.append("SYNTHETIC_IN_TEST: synthetic record prohibited from test split")

        # 2. Cross-split cluster leakage
        if composite_cluster_id in assigned_splits:
            expected_split = assigned_splits[composite_cluster_id]
            if split != expected_split:
                reasons.append(
                    f"CROSS_SPLIT_LEAKAGE: cluster {composite_cluster_id} previously assigned to {expected_split}, now conflicting with {split}"
                )

        passed = len(reasons) == 0
        return QCResult(
            passed=passed,
            status="ACCEPTED" if passed else "REJECTED",
            reason_codes=reasons,
            warnings=warnings,
            details={"post_id": post.post_id, "cluster_id": composite_cluster_id},
        )

    @classmethod
    def check_annotator_agreement(
        cls,
        submissions: List[HumanAnnotationSubmission],
    ) -> Dict[str, Any]:
        """
        Computes inter-annotator agreement across multiple submissions for a task.
        Preserves disagreements without overwriting.
        """
        if len(submissions) < 2:
            return {"status": "SINGLE_ANNOTATOR", "agreement": 1.0, "disagreements": []}

        risk_labels = [s.risk_level for s in submissions if s.risk_level]
        claim_types = [s.claim_type for s in submissions if s.claim_type]

        risk_agrees = len(set(risk_labels)) <= 1 if risk_labels else True
        claim_agrees = len(set(claim_types)) <= 1 if claim_types else True

        disagreements = []
        if not risk_agrees:
            disagreements.append(f"Risk level disagreement: {[r.value for r in risk_labels if r]}")
        if not claim_agrees:
            disagreements.append(f"Claim type disagreement: {[c.value for c in claim_types if c]}")

        has_consensus = risk_agrees and claim_agrees
        return {
            "status": "CONSENSUS" if has_consensus else "DISAGREEMENT",
            "agreement_rate": 1.0 if has_consensus else 0.0,
            "disagreements": disagreements,
            "annotators": [s.annotator_id for s in submissions],
        }
