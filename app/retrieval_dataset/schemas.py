import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator

from app.training.schemas import (
    AnnotationConfidence,
    EvidenceRelationLabel,
    ProvenanceMetadata,
    ProvenanceSourceType,
    QueryType,
    SplitName,
)


class NegativeType(str, Enum):
    """Controlled taxonomy for negative evidence types in retrieval."""
    HARD_TOPIC_NEGATIVE = "HARD_TOPIC_NEGATIVE"
    HARD_ENTITY_NEGATIVE = "HARD_ENTITY_NEGATIVE"
    HARD_SEMANTIC_NEGATIVE = "HARD_SEMANTIC_NEGATIVE"
    CROSS_ENTITY_NEGATIVE = "CROSS_ENTITY_NEGATIVE"
    TEMPORAL_NEGATIVE = "TEMPORAL_NEGATIVE"
    IRRELEVANT_NEGATIVE = "IRRELEVANT_NEGATIVE"


class NegativeEvidenceRecord(BaseModel):
    """Representation of an individual negative evidence document."""
    evidence_id: str
    evidence_text: str
    negative_type: NegativeType
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    evidence_relation: Optional[EvidenceRelationLabel] = None
    candidate_similarity: Optional[float] = None
    provenance: Optional[ProvenanceMetadata] = None

    @model_validator(mode="before")
    @classmethod
    def sync_text_and_source(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "text" in data and "evidence_text" not in data:
                data["evidence_text"] = data["text"]
            if "source_type" in data and isinstance(data.get("source_type"), str) and "provenance" not in data:
                data["provenance"] = ProvenanceMetadata(
                    source_type=ProvenanceSourceType.PUBLIC_DATASET,
                    source_name=data.get("source_title") or "TrustLens Evidence Registry",
                )
        return data

    @property
    def text(self) -> str:
        return self.evidence_text

    @model_validator(mode="after")
    def validate_negative_relation(self):
        if self.evidence_relation in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
            raise ValueError(
                f"Negative evidence relation cannot be {self.evidence_relation}. "
                f"Negatives must never support or contradict the claim."
            )
        return self


class PositiveEvidenceRecord(BaseModel):
    """Representation of an individual positive (claim-relevant) evidence document."""
    evidence_id: str
    evidence_text: str
    relation: EvidenceRelationLabel
    source_title: Optional[str] = None
    source_url: Optional[str] = None
    provenance: Optional[ProvenanceMetadata] = None

    @model_validator(mode="before")
    @classmethod
    def sync_text(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "text" in data and "evidence_text" not in data:
                data["evidence_text"] = data["text"]
            if "source_type" in data and isinstance(data.get("source_type"), str) and "provenance" not in data:
                data["provenance"] = ProvenanceMetadata(
                    source_type=ProvenanceSourceType.PUBLIC_DATASET,
                    source_name=data.get("source_title") or "TrustLens Evidence Registry",
                )
        return data

    @property
    def text(self) -> str:
        return self.evidence_text

    @model_validator(mode="after")
    def validate_positive_relation(self):
        if self.relation not in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
            raise ValueError(
                f"Positive evidence relation must be SUPPORTS or CONTRADICTS, got '{self.relation}'"
            )
        return self


class RetrievalExample(BaseModel):
    """
    Dedicated core retrieval example (triplet and multi-negative unit).
    Guarantees full provenance, claim traceability, cluster isolation, and semantic integrity.
    """
    retrieval_id: str = Field(default_factory=lambda: f"ret_{uuid.uuid4().hex[:10]}")
    post_id: str
    claim_id: str
    atomic_claim_id: Optional[str] = None
    query_id: str
    query: str
    query_type: QueryType = QueryType.VERIFICATION

    # Positive Evidence
    positive_evidence_id: str = ""
    positive_evidence_text: str = ""
    positive_relation: EvidenceRelationLabel = EvidenceRelationLabel.CONTRADICTS
    positive: Optional[PositiveEvidenceRecord] = None

    # Primary Negative Evidence
    negative_evidence_id: str = ""
    negative_evidence_text: str = ""
    negative_type: NegativeType = NegativeType.HARD_TOPIC_NEGATIVE

    # Multiple Negatives Support
    negatives: List[NegativeEvidenceRecord] = Field(default_factory=list)

    # Linguistic and Metadata
    language: str = "en"
    query_language: str = "en"
    evidence_language: str = "en"
    cross_language: bool = False

    source_type: Optional[str] = None
    provenance: Optional[ProvenanceMetadata] = None

    split: SplitName = SplitName.train
    cluster_id: str

    # Human Review Metadata
    human_reviewed: bool = False
    reviewer_id: Optional[str] = None
    review_confidence: Optional[AnnotationConfidence] = None
    review_notes: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def sync_before(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # 1. Sync positive
            pos = data.get("positive")
            if pos:
                if isinstance(pos, dict):
                    pos_id = pos.get("evidence_id")
                    pos_txt = pos.get("evidence_text") or pos.get("text")
                    pos_rel = pos.get("relation")
                    if not data.get("positive_evidence_id") and pos_id:
                        data["positive_evidence_id"] = pos_id
                    if not data.get("positive_evidence_text") and pos_txt:
                        data["positive_evidence_text"] = pos_txt
                    if "positive_relation" not in data and pos_rel:
                        data["positive_relation"] = pos_rel
                elif hasattr(pos, "evidence_id"):
                    if not data.get("positive_evidence_id"):
                        data["positive_evidence_id"] = pos.evidence_id
                    if not data.get("positive_evidence_text"):
                        data["positive_evidence_text"] = pos.evidence_text
                    if "positive_relation" not in data:
                        data["positive_relation"] = pos.relation

            # 2. Sync negative
            negs = data.get("negatives")
            if negs and isinstance(negs, list) and len(negs) > 0:
                first_neg = negs[0]
                if isinstance(first_neg, dict):
                    neg_id = first_neg.get("evidence_id")
                    neg_txt = first_neg.get("evidence_text") or first_neg.get("text")
                    neg_type = first_neg.get("negative_type")
                    if not data.get("negative_evidence_id") and neg_id:
                        data["negative_evidence_id"] = neg_id
                    if not data.get("negative_evidence_text") and neg_txt:
                        data["negative_evidence_text"] = neg_txt
                    if "negative_type" not in data and neg_type:
                        data["negative_type"] = neg_type
                elif hasattr(first_neg, "evidence_id"):
                    if not data.get("negative_evidence_id"):
                        data["negative_evidence_id"] = first_neg.evidence_id
                    if not data.get("negative_evidence_text"):
                        data["negative_evidence_text"] = first_neg.evidence_text
                    if "negative_type" not in data:
                        data["negative_type"] = first_neg.negative_type

            # 3. Default provenance if missing
            if not data.get("provenance"):
                data["provenance"] = ProvenanceMetadata(
                    source_type=ProvenanceSourceType.PUBLIC_DATASET,
                    source_name="TrustLens Evidence Grounding Registry",
                )
        return data

    @model_validator(mode="after")
    def validate_relations(self):
        # 1. Positive evidence must be SUPPORTS or CONTRADICTS
        if self.positive_relation not in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
            raise ValueError(
                f"Positive relation must be SUPPORTS or CONTRADICTS, got '{self.positive_relation}'"
            )

        # 2. Sync nested positive record
        if not self.positive:
            self.positive = PositiveEvidenceRecord(
                evidence_id=self.positive_evidence_id,
                evidence_text=self.positive_evidence_text,
                relation=self.positive_relation,
                provenance=self.provenance,
            )

        # 3. Ensure primary negative is included in negatives list
        if not self.negatives and self.negative_evidence_id:
            self.negatives.append(
                NegativeEvidenceRecord(
                    evidence_id=self.negative_evidence_id,
                    evidence_text=self.negative_evidence_text,
                    negative_type=self.negative_type,
                    evidence_relation=EvidenceRelationLabel.NEUTRAL,
                    provenance=self.provenance,
                )
            )

        # 4. Critical rule: Negative documents must NOT support or contradict the claim
        for neg in self.negatives:
            if neg.evidence_relation in (EvidenceRelationLabel.SUPPORTS, EvidenceRelationLabel.CONTRADICTS):
                raise ValueError(
                    f"Negative evidence {neg.evidence_id} has forbidden relation '{neg.evidence_relation}'. "
                    f"Negatives must never support or contradict the claim."
                )
        return self


class RetrievalPair(BaseModel):
    """
    Pairwise retrieval/reranker training example for Cross-Encoder or Bi-Encoder ranking.
    Label: 1 for relevant/positive evidence, 0 for negative evidence.
    """
    pair_id: str = Field(default_factory=lambda: f"pair_{uuid.uuid4().hex[:10]}")
    retrieval_id: str
    query_id: str
    query: str
    query_type: QueryType = QueryType.VERIFICATION

    evidence_id: str
    document_text: str
    label: int  # 1 = relevant, 0 = negative

    evidence_relation: Optional[EvidenceRelationLabel] = None
    negative_type: Optional[NegativeType] = None

    query_language: str = "en"
    evidence_language: str = "en"
    cross_language: bool = False

    split: SplitName = SplitName.train
    cluster_id: str

    provenance: Optional[ProvenanceMetadata] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RetrievalEvaluationItem(BaseModel):
    """
    Standard test evaluation item with query, ground truth relevant IDs, and candidate pool.
    Supports Recall@K, Precision@K, MRR, and nDCG@K evaluation without data leakage.
    """
    query_id: str
    claim_id: str
    query: str
    query_type: QueryType
    query_language: str
    relevant_evidence_ids: List[str]
    candidate_evidence_pool: List[Dict[str, Any]]
    split: SplitName
    cluster_id: str
