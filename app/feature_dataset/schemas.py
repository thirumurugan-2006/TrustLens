"""
TrustLens Structured Feature Dataset Schemas (Phase 6D).
Defines the feature registry, feature groups, data types, and canonical FeatureRecord unit.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.training.schemas import (
    ProvenanceMetadata,
    RiskLevel,
    SplitName,
)


class FeatureGroup(str, Enum):
    """Controlled taxonomy of structured feature groups."""
    TEXT = "TEXT"
    LANGUAGE = "LANGUAGE"
    CLAIM = "CLAIM"
    EVIDENCE = "EVIDENCE"
    RETRIEVAL = "RETRIEVAL"
    CONTRADICTION = "CONTRADICTION"
    IMAGE = "IMAGE"
    SOURCE_DOMAIN = "SOURCE_DOMAIN"
    GRAPH = "GRAPH"
    SCAM_SIGNAL = "SCAM_SIGNAL"


class FeatureDType(str, Enum):
    """Data types for feature columns in the feature matrix."""
    FLOAT = "float"
    INT = "int"
    BOOLEAN = "bool"
    CATEGORICAL = "category"


class FeatureDefinition(BaseModel):
    """
    Formal specification of a single observable feature.
    Guarantees documentation of source, missingness policy, and leakage risk.
    """
    name: str
    group: FeatureGroup
    dtype: FeatureDType
    description: str
    source: str
    missing_policy: str
    leakage_risk: str = "none"


class FeatureRecord(BaseModel):
    """
    Canonical training row: ONE POST / ONE DECISION INSTANCE.
    Preserves full lineage, cluster ID, split, and four-class ground truth risk label.
    """
    feature_id: str
    post_id: str
    cluster_id: str
    split: SplitName
    risk_label: RiskLevel

    # Observable feature vector
    features: Dict[str, Any]

    # Lineage and metadata
    provenance: Optional[ProvenanceMetadata] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureRegistry:
    """
    Master registry of all observable features in TrustLens Phase 6D.
    Every feature row column must be declared here.
    """

    DEFINITIONS: List[FeatureDefinition] = [
        # Group A: Text Features
        FeatureDefinition(
            name="text_length",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Total character length of raw post text",
            source="post.content.text",
            missing_policy="zero_if_empty",
        ),
        FeatureDefinition(
            name="word_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Number of whitespace-delimited tokens in post text",
            source="post.content.text",
            missing_policy="zero_if_empty",
        ),
        FeatureDefinition(
            name="sentence_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Number of sentences derived by sentence boundary punctuation",
            source="post.content.text",
            missing_policy="one_if_unsegmented",
        ),
        FeatureDefinition(
            name="uppercase_ratio",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.FLOAT,
            description="Proportion of uppercase alphabet characters to total alphabetic characters",
            source="post.content.text",
            missing_policy="zero_if_no_alpha",
        ),
        FeatureDefinition(
            name="digit_ratio",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.FLOAT,
            description="Proportion of numeric digit characters to total length",
            source="post.content.text",
            missing_policy="zero_if_empty",
        ),
        FeatureDefinition(
            name="punctuation_ratio",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.FLOAT,
            description="Proportion of punctuation characters to total length",
            source="post.content.text",
            missing_policy="zero_if_empty",
        ),
        FeatureDefinition(
            name="url_count_in_text",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Count of embedded URLs and http/https hyperlinks in text",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="phone_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Count of observable phone numbers or country dial codes (+91)",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="email_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Count of email addresses formatted with @ and valid domains",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="currency_symbol_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Count of currency symbols (₹, $, Rs, INR, USD)",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="exclamation_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Count of exclamation marks (!) in post text",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="question_count",
            group=FeatureGroup.TEXT,
            dtype=FeatureDType.INT,
            description="Count of question marks (?) in post text",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),

        # Group B: Language Features
        FeatureDefinition(
            name="language",
            group=FeatureGroup.LANGUAGE,
            dtype=FeatureDType.CATEGORICAL,
            description="Primary linguistic code (en, ta, hi, ta-en, hi-en)",
            source="post.language_info.primary",
            missing_policy="en_default",
        ),
        FeatureDefinition(
            name="language_id",
            group=FeatureGroup.LANGUAGE,
            dtype=FeatureDType.INT,
            description="Integer mapping of primary language",
            source="post.language_info.primary",
            missing_policy="zero_default",
        ),
        FeatureDefinition(
            name="script",
            group=FeatureGroup.LANGUAGE,
            dtype=FeatureDType.CATEGORICAL,
            description="Primary writing script (Latin, Tamil, Devanagari)",
            source="post.language_info.script",
            missing_policy="latin_default",
        ),
        FeatureDefinition(
            name="is_code_mixed",
            group=FeatureGroup.LANGUAGE,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating multilingual code-mixing (e.g., Hinglish, Tanglish)",
            source="post.language_info.code_mixed",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="is_transliterated",
            group=FeatureGroup.LANGUAGE,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating transliterated text into Latin script",
            source="post.language_info.transliterated",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="language_confidence",
            group=FeatureGroup.LANGUAGE,
            dtype=FeatureDType.FLOAT,
            description="Confidence score of language identification",
            source="post.language_info.metadata",
            missing_policy="one_if_unspecified",
        ),

        # Group C: Claim Features
        FeatureDefinition(
            name="claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Total count of claims extracted from post",
            source="post.claims",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="atomic_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Total count of atomic claim predicates",
            source="claims.atomic_claims",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="verifiable_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Count of extracted claims marked verifiable",
            source="claims.check_worthiness",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="non_verifiable_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Count of extracted claims marked unverifiable or opinion",
            source="claims.check_worthiness",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="claim_decomposition_confidence",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.FLOAT,
            description="Mean check-worthiness / decomposition confidence across claims",
            source="claims.check_worthiness",
            missing_policy="zero_if_no_claims",
        ),
        FeatureDefinition(
            name="has_financial_claim",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one FINANCIAL claim",
            source="claims.claim_type",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="has_job_claim",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one JOB / employment claim",
            source="claims.claim_type",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="has_giveaway_claim",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one GIVEAWAY / lottery claim",
            source="claims.claim_type",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="has_credential_claim",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one CREDENTIAL solicitation claim",
            source="claims.claim_type",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="has_shopping_claim",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one SHOPPING e-commerce claim",
            source="claims.claim_type",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="conditional_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Count of claims containing conditional clauses or prerequisite commitments",
            source="claims.atomic_claims.conditions",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="negated_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Count of claims containing explicit negation operators",
            source="claims.atomic_claims.negated",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="numeric_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Count of claims specifying numerical return percentages or amounts",
            source="claims.atomic_claims.value",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="temporal_claim_count",
            group=FeatureGroup.CLAIM,
            dtype=FeatureDType.INT,
            description="Count of claims specifying time periods (daily, weekly, monthly)",
            source="claims.atomic_claims.temporal_context",
            missing_policy="zero_if_none",
        ),

        # Group D: Evidence Features
        FeatureDefinition(
            name="evidence_count",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.INT,
            description="Total count of grounded evidence items retrieved for claims",
            source="evidence",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="has_evidence",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating if post has at least one grounded evidence item",
            source="evidence",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="claims_with_evidence",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.INT,
            description="Count of claims supported by at least one evidence item",
            source="evidence.claim_id",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="claims_without_evidence",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.INT,
            description="Count of claims having zero retrieved evidence",
            source="claims vs evidence",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="evidence_coverage_ratio",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.FLOAT,
            description="Proportion of claims that have grounded evidence items",
            source="evidence.claim_id / claim_count",
            missing_policy="zero_if_no_claims",
        ),
        FeatureDefinition(
            name="evidence_source_count",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.INT,
            description="Count of distinct evidence sources cited",
            source="evidence.source_type",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="independent_source_count",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.INT,
            description="Count of distinct domains among evidence source URLs",
            source="evidence.source_url",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="source_diversity",
            group=FeatureGroup.EVIDENCE,
            dtype=FeatureDType.FLOAT,
            description="Ratio of independent sources to total evidence items",
            source="independent_sources / evidence_count",
            missing_policy="zero_if_no_evidence",
        ),

        # Group E: Retrieval Features
        FeatureDefinition(
            name="retrieved_evidence_count",
            group=FeatureGroup.RETRIEVAL,
            dtype=FeatureDType.INT,
            description="Total count of retrieved candidates in retrieval pool",
            source="retrieval_dataset.eval_pool",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="relevant_evidence_count",
            group=FeatureGroup.RETRIEVAL,
            dtype=FeatureDType.INT,
            description="Count of relevant evidence items in retrieval pool",
            source="retrieval_dataset.positive",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="supporting_retrieval_count",
            group=FeatureGroup.RETRIEVAL,
            dtype=FeatureDType.INT,
            description="Count of supporting evidence retrieved",
            source="retrieval_dataset",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="contradicting_retrieval_count",
            group=FeatureGroup.RETRIEVAL,
            dtype=FeatureDType.INT,
            description="Count of contradicting evidence retrieved",
            source="retrieval_dataset",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="top_retrieval_score",
            group=FeatureGroup.RETRIEVAL,
            dtype=FeatureDType.FLOAT,
            description="Candidate similarity score or -1.0 if dense scores unassigned",
            source="retrieval_dataset.candidate_similarity",
            missing_policy="negative_one_if_missing",
        ),
        FeatureDefinition(
            name="retrieval_score_available",
            group=FeatureGroup.RETRIEVAL,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating if dense retrieval score is populated",
            source="retrieval_dataset",
            missing_policy="false_default",
        ),

        # Group F: Contradiction & Stance Features
        FeatureDefinition(
            name="support_count",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.INT,
            description="Count of evidence items with relation SUPPORTS",
            source="evidence.relation_label",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="contradict_count",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.INT,
            description="Count of evidence items with relation CONTRADICTS",
            source="evidence.relation_label",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="neutral_count",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.INT,
            description="Count of evidence items with relation NEUTRAL",
            source="evidence.relation_label",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="insufficient_count",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.INT,
            description="Count of evidence items with relation INSUFFICIENT",
            source="evidence.relation_label",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="support_ratio",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.FLOAT,
            description="Ratio of SUPPORTS evidence to total evidence items",
            source="support_count / evidence_count",
            missing_policy="zero_if_no_evidence",
        ),
        FeatureDefinition(
            name="contradiction_ratio",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.FLOAT,
            description="Ratio of CONTRADICTS evidence to total evidence items",
            source="contradict_count / evidence_count",
            missing_policy="zero_if_no_evidence",
        ),
        FeatureDefinition(
            name="neutral_ratio",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.FLOAT,
            description="Ratio of NEUTRAL evidence to total evidence items",
            source="neutral_count / evidence_count",
            missing_policy="zero_if_no_evidence",
        ),
        FeatureDefinition(
            name="insufficient_ratio",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.FLOAT,
            description="Ratio of INSUFFICIENT evidence to total evidence items",
            source="insufficient_count / evidence_count",
            missing_policy="zero_if_no_evidence",
        ),
        FeatureDefinition(
            name="support_to_contradiction_ratio",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.FLOAT,
            description="Ratio of support_count to (contradict_count + 1)",
            source="support_count / (contradict_count + 1)",
            missing_policy="zero_default",
        ),
        FeatureDefinition(
            name="contradiction_presence",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one CONTRADICTS evidence item",
            source="contradict_count > 0",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="support_presence",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of at least one SUPPORTS evidence item",
            source="support_count > 0",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="evidence_relation_entropy",
            group=FeatureGroup.CONTRADICTION,
            dtype=FeatureDType.FLOAT,
            description="Shannon entropy over 4 relation classes (SUPPORTS, CONTRADICTS, NEUTRAL, INSUFFICIENT)",
            source="relation distribution",
            missing_policy="zero_if_homogeneous",
        ),

        # Group G: Image & OCR Features
        FeatureDefinition(
            name="image_count",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.INT,
            description="Number of attached media images",
            source="post.media.images",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="has_image",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of image attachments",
            source="post.media.images",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="ocr_available",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating whether OCR text was successfully extracted",
            source="post.images_meta.ocr_text",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="ocr_confidence",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.FLOAT,
            description="Mean OCR engine confidence or -1.0 if unavailable",
            source="post.images_meta.ocr_confidence",
            missing_policy="negative_one_if_missing",
        ),
        FeatureDefinition(
            name="ocr_text_length",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.INT,
            description="Character length of extracted OCR text",
            source="post.images_meta.ocr_text",
            missing_policy="zero_if_missing",
        ),
        FeatureDefinition(
            name="image_reuse_count",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.INT,
            description="Count of previous cluster reuse matches for image hash",
            source="post.images_meta",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="phash_match_count",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.INT,
            description="Count of perceptual hash matches across image registry",
            source="post.images_meta.phash",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="ocr_unreliable",
            group=FeatureGroup.IMAGE,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating OCR text available but confidence is below 0.50",
            source="ocr_confidence < 0.5 and ocr_available",
            missing_policy="false_default",
        ),

        # Group H: Source & Domain Features
        FeatureDefinition(
            name="source_count",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.INT,
            description="Count of cited evidence sources",
            source="evidence.source_type",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="unique_source_count",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.INT,
            description="Count of unique source IDs cited in evidence",
            source="evidence.provenance.source_id",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="official_source_present",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating citation of official regulatory source",
            source="evidence.source_type == REGULATORY_ALERT",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="domain_count",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.INT,
            description="Count of distinct domains referenced in post or evidence URLs",
            source="urls",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="unique_domain_count",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.INT,
            description="Count of unique internet domains",
            source="urls",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="official_domain_match",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating evidence URL matches official government/regulatory domain",
            source="evidence.source_url",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="domain_age_days",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.FLOAT,
            description="Domain registration age in days or -1.0 if unqueried/missing",
            source="domain intelligence",
            missing_policy="negative_one_if_missing",
        ),
        FeatureDefinition(
            name="domain_age_available",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating if domain age is legitimately available",
            source="domain intelligence",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="shortened_url_count",
            group=FeatureGroup.SOURCE_DOMAIN,
            dtype=FeatureDType.INT,
            description="Count of shortened URLs (bit.ly, tinyurl, t.me) in post text",
            source="post.content.text",
            missing_policy="zero_if_none",
        ),

        # Group I: Graph-Derived Features
        FeatureDefinition(
            name="graph_node_count",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Total nodes in post's heterogeneous evidence graph",
            source="graph_dataset.nodes",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_edge_count",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Total edges in post's heterogeneous evidence graph",
            source="graph_dataset.edges",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_claim_nodes",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of CLAIM nodes in heterogeneous graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_evidence_nodes",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of EVIDENCE nodes in heterogeneous graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_url_nodes",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of URL nodes in heterogeneous graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_account_nodes",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of ACCOUNT nodes in heterogeneous graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_support_edges",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of SUPPORTS directed edges in graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_contradict_edges",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of CONTRADICTS directed edges in graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_neutral_edges",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of NEUTRAL_TO directed edges in graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_insufficient_edges",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.INT,
            description="Count of INSUFFICIENT_FOR directed edges in graph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),
        FeatureDefinition(
            name="graph_degree_mean",
            group=FeatureGroup.GRAPH,
            dtype=FeatureDType.FLOAT,
            description="Mean degree across heterogeneous nodes in subgraph",
            source="graph_dataset",
            missing_policy="zero_if_no_graph",
        ),

        # Group J: Scam Signal Features
        FeatureDefinition(
            name="guarantee_claim_present",
            group=FeatureGroup.SCAM_SIGNAL,
            dtype=FeatureDType.BOOLEAN,
            description="Observable claim asserting guaranteed profit or zero risk",
            source="semantic analysis",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="urgency_signal_present",
            group=FeatureGroup.SCAM_SIGNAL,
            dtype=FeatureDType.BOOLEAN,
            description="Observable semantic urgency cue (e.g., today only, hurry, limited slots)",
            source="semantic analysis",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="upfront_payment_present",
            group=FeatureGroup.SCAM_SIGNAL,
            dtype=FeatureDType.BOOLEAN,
            description="Observable advance fee, registration deposit, or processing fee request",
            source="semantic analysis",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="financial_claim_count",
            group=FeatureGroup.SCAM_SIGNAL,
            dtype=FeatureDType.INT,
            description="Count of claims specifically making financial assertions",
            source="claims.claim_type == FINANCIAL",
            missing_policy="zero_if_none",
        ),
        FeatureDefinition(
            name="currency_present",
            group=FeatureGroup.SCAM_SIGNAL,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of monetary amount or currency denomination",
            source="semantic analysis",
            missing_policy="false_default",
        ),
        FeatureDefinition(
            name="temporal_deadline_present",
            group=FeatureGroup.SCAM_SIGNAL,
            dtype=FeatureDType.BOOLEAN,
            description="Flag indicating presence of time-bound deadlines or promised turnarounds",
            source="semantic analysis",
            missing_policy="false_default",
        ),
    ]

    @classmethod
    def get_feature_names(cls) -> List[str]:
        return [f.name for f in cls.DEFINITIONS]

    @classmethod
    def get_feature_map(cls) -> Dict[str, FeatureDefinition]:
        return {f.name: f for f in cls.DEFINITIONS}
