import json
from pathlib import Path
from typing import List, Dict, Any

from app.claims.schemas import SourceSpan
from app.input.schemas import AcquisitionMethod, AuthorInfo, Platform, PostContent, PostMedia, PostType
from app.training.schemas import (
    AnnotationConfidence,
    AnnotationMetadata,
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    ImageMetadata,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    QueryType,
    ReviewStatus,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingAtomicClaim,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingQuery,
    TrainingRisk,
)


def get_synthetic_examples() -> Dict[str, Any]:
    common_provenance = ProvenanceMetadata(
        source_type=ProvenanceSourceType.SYNTHETIC,
        source_name="TrustLens Synthetic Verification Benchmark",
        source_id="synthetic_bench_v1",
        collection_method="rule_synthesized",
        license="CC-BY-4.0",
    )

    common_split = SplitMetadata(
        split=SplitName.validation,
        source_group_id="synth_group_01",
        campaign_group_id="synth_campaign_01",
    )

    common_annotation = AnnotationMetadata(
        annotator_id="expert_reviewer_01",
        confidence=AnnotationConfidence.HIGH,
        notes="Synthetic schema validation benchmark record",
    )

    # 1. English Financial Example
    post_1 = TrainingPost(
        post_id="ex_post_001_en_fin",
        platform=Platform.text,
        post_type=PostType.text,
        content=PostContent(text="Guaranteed 50% monthly returns on crypto deposit! Deposit ₹10,000 today and receive ₹15,000 next month."),
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_1 = TrainingClaim(
        claim_id="ex_claim_001",
        post_id="ex_post_001_en_fin",
        claim_text="Guaranteed 50% monthly returns on crypto deposit",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="en",
        script="Latin",
        provenance=common_provenance,
        split_info=common_split,
        annotation=common_annotation,
        is_example=True,
    )
    atomic_1a = TrainingAtomicClaim(
        atomic_claim_id="ex_atomic_001a",
        parent_claim_id="ex_claim_001",
        original_text="Deposit ₹10,000 today",
        normalized_text="Investor deposits ₹10,000 today",
        claim_type="FINANCIAL",
        subject="Investor",
        predicate="deposits",
        value="10,000",
        currency="INR",
        temporal_context={"relative": "today"},
        provenance=common_provenance,
        is_example=True,
    )
    atomic_1b = TrainingAtomicClaim(
        atomic_claim_id="ex_atomic_001b",
        parent_claim_id="ex_claim_001",
        original_text="receive ₹15,000 next month",
        normalized_text="Investor receives ₹15,000 next month",
        claim_type="FINANCIAL",
        subject="Investor",
        predicate="receives",
        value="15,000",
        currency="INR",
        temporal_context={"relative": "next month"},
        relationships=[{"target": "ex_atomic_001a", "type": "promised_outcome"}],
        provenance=common_provenance,
        is_example=True,
    )
    evidence_1 = TrainingEvidence(
        evidence_id="ex_ev_001",
        claim_id="ex_claim_001",
        evidence_text="Reserve Bank of India regulatory bulletin: crypto entities offering guaranteed fixed returns operate without authorization.",
        source_url="https://example.gov.in/rbi/crypto_warning.pdf",
        source_type="REGULATORY_BULLETIN",
        source_title="RBI Advisory on Unauthorized Investment Schemes",
        retrieval_method="dense_retrieval",
        language="en",
        relation_label=EvidenceRelationLabel.CONTRADICTS,
        provenance=common_provenance,
        annotation=common_annotation,
        is_example=True,
    )
    risk_1 = TrainingRisk(
        risk_id="ex_risk_001",
        post_id="ex_post_001_en_fin",
        claim_ids=["ex_claim_001"],
        risk_level=RiskLevel.HIGH,
        risk_factors=["guaranteed_high_returns", "unauthorized_entity", "regulatory_contradiction"],
        evidence_summary="Contradicted by official regulatory warnings against guaranteed crypto returns.",
        provenance=common_provenance,
        annotation=common_annotation,
        is_example=True,
    )

    # 2. Hindi Financial Example
    post_2 = TrainingPost(
        post_id="ex_post_002_hi_fin",
        platform=Platform.text,
        post_type=PostType.text,
        content=PostContent(text="प्रति माह ₹50,000 की गारंटीड कमाई! अभी ₹500 रजिस्ट्रेशन शुल्क जमा करें।"),
        language_info=LanguageMetadata(primary="hi", languages=["hi"], script=["Devanagari"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_2 = TrainingClaim(
        claim_id="ex_claim_002",
        post_id="ex_post_002_hi_fin",
        claim_text="प्रति माह ₹50,000 की गारंटीड कमाई के लिए ₹500 रजिस्ट्रेशन शुल्क",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="hi",
        script="Devanagari",
        provenance=common_provenance,
        is_example=True,
    )

    # 3. Tamil Financial Example
    post_3 = TrainingPost(
        post_id="ex_post_003_ta_fin",
        platform=Platform.text,
        post_type=PostType.text,
        content=PostContent(text="மாதம் ₹50,000 சம்பாதிக்கலாம். தொடக்க கட்டணம் ₹500 செலுத்தவும்."),
        language_info=LanguageMetadata(primary="ta", languages=["ta"], script=["Tamil"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_3 = TrainingClaim(
        claim_id="ex_claim_003",
        post_id="ex_post_003_ta_fin",
        claim_text="மாதம் ₹50,000 சம்பாதிக்கலாம்",
        claim_type=ClaimType.JOB,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="ta",
        script="Tamil",
        provenance=common_provenance,
        is_example=True,
    )

    # 4. Tanglish Example
    post_4 = TrainingPost(
        post_id="ex_post_004_tanglish",
        platform=Platform.text,
        post_type=PostType.text,
        content=PostContent(text="Intha scheme la invest panna double returns kedaikum. Fast payout guaranteed."),
        language_info=LanguageMetadata(primary="ta", languages=["ta", "en"], script=["Latin"], code_mixed=True, transliterated=True),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_4 = TrainingClaim(
        claim_id="ex_claim_004",
        post_id="ex_post_004_tanglish",
        claim_text="Intha scheme la invest panna double returns kedaikum",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="ta",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )

    # 5. Conditional Claim Example
    post_5 = TrainingPost(
        post_id="ex_post_005_conditional",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="If you transfer ₹5,000 within 24 hours, your account unlock code will be sent immediately."),
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_5 = TrainingClaim(
        claim_id="ex_claim_005",
        post_id="ex_post_005_conditional",
        claim_text="If you transfer ₹5,000 within 24 hours, your account unlock code will be sent immediately",
        claim_type=ClaimType.CREDENTIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )
    atomic_5a = TrainingAtomicClaim(
        atomic_claim_id="ex_atomic_005a",
        parent_claim_id="ex_claim_005",
        original_text="transfer ₹5,000 within 24 hours",
        normalized_text="User transfers ₹5,000 within 24 hours",
        claim_type="PAYMENT",
        subject="User",
        predicate="transfers",
        value="5,000",
        currency="INR",
        temporal_context={"window": "24 hours"},
        provenance=common_provenance,
        is_example=True,
    )
    atomic_5b = TrainingAtomicClaim(
        atomic_claim_id="ex_atomic_005b",
        parent_claim_id="ex_claim_005",
        original_text="account unlock code will be sent immediately",
        normalized_text="Support sends account unlock code immediately",
        claim_type="CREDENTIAL",
        subject="Support",
        predicate="sends",
        object="unlock code",
        conditions={"prerequisite": "transfer ₹5,000 within 24 hours"},
        relationships=[{"target": "ex_atomic_005a", "type": "promised_outcome"}],
        provenance=common_provenance,
        is_example=True,
    )

    # 6. Negated Claim Example
    post_6 = TrainingPost(
        post_id="ex_post_006_negated",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="TrustLens disclaimer: This community group is NOT affiliated with the Reserve Bank of India."),
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_6 = TrainingClaim(
        claim_id="ex_claim_006",
        post_id="ex_post_006_negated",
        claim_text="This community group is NOT affiliated with the Reserve Bank of India",
        claim_type=ClaimType.OTHER,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )
    atomic_6 = TrainingAtomicClaim(
        atomic_claim_id="ex_atomic_006",
        parent_claim_id="ex_claim_006",
        original_text="This community group is NOT affiliated with the Reserve Bank of India",
        normalized_text="Community group is not affiliated with Reserve Bank of India",
        claim_type="OTHER",
        subject="Community group",
        predicate="affiliated_with",
        object="Reserve Bank of India",
        polarity="NEGATIVE",
        negated=True,
        provenance=common_provenance,
        is_example=True,
    )

    # 7. Attributed Claim Example
    post_7 = TrainingPost(
        post_id="ex_post_007_attributed",
        platform=Platform.telegram,
        post_type=PostType.text,
        content=PostContent(text="According to Telegram administrator @CryptoGod88, all withdrawals will be processed in 5 minutes."),
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_7 = TrainingClaim(
        claim_id="ex_claim_007",
        post_id="ex_post_007_attributed",
        claim_text="All withdrawals will be processed in 5 minutes",
        claim_type=ClaimType.FINANCIAL,
        detection_label=ClaimDetectionLabel.CLAIM,
        attribution={"speaker": "@CryptoGod88", "channel": "Telegram"},
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )

    # 8. Insufficient Evidence Example
    post_8 = TrainingPost(
        post_id="ex_post_008_insufficient",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="Local potter Ramesh has started selling terracotta tea cups near Madurai temple."),
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_8 = TrainingClaim(
        claim_id="ex_claim_008",
        post_id="ex_post_008_insufficient",
        claim_text="Ramesh started selling terracotta tea cups near Madurai temple",
        claim_type=ClaimType.OTHER,
        detection_label=ClaimDetectionLabel.CLAIM,
        check_worthiness=0.2,
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )
    evidence_8 = TrainingEvidence(
        evidence_id="ex_ev_008",
        claim_id="ex_claim_008",
        evidence_text="No local or commercial records found regarding Madurai street artisan Ramesh.",
        source_url=None,
        source_type="RETRIEVAL_EMPTY",
        source_title="Search Index Report",
        retrieval_method="web_search",
        language="en",
        relation_label=EvidenceRelationLabel.INSUFFICIENT,
        provenance=common_provenance,
        is_example=True,
    )
    risk_8 = TrainingRisk(
        risk_id="ex_risk_008",
        post_id="ex_post_008_insufficient",
        claim_ids=["ex_claim_008"],
        risk_level=RiskLevel.INSUFFICIENT,
        risk_factors=[],
        evidence_summary="Zero external verification sources found; statement represents benign local claim with insufficient risk signals.",
        provenance=common_provenance,
        is_example=True,
    )

    # 9. Multimodal Example (Screenshot)
    post_9 = TrainingPost(
        post_id="ex_post_009_multimodal",
        platform=Platform.screenshot,
        post_type=PostType.image_text,
        content=PostContent(text="See payment confirmation below"),
        media=PostMedia(images=["data/raw/screenshots/fake_receipt_01.png"]),
        images_meta=[
            ImageMetadata(
                image_id="img_receipt_01",
                image_path="data/raw/screenshots/fake_receipt_01.png",
                image_hash="sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                ocr_text="Payment Successful to John Doe ₹25,000 Ref: TXN998811",
                ocr_language="en",
                ocr_confidence=0.95,
                ocr_engine="easyocr_en",
                image_quality="HIGH",
            )
        ],
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_9 = TrainingClaim(
        claim_id="ex_claim_009",
        post_id="ex_post_009_multimodal",
        claim_text="Payment Successful to John Doe ₹25,000 Ref: TXN998811",
        claim_type=ClaimType.PAYMENT,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )

    # 10. Multiple-Claim Example (Job + Payment)
    post_10 = TrainingPost(
        post_id="ex_post_010_multiple_claims",
        platform=Platform.generic,
        post_type=PostType.text,
        content=PostContent(text="Urgent Hiring: Data Entry operators! Earn ₹35,000 per month. Free company laptop provided upon ₹1,000 refundable security deposit."),
        language_info=LanguageMetadata(primary="en", languages=["en"], script=["Latin"]),
        provenance=common_provenance,
        split_info=common_split,
        is_example=True,
    )
    claim_10a = TrainingClaim(
        claim_id="ex_claim_010a",
        post_id="ex_post_010_multiple_claims",
        claim_text="Earn ₹35,000 per month as Data Entry operator",
        claim_type=ClaimType.JOB,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )
    claim_10b = TrainingClaim(
        claim_id="ex_claim_010b",
        post_id="ex_post_010_multiple_claims",
        claim_text="Free company laptop provided upon ₹1,000 refundable security deposit",
        claim_type=ClaimType.PAYMENT,
        detection_label=ClaimDetectionLabel.CLAIM,
        language="en",
        script="Latin",
        provenance=common_provenance,
        is_example=True,
    )

    return {
        "posts": [post_1, post_2, post_3, post_4, post_5, post_6, post_7, post_8, post_9, post_10],
        "claims": [claim_1, claim_2, claim_3, claim_4, claim_5, claim_6, claim_7, claim_8, claim_9, claim_10a, claim_10b],
        "atomic_claims": [atomic_1a, atomic_1b, atomic_5a, atomic_5b, atomic_6],
        "evidence": [evidence_1, evidence_8],
        "risks": [risk_1, risk_8],
    }


def export_examples_to_json(out_path: str = "data/schema/examples.json"):
    data = get_synthetic_examples()
    serialized = {
        key: [item.model_dump() for item in items]
        for key, items in data.items()
    }
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)
    return path
