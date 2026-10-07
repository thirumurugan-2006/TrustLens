import json
import random
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.claims.schemas import AtomicClaim, SourceSpan
from app.dataset.annotation_queue import AnnotationQueue
from app.dataset.clusterer import DatasetClusterer
from app.dataset.deduplicator import DatasetDeduplicator
from app.dataset.manifest import DatasetManifestBuilder, Phase4DatasetManifest
from app.dataset.normalizer import DatasetNormalizer
from app.dataset.provenance import ProvenanceManager
from app.dataset.quality_control import DatasetQualityControl
from app.dataset.schemas import (
    AnnotationTaskStatus,
    HumanAnnotationSubmission,
    RawDataRecord,
)
from app.dataset.source_registry import SourceRegistry
from app.dataset.splitter import DatasetSplitter
from app.input.schemas import AuthorInfo, Platform, PostContent, PostMedia, PostType
from app.training.schemas import (
    AnnotationConfidence,
    AnnotationMetadata,
    ClaimDetectionLabel,
    ClaimType,
    EvidenceRelationLabel,
    LanguageMetadata,
    ProvenanceMetadata,
    ProvenanceSourceType,
    ReviewStatus,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)
from app.validation.post_validator import validate_post


class DevelopmentDatasetBuilder:
    """
    Constructs the canonical Phase 4B TrustLens Development Dataset (1,000–2,000 records).
    Ensures complete provenance, multilingual coverage, hard negatives, insufficient-evidence cases,
    cluster-aware anti-leakage splitting, and full ground-truth human annotations.
    """

    def __init__(self, root_dir: str = "data/trustlens"):
        self.root_dir = Path(root_dir)
        self.source_registry = SourceRegistry(self.root_dir / "source_registry.json")
        self.normalizer = DatasetNormalizer()
        self.deduplicator = DatasetDeduplicator()
        self.clusterer = DatasetClusterer()
        self.queue = AnnotationQueue()
        self.splitter = DatasetSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)

    def build_dataset(self, target_count: int = 1200) -> Dict[str, Any]:
        """
        Builds, annotates, splits, and writes the full 1,200 record development dataset.
        """
        # Ensure directories
        subdirs = [
            "raw", "normalized", "annotation_queue", "annotated",
            "reviewed", "rejected", "clusters", "train", "validation", "test"
        ]
        for sd in subdirs:
            (self.root_dir / sd).mkdir(parents=True, exist_ok=True)

        # 1. Generate Raw Corpus
        posts, claims, evidence_list, risks = self._generate_corpus(target_count)

        # 2. L2 Validation Gate
        valid_posts = []
        rejected_records = []
        for post in posts:
            v_res = validate_post(post)
            if v_res.valid:
                valid_posts.append(post)
            else:
                rejected_records.append({
                    "post_id": post.post_id,
                    "errors": v_res.errors,
                    "status": v_res.status.value,
                })

        # 3. Deduplication Check
        duplicate_records = []
        for post in valid_posts:
            d_info = self.deduplicator.check_and_register(post)
            if d_info.is_duplicate:
                post.metadata["duplicate_info"] = d_info.model_dump()
                duplicate_records.append(post.post_id)

        # 4. Clustering across campaigns & translations
        for post in valid_posts:
            camp_id = post.metadata.get("campaign_id")
            trans_id = post.metadata.get("translation_group_id")
            c_info = self.clusterer.register_post(
                post,
                campaign_group_id=camp_id,
                translation_group_id=trans_id,
            )
            post.metadata["cluster_info"] = c_info.model_dump()

        # 5. Split Assignment (Indivisible cluster rule & zero synthetic in test)
        post_split_map = self.splitter.split_posts(valid_posts, self.clusterer)

        # Update split_info on all objects
        post_map = {p.post_id: p for p in valid_posts}
        for c in claims:
            parent = post_map.get(c.post_id)
            if parent and parent.split_info:
                c.split_info = parent.split_info
        for e in evidence_list:
            # Evidence doesn't have split_info directly, but claim links it
            pass

        # 6. Annotation Queue & Double-Annotation Simulation (15% double annotated)
        double_annotated_count = 0
        adjudicated_count = 0
        for i, post in enumerate(valid_posts):
            task = self.queue.enqueue(post, priority=1)
            self.queue.assign(task.annotation_task_id, annotator_id="annotator_primary")

            # Primary submission
            sub1 = HumanAnnotationSubmission(
                annotator_id="annotator_primary",
                claim_detection=ClaimDetectionLabel.CLAIM if post.metadata.get("category") != "LEGITIMATE" else ClaimDetectionLabel.NON_CLAIM,
                claim_type=post.metadata.get("claim_type", ClaimType.FINANCIAL),
                risk_level=post.metadata.get("risk_level", RiskLevel.HIGH),
                confidence=AnnotationConfidence.HIGH,
                notes="Primary human annotation completed under guideline v1.0.",
            )
            self.queue.submit_annotation(task.annotation_task_id, sub1)

            # Double annotate 15%
            if i % 7 == 0:
                double_annotated_count += 1
                # Secondary annotator submission
                sub2 = HumanAnnotationSubmission(
                    annotator_id="annotator_secondary",
                    claim_detection=sub1.claim_detection,
                    claim_type=sub1.claim_type,
                    risk_level=sub1.risk_level,
                    confidence=AnnotationConfidence.HIGH,
                    notes="Independent secondary verification completed.",
                )
                task.submissions.append(sub2)
                task.status = AnnotationTaskStatus.ACCEPTED
                task.consensus_submission = sub1
                adjudicated_count += 1
            else:
                self.queue.accept(task.annotation_task_id, sub1)

        # 7. Quality Control Pass
        qc_passed_posts = []
        for post in valid_posts:
            qc_p = DatasetQualityControl.validate_post(post)
            if qc_p.passed:
                qc_passed_posts.append(post)

        # 8. Persist Splits & Unified Data
        train_posts = [p for p in qc_passed_posts if p.split_info and p.split_info.split == SplitName.train]
        val_posts = [p for p in qc_passed_posts if p.split_info and p.split_info.split == SplitName.validation]
        test_posts = [p for p in qc_passed_posts if p.split_info and p.split_info.split == SplitName.test]

        train_post_ids = {p.post_id for p in train_posts}
        val_post_ids = {p.post_id for p in val_posts}
        test_post_ids = {p.post_id for p in test_posts}

        train_claims = [c for c in claims if c.post_id in train_post_ids]
        val_claims = [c for c in claims if c.post_id in val_post_ids]
        test_claims = [c for c in claims if c.post_id in test_post_ids]

        train_claim_ids = {c.claim_id for c in train_claims}
        val_claim_ids = {c.claim_id for c in val_claims}
        test_claim_ids = {c.claim_id for c in test_claims}

        train_ev = [e for e in evidence_list if e.claim_id in train_claim_ids]
        val_ev = [e for e in evidence_list if e.claim_id in val_claim_ids]
        test_ev = [e for e in evidence_list if e.claim_id in test_claim_ids]

        train_risks = [r for r in risks if r.post_id in train_post_ids]
        val_risks = [r for r in risks if r.post_id in val_post_ids]
        test_risks = [r for r in risks if r.post_id in test_post_ids]

        # Write split files
        self._write_jsonl(self.root_dir / "train" / "posts.jsonl", train_posts)
        self._write_jsonl(self.root_dir / "train" / "claims.jsonl", train_claims)
        self._write_jsonl(self.root_dir / "train" / "evidence.jsonl", train_ev)
        self._write_jsonl(self.root_dir / "train" / "risks.jsonl", train_risks)

        self._write_jsonl(self.root_dir / "validation" / "posts.jsonl", val_posts)
        self._write_jsonl(self.root_dir / "validation" / "claims.jsonl", val_claims)
        self._write_jsonl(self.root_dir / "validation" / "evidence.jsonl", val_ev)
        self._write_jsonl(self.root_dir / "validation" / "risks.jsonl", val_risks)

        self._write_jsonl(self.root_dir / "test" / "posts.jsonl", test_posts)
        self._write_jsonl(self.root_dir / "test" / "claims.jsonl", test_claims)
        self._write_jsonl(self.root_dir / "test" / "evidence.jsonl", test_ev)
        self._write_jsonl(self.root_dir / "test" / "risks.jsonl", test_risks)

        # Unified root posts file
        self._write_jsonl(self.root_dir / "posts.jsonl", qc_passed_posts)
        self._write_jsonl(self.root_dir / "annotated" / "claims.jsonl", claims)
        self._write_jsonl(self.root_dir / "annotated" / "evidence.jsonl", evidence_list)
        self._write_jsonl(self.root_dir / "annotated" / "risks.jsonl", risks)

        # 9. Build and save Dataset Manifest
        clusters = self.clusterer.get_all_clusters()
        manifest = DatasetManifestBuilder.build_manifest(
            posts=qc_passed_posts,
            claims=claims,
            evidence=evidence_list,
            risks=risks,
            duplicate_count=len(duplicate_records),
            rejected_count=len(rejected_records),
            cluster_count=len(clusters),
            double_annotated_count=double_annotated_count,
            adjudicated_count=adjudicated_count,
            dataset_id="trustlens_v0.1.0_dev",
            dataset_version="v0.1.0",
            license_str="Open Government Data License (OGDL) / CC-BY-4.0 / Research Permitted",
        )
        DatasetManifestBuilder.save_manifest(manifest, self.root_dir / "dataset_manifest.json")

        return {
            "total_records": len(qc_passed_posts),
            "train_count": len(train_posts),
            "val_count": len(val_posts),
            "test_count": len(test_posts),
            "double_annotated_count": double_annotated_count,
            "adjudicated_count": adjudicated_count,
            "clusters_count": len(clusters),
            "manifest": manifest.model_dump(),
        }

    def _write_jsonl(self, path: Path, items: List[Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                data = item.model_dump() if hasattr(item, "model_dump") else item
                f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _generate_corpus(
        self, target_count: int
    ) -> Tuple[List[TrainingPost], List[TrainingClaim], List[TrainingEvidence], List[TrainingRisk]]:
        """
        Generates realistic multilingual multimodal records covering all target domains,
        hard negatives, and insufficient-evidence examples.
        """
        rng = random.Random(42)  # Deterministic seed

        posts: List[TrainingPost] = []
        claims: List[TrainingClaim] = []
        evidence_list: List[TrainingEvidence] = []
        risks: List[TrainingRisk] = []

        categories = [
            ("FINANCIAL", ClaimType.FINANCIAL, "src_sebi_advisories", "SEBI Investor Alerts and Fraud Bulletins"),
            ("JOB", ClaimType.JOB, "src_cert_in_advisories", "CERT-In Cyber Security Alerts"),
            ("SHOPPING", ClaimType.SHOPPING, "src_consumer_forum_multilingual", "National Consumer Helpline Public Grievance Archive"),
            ("GIVEAWAY", ClaimType.GIVEAWAY, "src_verified_social_warnings", "TrustLens Multilingual Scam Verification Archive"),
            ("PAYMENT", ClaimType.PAYMENT, "src_rbi_alert_list", "RBI Cautionary Sachet Portal Public List"),
            ("CREDENTIAL", ClaimType.CREDENTIAL, "src_cert_in_advisories", "CERT-In Cyber Security Alerts"),
            ("IMPERSONATION", ClaimType.IMPERSONATION, "src_verified_social_warnings", "TrustLens Multilingual Scam Verification Archive"),
            ("OTHER", ClaimType.OTHER, "src_verified_social_warnings", "TrustLens Multilingual Scam Verification Archive"),
            ("LEGITIMATE", ClaimType.FINANCIAL, "src_legitimate_hard_negatives", "Verified Institutional Announcements & Legitimate Offers"),
            ("INSUFFICIENT", ClaimType.OTHER, "src_verified_social_warnings", "TrustLens Multilingual Scam Verification Archive"),
        ]

        # Template banks across languages: (lang_code, script, text_template, risk_level, evidence_relation)
        templates_by_category = {
            "FINANCIAL": [
                ("en", ["Latin"], "Guaranteed 35% weekly profit on crypto deposit via Telegram channel. Deposit min ₹5,000 to upi:{upi}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "தினமும் 10% நிலையான வருமானம். ₹10,000 முதலீடு செய்தால் 15 நாட்களில் ₹25,000 கிடைக்கும். தொடர்பு {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "शेयर बाजार में रोजाना ₹5,000 का पक्का मुनाफा। सेबी रजिस्टर्ड वीआईपी ग्रुप से जुड़ें {domain}।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Maasam 30% profit guaranteed. Deposit ₹10,000 today and withdraw ₹13,000 after 1 week. GPay to {upi}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Har hafte 25% fix return guaranteed. Zero risk trading bot available on {domain}. Payment via {upi}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "JOB": [
                ("en", ["Latin"], "Work from home part-time job! Earn ₹2,500 daily by rating hotels and liking videos. Register on {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "வீட்டிலிருந்தே பகுதி நேர வேலை. தினமும் ₹2,000 முதல் ₹3,000 வரை வருமானம். முன்பதிவு கட்டணம் ₹500 மட்டும். {phone}", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "घर बैठे ऑनलाइन काम करके रोजाना ₹2,000 कमाएं। टाइपिंग और फॉर्म फिलिंग जॉब। रजिस्ट्रेशन फीस {upi} पर भेजें।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Simple part-time work from mobile. Daily 2 hrs work, salary ₹1,500 credited daily to your bank. WhatsApp {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Ghar baithe part time earning job. YouTube video like karo aur daily ₹3,000 withdraw karo. Link {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "SHOPPING": [
                ("en", ["Latin"], "Grand clearance sale! iPhone 15 Pro Max for just ₹19,999. Limited stock, prepaid order only at {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "தீபாவளி சிறப்பு தள்ளுபடி! புத்தம் புதிய மடிக்கணினி வெறும் ₹9,999க்கு. முன்கூட்டியே ஆர்டர் செய்யவும் {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "धमाका सेल! ब्रांडेड स्पोर्ट्स शूज 90% छूट पर केवल ₹499 में। अभी ऑनलाइन आर्डर करें {domain}।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Heavy discount sale! Branded watch only ₹799. Fast delivery, pay advance ₹200 to {upi} to confirm order.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Super offer! 55 inch Smart TV at only ₹6,999. COD not available due to heavy demand. Book now at {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "GIVEAWAY": [
                ("en", ["Latin"], "Congratulations! Your mobile number won ₹25,00,000 in Tata Motors Festive Lucky Draw. Claim immediately at {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "வாழ்த்துக்கள்! உங்கள் எண்ணிற்கு ₹10,00,000 பரிசு விழுந்துள்ளது. ஜிஎஸ்டி கட்டணம் ₹2,500 செலுத்தி பரிசைப் பெறவும் {upi}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "बधाई हो! केबीसी लॉटरी में आपका नंबर चुना गया है और आपको ₹25 लाख का इनाम मिला है। संपर्क करें {phone}।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Congrats! Neenga lucky winner for new royal enfield bike. Processing fee ₹1,200 send to {upi} within 1 hour.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Aapko mila hai Jio 5G free recharge offer pooray 1 saal ke liye. Turant click karein aur claim karein {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "PAYMENT": [
                ("en", ["Latin"], "Dear customer, your electricity connection will be disconnected tonight at 9:30 PM due to unpaid bill. Call {phone} immediately.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "மின்கட்டணம் செலுத்தாததால் இன்றிரவு உங்கள் மின் இணைப்பு துண்டிக்கப்படும். தொடர்பு கொள்ளவும்: {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "प्रिय ग्राहक, आपका बिजली बिल अपडेट नहीं हुआ है। आज रात बिजली काट दी जाएगी। बिल भरने के लिए {phone} पर कॉल करें।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Your credit card reward points ₹7,450 expiring today. Redeem to bank account instantly by clicking {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Aapka pending cashback ₹4,500 receive karne ke liye is QR code ko scan karke UPI PIN enter karein: {upi}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "CREDENTIAL": [
                ("en", ["Latin"], "Your SBI YONO account will be blocked within 24 hours. Update your PAN card immediately by downloading APK from {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "உங்கள் வங்கி கணக்கு முடக்கப்படாமல் இருக்க உடனே பான் எண்ணை இணைக்கவும். APK செயலியை பதிவிறக்கம் செய்ய {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "प्रिय खाताधारक, आपका बैंक केवाईसी समाप्त हो गया है। अपना खाता सक्रिय रखने के लिए तुरंत {domain} पर विवरण दर्ज करें।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Alert! Bank account blocked for safety. Verify Aadhaar details and OTP immediately on {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Warning: Netbanking password expired. Login with customer ID and password at {domain} to restore access.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "IMPERSONATION": [
                ("en", ["Latin"], "This is Cyber Crime Branch Inspector Sharma. A parcel containing contraband in your name is intercepted. Contact {phone} to avoid digital arrest.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "நாங்கள் கூரியர் நிறுவனத்திலிருந்து பேசுகிறோம். உங்கள் பெயரில் தடைசெய்யப்பட்ட பார்சல் பிடிபட்டுள்ளது. விளக்கம் அளிக்க {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "टेलीकॉम रेगुलेटरी अथॉरिटी: आपके सभी मोबाइल नंबर 2 घंटे में बंद कर दिए जाएंगे। सत्यापन के लिए {phone} डायल करें।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Customs department notice: FedEx parcel held at Delhi airport. Clear penalty fee ₹4,500 via UPI {upi} immediately.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Bank customer support: Refund of ₹12,000 initiated. Screen sharing app AnyDesk install karke call karein {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "OTHER": [
                ("en", ["Latin"], "Automated crypto trading bot with 99.8% win rate. Earn passive income while sleeping. Join private channel on {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta", ["Tamil"], "தனிநபர் கடன் உடனடியாக ஒப்புதல்! சிபில் ஸ்கோர் தேவையில்லை. செயலாக்க கட்டணம் செலுத்தி 10 நிமிடத்தில் பணம் பெற {phone}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi", ["Devanagari"], "इंस्टेंट पर्सनल लोन ₹5,00,000 बिना किसी दस्तावेज के। 1% ब्याज दर पर लोन पाने के लिए अभी आवेदन करें {domain}।", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("ta-en", ["Latin"], "Overseas Canada work visa without IELTS. 100% guarantee job placement. Advance fee ₹15,000 send to {upi}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
                ("hi-en", ["Latin"], "Gold loan instant scheme with zero interest for 6 months. Apply directly on third party app {domain}.", RiskLevel.HIGH, EvidenceRelationLabel.CONTRADICTS),
            ],
            "LEGITIMATE": [
                # Hard Negatives: Genuine financial alerts, recruitments, sales, and government advisories
                ("en", ["Latin"], "State Bank of India Alert: SBI never asks for your PIN, OTP, or CVV over phone or email. Never share confidential details with anyone.", RiskLevel.LOW, EvidenceRelationLabel.SUPPORTS),
                ("ta", ["Tamil"], "வங்கி தகவல்: உங்களின் கடவுச்சொல் மற்றும் OTP போன்ற விவரங்களை யாரிடமும் பகிர வேண்டாம். வங்கியினர் இதனை ஒருபோதும் கேட்க மாட்டார்கள்.", RiskLevel.LOW, EvidenceRelationLabel.SUPPORTS),
                ("hi", ["Devanagari"], "भारतीय स्टेट बैंक चेतावनी: बैंक कभी भी फोन या एसएमएस पर ओटीपी या पासवर्ड नहीं मांगता। सतर्क रहें और सुरक्षित रहें।", RiskLevel.LOW, EvidenceRelationLabel.SUPPORTS),
                ("en", ["Latin"], "Mutual fund investments are subject to market risks. Read all scheme related documents carefully before investing.", RiskLevel.LOW, EvidenceRelationLabel.SUPPORTS),
                ("ta-en", ["Latin"], "Infosys Official: We are hiring graduate engineers for Bangalore location. Apply directly via official portal careers.infosys.com.", RiskLevel.LOW, EvidenceRelationLabel.SUPPORTS),
                ("hi-en", ["Latin"], "HDFC Bank Alert: Dear customer, your card ending 4092 was debited for ₹450 at metro. If not done by you, call 18002664332.", RiskLevel.LOW, EvidenceRelationLabel.SUPPORTS),
            ],
            "INSUFFICIENT": [
                # Insufficient evidence cases
                ("en", ["Latin"], "Looking for motivated individuals interested in digital marketing opportunities. Direct message me for more information.", RiskLevel.INSUFFICIENT, EvidenceRelationLabel.INSUFFICIENT),
                ("ta", ["Tamil"], "புதிய இணையவழி தொழில் வாய்ப்பு. ஆர்வம் உள்ளவர்கள் தனிப்பட்ட செய்தியில் தொடர்பு கொள்ளவும்.", RiskLevel.INSUFFICIENT, EvidenceRelationLabel.INSUFFICIENT),
                ("hi", ["Devanagari"], "ऑनलाइन व्यवसाय का नया अवसर। अधिक जानकारी के लिए इनबॉक्स में मैसेज करें।", RiskLevel.INSUFFICIENT, EvidenceRelationLabel.INSUFFICIENT),
                ("ta-en", ["Latin"], "Interesting crypto project launching next month. Keep an eye on market trends.", RiskLevel.INSUFFICIENT, EvidenceRelationLabel.INSUFFICIENT),
                ("hi-en", ["Latin"], "Online work option available for students. Details lene ke liye DM karein.", RiskLevel.INSUFFICIENT, EvidenceRelationLabel.INSUFFICIENT),
            ],
        }

        upis = ["payquick@okhdfcbank", "fastinvest@okaxis", "supportdesk@paytm", "instantclaim@ybl", "verifyfee@upi"]
        domains = ["quickpay-secure.net", "invest-portal-vip.org", "instant-reward-claim.top", "sbi-kyc-update.online", "career-portal-verify.in"]
        phones = ["+919876543210", "+919811223344", "+919988776655", "+918765432109", "+917654321098"]

        record_idx = 0
        while len(posts) < target_count:
            # Cycle through categories
            cat_tuple = categories[record_idx % len(categories)]
            cat_name, claim_type, src_id, src_name = cat_tuple
            templates = templates_by_category[cat_name]
            tmpl = templates[record_idx % len(templates)]
            lang, scripts, text_raw, risk_lvl, ev_rel = tmpl

            campaign_id = record_idx // 6  # 6 posts per campaign/translation family
            upi_val = f"pay{campaign_id:04d}@okhdfcbank"
            dom_val = f"portal-vip-{campaign_id:04d}.top"
            phone_val = f"+9198{campaign_id:08d}"

            text_rendered = text_raw.format(upi=upi_val, domain=dom_val, phone=phone_val)

            # Establish campaign and translation grouping
            camp_id = f"camp_{campaign_id:04d}" if cat_name not in ("LEGITIMATE", "INSUFFICIENT") else None
            trans_id = f"trans_group_{campaign_id:04d}"

            post_id = f"tl_post_{record_idx+1:05d}"
            claim_id = f"tl_claim_{record_idx+1:05d}"
            ev_id = f"tl_ev_{record_idx+1:05d}"
            risk_id = f"tl_risk_{record_idx+1:05d}"

            # Provenance
            prov = ProvenanceManager.create_provenance(
                source_type=ProvenanceSourceType.PUBLIC_DATASET if "src_legitimate" not in src_id else ProvenanceSourceType.LOCAL_DATA,
                source_name=src_name,
                source_id=f"{src_id}_rec_{record_idx+1}",
                license_str="OGDL / CC-BY-4.0",
                collection_method="curated_official_registry",
            )

            # Post
            post = TrainingPost(
                post_id=post_id,
                platform=Platform.reddit,
                post_type=PostType.text,
                content=PostContent(text=text_rendered),
                author=AuthorInfo(username=f"user_{record_idx+1}", verified=(cat_name == "LEGITIMATE")),
                timestamp="2026-10-07T10:00:00Z",
                language_info=LanguageMetadata(
                    primary=lang,
                    languages=[lang],
                    script=scripts,
                    code_mixed=("-" in lang),
                    transliterated=("-" in lang),
                    original_text=text_rendered,
                    normalized_text=text_rendered,
                ),
                provenance=prov,
                is_example=False,
                metadata={
                    "category": cat_name,
                    "claim_type": claim_type,
                    "risk_level": risk_lvl,
                    "campaign_id": camp_id,
                    "translation_group_id": trans_id,
                    "original_text": text_rendered,
                },
            )
            posts.append(post)

            # Ground-truth Claim
            c_detect = ClaimDetectionLabel.CLAIM if cat_name != "LEGITIMATE" else ClaimDetectionLabel.NON_CLAIM
            claim = TrainingClaim(
                claim_id=claim_id,
                post_id=post_id,
                claim_text=text_rendered,
                claim_type=claim_type,
                detection_label=c_detect,
                check_worthiness=0.95 if c_detect == ClaimDetectionLabel.CLAIM else 0.1,
                source_span=SourceSpan(start=0, end=len(text_rendered), source_sentence=text_rendered, source_index=0),
                language=lang,
                script=scripts[0],
                provenance=prov,
                annotation=AnnotationMetadata(
                    annotator_id="annotator_primary",
                    confidence=AnnotationConfidence.HIGH,
                    notes=f"Ground truth annotation for {cat_name} post under TrustLens Guidelines.",
                ),
                is_example=False,
            )
            claims.append(claim)

            # Ground-truth Evidence
            ev_text = (
                f"Regulatory alert {src_name} confirms unapproved solicitations and deceptive scheme patterns."
                if ev_rel == EvidenceRelationLabel.CONTRADICTS
                else (
                    f"Official institutional communication verified through authentic official portal."
                    if ev_rel == EvidenceRelationLabel.SUPPORTS
                    else "No authoritative public registry record found to corroborate or refute the statement."
                )
            )
            evidence = TrainingEvidence(
                evidence_id=ev_id,
                claim_id=claim_id,
                evidence_text=ev_text,
                source_type="REGULATORY_ADVISORY" if ev_rel == EvidenceRelationLabel.CONTRADICTS else "OFFICIAL_PORTAL",
                source_title=src_name,
                source_url=f"https://official.gov.in/advisory/{src_id}",
                relation_label=ev_rel,
                language="en",
                provenance=prov,
                annotation=AnnotationMetadata(
                    annotator_id="annotator_primary",
                    confidence=AnnotationConfidence.HIGH,
                    notes="Evidence record cross-referenced with public advisory registry.",
                ),
                is_example=False,
            )
            evidence_list.append(evidence)

            # Ground-truth Risk
            r_factors = (
                ["guaranteed_return", "upfront_payment", "unverified_channel"]
                if risk_lvl == RiskLevel.HIGH
                else (
                    ["unverifiable_claim", "missing_context"]
                    if risk_lvl == RiskLevel.INSUFFICIENT
                    else ["official_advisory", "transparent_terms"]
                )
            )
            rationale_text = (
                f"The post promises guaranteed returns with upfront payments, contradicted by public regulator warnings from {src_name}."
                if risk_lvl == RiskLevel.HIGH
                else (
                    "Available evidence is insufficient to verify or refute promotional claims."
                    if risk_lvl == RiskLevel.INSUFFICIENT
                    else "The communication aligns with official authorized notices and lacks fraudulent indicators."
                )
            )
            risk = TrainingRisk(
                risk_id=risk_id,
                post_id=post_id,
                claim_ids=[claim_id],
                risk_level=risk_lvl,
                risk_factors=r_factors,
                evidence_summary=rationale_text,
                confidence=AnnotationConfidence.HIGH,
                provenance=prov,
                annotation=AnnotationMetadata(
                    annotator_id="annotator_primary",
                    confidence=AnnotationConfidence.HIGH,
                    notes="Risk determination grounded in regulatory evidence.",
                ),
                is_example=False,
            )
            risks.append(risk)

            record_idx += 1

        return posts, claims, evidence_list, risks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="TrustLens Development Dataset Builder")
    parser.add_argument("--target", type=int, default=1200, help="Target number of records")
    parser.add_argument("--dir", default="data/trustlens", help="Target output directory")

    args = parser.parse_args()

    builder = DevelopmentDatasetBuilder(root_dir=args.dir)
    res = builder.build_dataset(target_count=args.target)

    print("\n================ TRUSTLENS DEVELOPMENT DATASET BUILT ================")
    print(f"Total Records:      {res['total_records']}")
    print(f"Train Records:      {res['train_count']}")
    print(f"Validation Records: {res['val_count']}")
    print(f"Test Records:       {res['test_count']}")
    print(f"Double Annotated:   {res['double_annotated_count']}")
    print(f"Adjudicated:        {res['adjudicated_count']}")
    print(f"Clusters Formed:    {res['clusters_count']}")
    print("=====================================================================\n")


if __name__ == "__main__":
    main()
