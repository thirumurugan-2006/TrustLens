"""
TrustLens Node Builder (Phase 6C).
Builds grounded, typed heterogeneous graph nodes from posts, claims, evidence, and multimodal media.
"""

import hashlib
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from app.graph_dataset.schemas import (
    AccountNode,
    ClaimNode,
    CommentNode,
    EvidenceNode,
    GraphNode,
    ImageNode,
    NodeType,
    PostNode,
    UrlNode,
)
from app.training.schemas import (
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
)


class NodeBuilder:
    """
    Constructs strongly-typed heterogeneous graph nodes from grounded source records.
    Never fabricates nodes or multimodal placeholders.
    """

    @staticmethod
    def normalize_url(url: str) -> str:
        """Normalizes URL string without stripping meaningful query paths."""
        return url.strip()

    @staticmethod
    def extract_domain(url: str) -> str:
        """Extracts netloc / domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            return domain if domain else "unknown_domain"
        except Exception:
            return "unknown_domain"

    def build_post_node(
        self,
        post: TrainingPost,
        cluster_id: str,
        split: SplitName,
    ) -> PostNode:
        """Builds a POST node from a TrainingPost."""
        text = post.content.text if (post.content and post.content.text) else getattr(post, "text", "")
        lang = post.language_info.primary if post.language_info else "en"
        script = post.language_info.script[0] if (post.language_info and post.language_info.script) else "Latin"
        platform = post.platform.value if hasattr(post.platform, "value") else str(post.platform)

        return PostNode(
            node_id=f"node_post_{post.post_id}",
            node_type=NodeType.POST,
            source_id=post.post_id,
            post_id=post.post_id,
            text=text,
            language=lang,
            script=script,
            platform=platform,
            split=split,
            cluster_id=cluster_id,
            metadata={
                "campaign_group_id": getattr(post.split_info, "campaign_group_id", None) if post.split_info else None,
                "source_group_id": getattr(post.split_info, "source_group_id", None) if post.split_info else None,
                "translation_group_id": getattr(post.split_info, "translation_group_id", None) if post.split_info else None,
                "timestamp": post.timestamp,
            },
        )

    def build_claim_node(
        self,
        claim: TrainingClaim,
        cluster_id: str,
        split: SplitName,
    ) -> ClaimNode:
        """Builds a CLAIM node from a TrainingClaim."""
        c_type = claim.claim_type.value if hasattr(claim.claim_type, "value") else str(claim.claim_type)
        atomic_id = f"atomic_{claim.claim_id}_01"

        return ClaimNode(
            node_id=f"node_claim_{claim.claim_id}",
            node_type=NodeType.CLAIM,
            source_id=claim.claim_id,
            claim_id=claim.claim_id,
            parent_post_id=claim.post_id,
            atomic_claim_id=atomic_id,
            claim_text=claim.claim_text,
            claim_type=c_type,
            language=claim.language or "en",
            split=split,
            cluster_id=cluster_id,
            metadata={
                "check_worthiness": claim.check_worthiness,
                "detection_label": getattr(claim.detection_label, "value", str(claim.detection_label)),
                "claim_status": claim.claim_status,
            },
        )

    def build_evidence_node(
        self,
        evidence: TrainingEvidence,
        cluster_id: str,
        split: SplitName,
    ) -> EvidenceNode:
        """Builds an EVIDENCE node from a TrainingEvidence record."""
        return EvidenceNode(
            node_id=f"node_ev_{evidence.evidence_id}",
            node_type=NodeType.EVIDENCE,
            source_id=evidence.evidence_id,
            evidence_id=evidence.evidence_id,
            claim_id=evidence.claim_id,
            evidence_text=evidence.evidence_text,
            source_type=evidence.source_type,
            source_url=evidence.source_url,
            relation=evidence.relation_label,
            provenance=evidence.provenance,
            split=split,
            cluster_id=cluster_id,
            metadata={
                "source_title": evidence.source_title,
                "retrieval_method": evidence.retrieval_method,
                "reliability_metadata": evidence.reliability_metadata,
            },
        )

    def build_url_node(
        self,
        url: str,
        cluster_id: str,
        split: SplitName,
    ) -> UrlNode:
        """Builds a URL node from a valid URL string."""
        clean_url = self.normalize_url(url)
        domain = self.extract_domain(clean_url)
        url_hash = hashlib.sha256(clean_url.encode("utf-8")).hexdigest()[:12]
        url_id = f"url_{url_hash}"

        return UrlNode(
            node_id=f"node_{url_id}",
            node_type=NodeType.URL,
            source_id=url_id,
            url_id=url_id,
            url=clean_url,
            domain=domain,
            split=split,
            cluster_id=cluster_id,
            metadata={"domain": domain},
        )

    def build_account_node(
        self,
        post: TrainingPost,
        cluster_id: str,
        split: SplitName,
    ) -> Optional[AccountNode]:
        """Builds an ACCOUNT node from post author info if available."""
        if not post.author or not post.author.username:
            return None

        username = post.author.username
        platform = post.platform.value if hasattr(post.platform, "value") else str(post.platform)
        acc_hash = hashlib.sha256(f"{platform}:{username}".encode("utf-8")).hexdigest()[:10]
        acc_id = f"acc_{platform}_{acc_hash}"

        return AccountNode(
            node_id=f"node_{acc_id}",
            node_type=NodeType.ACCOUNT,
            source_id=acc_id,
            account_id=acc_id,
            platform=platform,
            username=username,
            is_verified=bool(post.author.verified),
            split=split,
            cluster_id=cluster_id,
            metadata={
                "display_name": post.author.display_name,
                "platform": platform,
            },
        )

    def build_image_node(
        self,
        image_meta: Dict[str, Any],
        cluster_id: str,
        split: SplitName,
    ) -> ImageNode:
        """Builds an IMAGE node from image metadata."""
        img_id = image_meta.get("image_id") or f"img_{hashlib.sha256(str(image_meta).encode()).hexdigest()[:10]}"
        return ImageNode(
            node_id=f"node_img_{img_id}",
            node_type=NodeType.IMAGE,
            source_id=img_id,
            image_id=img_id,
            image_hash=image_meta.get("image_hash"),
            phash=image_meta.get("phash"),
            split=split,
            cluster_id=cluster_id,
            metadata=image_meta,
        )

    def build_comment_node(
        self,
        comment_dict: Dict[str, Any],
        parent_post_id: str,
        cluster_id: str,
        split: SplitName,
    ) -> CommentNode:
        """Builds a COMMENT node from comment dictionary."""
        cmt_id = comment_dict.get("comment_id") or f"cmt_{hashlib.sha256(str(comment_dict).encode()).hexdigest()[:10]}"
        return CommentNode(
            node_id=f"node_cmt_{cmt_id}",
            node_type=NodeType.COMMENT,
            source_id=cmt_id,
            comment_id=cmt_id,
            parent_post_id=parent_post_id,
            comment_text=comment_dict.get("text") or comment_dict.get("comment_text", ""),
            stance=comment_dict.get("stance"),
            language=comment_dict.get("language", "en"),
            split=split,
            cluster_id=cluster_id,
            metadata=comment_dict,
        )
