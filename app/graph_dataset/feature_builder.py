"""
TrustLens Graph Feature Builder (Phase 6C).
Builds grounded feature representations for heterogeneous nodes without fabricating embeddings or fake scores.
"""

from typing import Any, Dict, Optional

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


class GraphFeatureBuilder:
    """
    Extracts structured discrete and categorical features from grounded node metadata.
    Strict rule: Never fabricate dense text embeddings, CLIP vectors, or domain scores.
    """

    # Controlled categorical mappings
    LANGUAGE_VOCAB = {"en": 0, "ta": 1, "hi": 2, "ta-en": 3, "hi-en": 4}
    PLATFORM_VOCAB = {"reddit": 0, "telegram": 1, "x": 2, "instagram": 3, "facebook": 4, "whatsapp": 5, "youtube": 6, "linkedin": 7, "tiktok": 8, "unknown": 9}
    CLAIM_TYPE_VOCAB = {"FINANCIAL": 0, "JOB": 1, "SHOPPING": 2, "GIVEAWAY": 3, "PAYMENT": 4, "CREDENTIAL": 5, "IMPERSONATION": 6, "OTHER": 7, "UNCLASSIFIED": 8}
    RELATION_VOCAB = {"SUPPORTS": 0, "CONTRADICTS": 1, "NEUTRAL": 2, "INSUFFICIENT": 3}

    @classmethod
    def build_post_features(
        cls,
        post_node: PostNode,
        claim_count: int = 1,
        image_count: int = 0,
        comment_count: int = 0,
    ) -> Dict[str, Any]:
        """Constructs grounded POST node features."""
        lang_idx = cls.LANGUAGE_VOCAB.get(post_node.language.lower(), cls.LANGUAGE_VOCAB["en"])
        plat_idx = cls.PLATFORM_VOCAB.get(post_node.platform.lower(), cls.PLATFORM_VOCAB["unknown"])
        text_len = len(post_node.text.split()) if post_node.text else 0

        return {
            "language_idx": lang_idx,
            "platform_idx": plat_idx,
            "token_count": text_len,
            "claim_count": claim_count,
            "image_count": image_count,
            "comment_count": comment_count,
            "text_embedding_ref": None,
            "embedding_available": False,
        }

    @classmethod
    def build_claim_features(
        cls,
        claim_node: ClaimNode,
    ) -> Dict[str, Any]:
        """Constructs grounded CLAIM node features."""
        ctype_idx = cls.CLAIM_TYPE_VOCAB.get(claim_node.claim_type.upper(), cls.CLAIM_TYPE_VOCAB["UNCLASSIFIED"])
        lang_idx = cls.LANGUAGE_VOCAB.get(claim_node.language.lower(), cls.LANGUAGE_VOCAB["en"])
        check_worth = float(claim_node.metadata.get("check_worthiness", 1.0))

        return {
            "claim_type_idx": ctype_idx,
            "language_idx": lang_idx,
            "check_worthiness": check_worth,
            "is_verifiable": True,
            "claim_embedding_ref": None,
            "embedding_available": False,
        }

    @classmethod
    def build_evidence_features(
        cls,
        evidence_node: EvidenceNode,
    ) -> Dict[str, Any]:
        """Constructs grounded EVIDENCE node features."""
        rel_val = evidence_node.relation.value if hasattr(evidence_node.relation, "value") else str(evidence_node.relation)
        rel_idx = cls.RELATION_VOCAB.get(rel_val.upper(), cls.RELATION_VOCAB["NEUTRAL"])
        has_url = evidence_node.source_url is not None and len(evidence_node.source_url.strip()) > 0

        return {
            "relation_idx": rel_idx,
            "has_source_url": has_url,
            "source_type": evidence_node.source_type,
            "evidence_embedding_ref": None,
            "embedding_available": False,
        }

    @classmethod
    def build_url_features(
        cls,
        url_node: UrlNode,
    ) -> Dict[str, Any]:
        """Constructs grounded URL node features."""
        is_gov = any(gov in url_node.domain for gov in [".gov", "gov.in", "official.gov", "sebi", "rbi", "cert-in"])
        return {
            "domain": url_node.domain,
            "is_official_registry": is_gov,
            "url_length": len(url_node.url),
        }

    @classmethod
    def build_account_features(
        cls,
        account_node: AccountNode,
    ) -> Dict[str, Any]:
        """Constructs grounded ACCOUNT node features."""
        plat_idx = cls.PLATFORM_VOCAB.get(account_node.platform.lower(), cls.PLATFORM_VOCAB["unknown"])
        return {
            "platform_idx": plat_idx,
            "is_verified": account_node.is_verified,
            "account_age_days": None,  # Grounded None, not fabricated
        }

    @classmethod
    def build_image_features(
        cls,
        image_node: ImageNode,
    ) -> Dict[str, Any]:
        """Constructs grounded IMAGE node features."""
        return {
            "has_phash": image_node.phash is not None,
            "has_image_hash": image_node.image_hash is not None,
            "clip_embedding_ref": None,
            "embedding_available": False,
        }

    @classmethod
    def build_comment_features(
        cls,
        comment_node: CommentNode,
    ) -> Dict[str, Any]:
        """Constructs grounded COMMENT node features."""
        return {
            "stance": comment_node.stance,
            "language": comment_node.language,
            "length": len(comment_node.comment_text.split()) if comment_node.comment_text else 0,
        }

    def attach_features_to_node(
        self,
        node: GraphNode,
        **context: Any,
    ) -> GraphNode:
        """Attaches computed features dictionary to the appropriate node instance."""
        if isinstance(node, PostNode):
            node.features = self.build_post_features(
                node,
                claim_count=context.get("claim_count", 1),
                image_count=context.get("image_count", 0),
                comment_count=context.get("comment_count", 0),
            )
        elif isinstance(node, ClaimNode):
            node.features = self.build_claim_features(node)
        elif isinstance(node, EvidenceNode):
            node.features = self.build_evidence_features(node)
        elif isinstance(node, UrlNode):
            node.features = self.build_url_features(node)
        elif isinstance(node, AccountNode):
            node.features = self.build_account_features(node)
        elif isinstance(node, ImageNode):
            node.features = self.build_image_features(node)
        elif isinstance(node, CommentNode):
            node.features = self.build_comment_features(node)
        return node
