"""
TrustLens Domain Feature Extractor (Phase 6D).
Extracts grounded domain, URL structure, and official authority matches.
Enforces explicit missingness representation for domain age without external WHOIS dependency.
"""

from typing import Any, Dict, List, Set
from urllib.parse import urlparse

from app.training.schemas import TrainingEvidence, TrainingPost


class DomainFeatureExtractor:
    """Extracts URL and domain intelligence features."""

    OFFICIAL_TLDS = {".gov", "gov.in", "official.gov", "sebi.gov.in", "rbi.org.in", "cert-in.org.in"}
    SHORTENERS = {"bit.ly", "tinyurl.com", "t.me", "is.gd", "buff.ly", "ow.ly", "goo.gl"}

    @classmethod
    def extract(
        cls,
        post: TrainingPost,
        evidence_list: List[TrainingEvidence],
    ) -> Dict[str, Any]:
        """Extracts domain features from post content and evidence URLs."""
        urls: List[str] = []

        # From post content
        if post.content and post.content.urls:
            urls.extend(post.content.urls)
        text = post.content.text if post.content else ""

        # From evidence
        for e in evidence_list:
            if e.source_url:
                urls.append(e.source_url)

        domains: Set[str] = set()
        official_match = False
        shortened_count = 0

        # Check shortened links in text
        for shortener in cls.SHORTENERS:
            if shortener in text.lower():
                shortened_count += 1

        for u in urls:
            try:
                netloc = urlparse(u).netloc.lower()
                if netloc:
                    domains.add(netloc)
                    if any(off in netloc for off in cls.OFFICIAL_TLDS):
                        official_match = True
                    if any(sh in netloc for sh in cls.SHORTENERS):
                        shortened_count += 1
            except Exception:
                pass

        domain_count = len(domains)
        unique_domain_count = len(domains)

        return {
            "domain_count": domain_count,
            "unique_domain_count": unique_domain_count,
            "official_domain_match": official_match,
            "domain_age_days": -1.0,
            "domain_age_available": False,
            "shortened_url_count": shortened_count,
        }
