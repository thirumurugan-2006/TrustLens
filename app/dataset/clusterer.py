import re
from typing import Dict, List, Optional, Set

from app.dataset.schemas import ClusterInfo
from app.training.schemas import TrainingPost


class DisjointSetUnion:
    """Disjoint Set Union (DSU) with path compression and union by rank."""

    def __init__(self):
        self.parent: Dict[str, str] = {}
        self.rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.rank[item] = 0
            return item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1: str, item2: str) -> None:
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1


class DatasetClusterer:
    """
    Manages leakage grouping across:
    1. source_group_id (author / domain)
    2. campaign_group_id (shared target URLs, UPI handles, phone numbers)
    3. post_family_id (near-duplicate text clusters)
    4. image_family_id (pHash image variants)
    5. translation_group_id (cross-lingual content variants, e.g. English + Tamil + Tanglish)

    Computes composite cluster closure using Disjoint Set Union (DSU)
    guaranteeing that related variants are never distributed across different splits.
    """

    def __init__(self):
        self.dsu = DisjointSetUnion()
        self.post_clusters: Dict[str, ClusterInfo] = {}

    @staticmethod
    def extract_campaign_signals(text: str) -> List[str]:
        """Extracts high-fidelity campaign signatures: URLs, UPI handles, phone numbers."""
        signals = []
        # UPI handles (e.g. name@okaxis, pay@upi)
        upis = re.findall(r"[\w\.\-]+@(?:okhdfcbank|okaxis|oksbi|paytm|ybl|upi|apl)", text, re.IGNORECASE)
        signals.extend([f"upi:{u.lower()}" for u in upis])

        # URLs / domains
        urls = re.findall(r"https?://(?:www\.)?([^\s/\?]+)", text, re.IGNORECASE)
        signals.extend([f"domain:{u.lower()}" for u in urls if "example.com" not in u.lower()])

        # 10-digit phone numbers
        phones = re.findall(r"(?:(?:\+91|0)?[6-9]\d{9})", text)
        signals.extend([f"phone:{p}" for p in phones])

        return signals

    def register_post(
        self,
        post: TrainingPost,
        post_family_id: Optional[str] = None,
        image_family_id: Optional[str] = None,
        translation_group_id: Optional[str] = None,
        campaign_group_id: Optional[str] = None,
    ) -> ClusterInfo:
        """
        Registers a TrainingPost, establishes relations across clusters,
        and computes the composite cluster identifier.
        """
        post_id = post.post_id

        # 1. Author source grouping
        author_val = post.author.username or post.author.id
        source_group_id = f"src_{author_val}" if author_val else None

        # 2. Extract or use campaign grouping
        if not campaign_group_id:
            signals = self.extract_campaign_signals(post.content.text or "")
            if signals:
                campaign_group_id = f"camp_{signals[0]}"

        # Base node for post
        self.dsu.find(post_id)

        # Connect all grouping keys in DSU
        groups_to_link = []
        if source_group_id:
            groups_to_link.append(f"G_SRC:{source_group_id}")
        if campaign_group_id:
            groups_to_link.append(f"G_CAMP:{campaign_group_id}")
        if post_family_id:
            groups_to_link.append(f"G_POSTFAM:{post_family_id}")
        if image_family_id:
            groups_to_link.append(f"G_IMGFAM:{image_family_id}")
        if translation_group_id:
            groups_to_link.append(f"G_TRANS:{translation_group_id}")

        for g in groups_to_link:
            self.dsu.union(post_id, g)

        # Composite cluster is the DSU root
        composite_id = self.dsu.find(post_id)

        cluster_info = ClusterInfo(
            source_group_id=source_group_id,
            campaign_group_id=campaign_group_id,
            post_family_id=post_family_id,
            image_family_id=image_family_id,
            translation_group_id=translation_group_id,
            composite_cluster_id=composite_id,
        )

        self.post_clusters[post_id] = cluster_info
        return cluster_info

    def get_composite_cluster(self, post_id: str) -> str:
        """Returns the canonical root cluster ID for a post."""
        return self.dsu.find(post_id)

    def get_all_clusters(self) -> Dict[str, List[str]]:
        """Returns mapping from composite_cluster_id -> list of post_ids."""
        clusters: Dict[str, List[str]] = {}
        for post_id in self.post_clusters:
            root = self.dsu.find(post_id)
            clusters.setdefault(root, []).append(post_id)
        return clusters
