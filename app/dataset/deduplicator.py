import hashlib
from typing import Dict, List, Optional, Set, Tuple

from app.dataset.schemas import DuplicateInfo, DuplicateType
from app.training.schemas import TrainingPost


def compute_text_hash(text: str) -> str:
    """Computes SHA-256 hash of normalized text."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def compute_token_shingles(text: str, n: int = 3) -> Set[str]:
    """Generates character n-grams for fast Jaccard near-duplicate comparison."""
    clean = " ".join(text.lower().split())
    if len(clean) < n:
        return {clean}
    return {clean[i : i + n] for i in range(len(clean) - n + 1)}


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """Computes Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def compute_simulated_phash(image_identifier: str) -> str:
    """
    Computes a 64-bit hex hash representation for an image path or identifier.
    In testing or standard processing without heavyweight PIL rendering,
    derives a stable 16-character hex hash.
    """
    return hashlib.sha256(image_identifier.encode("utf-8")).hexdigest()[:16]


def hamming_distance(hex_hash1: str, hex_hash2: str) -> int:
    """Computes bitwise Hamming distance between two 16-char hex strings."""
    try:
        val1 = int(hex_hash1, 16)
        val2 = int(hex_hash2, 16)
        return bin(val1 ^ val2).count("1")
    except ValueError:
        return 64


class DatasetDeduplicator:
    """
    Deterministic deduplication engine identifying exact duplicates,
    near-duplicate text paraphrases, and screenshot image variants.
    Never deletes records; flags duplicate_of, duplicate_type, and duplicate_score.
    """

    def __init__(self, text_similarity_threshold: float = 0.85, image_hamming_threshold: int = 5):
        self.text_similarity_threshold = text_similarity_threshold
        self.image_hamming_threshold = image_hamming_threshold

        # Indices of canonical records
        self._exact_hash_index: Dict[str, str] = {}  # text_hash -> canonical_post_id
        self._shingle_index: Dict[str, Set[str]] = {}  # post_id -> shingles
        self._canonical_posts: Dict[str, TrainingPost] = {}  # post_id -> TrainingPost
        self._image_hash_index: Dict[str, str] = {}  # image_hash -> canonical_post_id

    def check_and_register(self, post: TrainingPost) -> DuplicateInfo:
        """
        Evaluates a TrainingPost against existing records.
        If unique, registers it as a canonical record.
        If duplicate, returns DuplicateInfo referencing the canonical record.
        """
        text = post.content.text or ""
        text_hash = compute_text_hash(text)

        # 1. Check Exact Text Duplicate
        if text_hash in self._exact_hash_index:
            canonical_id = self._exact_hash_index[text_hash]
            return DuplicateInfo(
                is_duplicate=True,
                duplicate_of=canonical_id,
                duplicate_type=DuplicateType.EXACT,
                duplicate_score=1.0,
            )

        # 2. Check Near-Duplicate Text (Shingles / Jaccard)
        current_shingles = compute_token_shingles(text)
        for canon_id, canon_shingles in self._shingle_index.items():
            sim = jaccard_similarity(current_shingles, canon_shingles)
            if sim >= self.text_similarity_threshold:
                return DuplicateInfo(
                    is_duplicate=True,
                    duplicate_of=canon_id,
                    duplicate_type=DuplicateType.TEXT_SIMILARITY,
                    duplicate_score=round(sim, 4),
                )

        # 3. Check Image / Screenshot Variant (pHash)
        if post.media.images:
            for img_path in post.media.images:
                img_hash = compute_simulated_phash(img_path)
                for canon_img_hash, canon_id in self._image_hash_index.items():
                    dist = hamming_distance(img_hash, canon_img_hash)
                    if dist <= self.image_hamming_threshold:
                        return DuplicateInfo(
                            is_duplicate=True,
                            duplicate_of=canon_id,
                            duplicate_type=DuplicateType.IMAGE_PHASH,
                            duplicate_score=round(1.0 - (dist / 64.0), 4),
                        )

        # Record is UNIQUE: Register as canonical
        self._exact_hash_index[text_hash] = post.post_id
        self._shingle_index[post.post_id] = current_shingles
        self._canonical_posts[post.post_id] = post

        if post.media.images:
            for img_path in post.media.images:
                self._image_hash_index[compute_simulated_phash(img_path)] = post.post_id

        return DuplicateInfo(is_duplicate=False)
