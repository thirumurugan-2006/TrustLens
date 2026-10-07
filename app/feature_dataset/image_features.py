"""
TrustLens Image & OCR Feature Extractor (Phase 6D).
Extracts grounded media volume, OCR text presence, confidence, and hash match features.
"""

from typing import Any, Dict, List

from app.training.schemas import ImageMetadata, TrainingPost


class ImageFeatureExtractor:
    """Extracts image and OCR features from post media metadata."""

    @classmethod
    def extract(cls, post: TrainingPost) -> Dict[str, Any]:
        """Extracts image and OCR statistics from a TrainingPost."""
        images_meta = getattr(post, "images_meta", []) or []
        media_images = getattr(post.media, "images", []) if post.media else []
        img_count = max(len(images_meta), len(media_images))

        if img_count == 0:
            return {
                "image_count": 0,
                "has_image": False,
                "ocr_available": False,
                "ocr_confidence": -1.0,
                "ocr_text_length": 0,
                "image_reuse_count": 0,
                "phash_match_count": 0,
                "ocr_unreliable": False,
            }

        ocr_available = False
        ocr_conf_sum = 0.0
        ocr_count = 0
        ocr_len = 0
        phash_count = 0
        reuse_count = 0

        for im in images_meta:
            im_dict = im if isinstance(im, dict) else (im.model_dump() if hasattr(im, "model_dump") else {})
            ocr_text = im_dict.get("ocr_text")
            if ocr_text:
                ocr_available = True
                ocr_len += len(ocr_text)
                conf = im_dict.get("ocr_confidence")
                if conf is not None:
                    ocr_conf_sum += float(conf)
                    ocr_count += 1
            if im_dict.get("phash"):
                phash_count += 1
            if im_dict.get("image_reuse"):
                reuse_count += 1

        mean_conf = round(ocr_conf_sum / ocr_count, 4) if ocr_count > 0 else -1.0
        unreliable = bool(ocr_available and (mean_conf >= 0.0 and mean_conf < 0.50))

        return {
            "image_count": img_count,
            "has_image": True,
            "ocr_available": ocr_available,
            "ocr_confidence": mean_conf,
            "ocr_text_length": ocr_len,
            "image_reuse_count": reuse_count,
            "phash_match_count": phash_count,
            "ocr_unreliable": unreliable,
        }
