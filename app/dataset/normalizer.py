import csv
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.dataset.schemas import RawDataRecord
from app.input.schemas import AuthorInfo, Platform, PostContent, PostMedia, PostType
from app.preprocessing.language.language_analyzer import LanguageAnalyzer
from app.training.schemas import ImageMetadata, LanguageMetadata, ProvenanceMetadata, TrainingPost


class NormalizationError(ValueError):
    """Raised when an input record cannot be parsed or normalized."""
    pass


class DatasetNormalizer:
    """
    Normalizes heterogeneous input formats (JSON, JSONL, CSV, TXT) into canonical
    TrainingPost instances while preserving raw records and original text strings.
    """

    def __init__(self):
        self.language_analyzer = LanguageAnalyzer()

    def clean_text(self, text: str) -> str:
        """Clean normalized text while preserving full semantic content."""
        if not text:
            return ""
        # Collapse excessive whitespace and strip
        return re.sub(r"\s+", " ", text).strip()

    def parse_file(self, file_path: Union[str, Path]) -> List[RawDataRecord]:
        """
        Reads a file (JSON, JSONL, CSV, TXT) and returns raw preserved records.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

        suffix = path.suffix.lower()
        records: List[RawDataRecord] = []

        if suffix == ".json":
            content = path.read_text(encoding="utf-8")
            data = json.loads(content)
            if isinstance(data, list):
                for item in data:
                    records.append(RawDataRecord(format="json", raw_payload=item, source_path=str(path)))
            elif isinstance(data, dict):
                # Check if it has a list under 'posts', 'records', 'data'
                for key in ["posts", "records", "data", "items"]:
                    if key in data and isinstance(data[key], list):
                        for item in data[key]:
                            records.append(RawDataRecord(format="json", raw_payload=item, source_path=str(path)))
                        break
                else:
                    records.append(RawDataRecord(format="json", raw_payload=data, source_path=str(path)))

        elif suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line_s = line.strip()
                    if line_s:
                        item = json.loads(line_s)
                        records.append(RawDataRecord(format="jsonl", raw_payload=item, source_path=str(path)))

        elif suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(RawDataRecord(format="csv", raw_payload=dict(row), source_path=str(path)))

        elif suffix in (".txt", ".text"):
            content = path.read_text(encoding="utf-8")
            # If multiple paragraphs separated by double newline, split into records
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            if not paragraphs and content.strip():
                paragraphs = [content.strip()]
            for p in paragraphs:
                records.append(RawDataRecord(format="txt", raw_payload={"text": p}, source_path=str(path)))

        else:
            raise NormalizationError(f"Unsupported file format: {suffix}")

        return records

    def normalize_record(
        self,
        raw_record: RawDataRecord,
        provenance: ProvenanceMetadata,
        default_platform: Union[Platform, str] = Platform.generic,
    ) -> TrainingPost:
        """
        Converts a RawDataRecord into a canonical TrainingPost, retaining original and normalized text.
        """
        payload = raw_record.raw_payload

        # 1. Extract Post ID
        post_id = str(payload.get("post_id") or payload.get("id") or f"post_{uuid.uuid4().hex[:12]}")

        # 2. Extract and preserve text
        original_text = str(
            payload.get("original_text")
            or payload.get("text")
            or payload.get("content", {}).get("text")
            or payload.get("body")
            or payload.get("title")
            or ""
        )
        normalized_text = self.clean_text(original_text)

        title = payload.get("title") or (payload.get("content", {}).get("title") if isinstance(payload.get("content"), dict) else None)

        # 3. Extract platform
        raw_platform = payload.get("platform") or default_platform
        if isinstance(raw_platform, Platform):
            platform_val = raw_platform
        else:
            p_str = str(raw_platform).lower().strip()
            platform_val = p_str if p_str in {p.value for p in Platform} else Platform.generic

        # 4. Extract media / images
        images = []
        if "images" in payload and isinstance(payload["images"], list):
            images = [str(img) for img in payload["images"]]
        elif "media" in payload and isinstance(payload["media"], dict):
            images = [str(img) for img in payload["media"].get("images", [])]
        elif "image_path" in payload and payload["image_path"]:
            images = [str(payload["image_path"])]

        # Determine PostType
        if images and normalized_text:
            post_type = PostType.image_text
        elif images:
            post_type = PostType.image
        else:
            post_type = PostType.text

        # 5. Extract author
        raw_author = payload.get("author")
        if isinstance(raw_author, dict):
            author = AuthorInfo(
                id=raw_author.get("id"),
                username=raw_author.get("username"),
                display_name=raw_author.get("display_name"),
                verified=bool(raw_author.get("verified", False)),
            )
        elif isinstance(raw_author, str):
            author = AuthorInfo(username=raw_author, display_name=raw_author)
        else:
            author = AuthorInfo()

        # 6. Extract timestamp
        ts = payload.get("timestamp") or datetime.now(timezone.utc).isoformat()

        # 7. Analyze language using L3
        lang_analysis = self.language_analyzer.analyze(normalized_text or original_text)
        primary_lang = payload.get("language") or lang_analysis.get("primary_language", "en")
        detected_langs = lang_analysis.get("detected_languages", [primary_lang])
        scripts = lang_analysis.get("scripts", ["Latin"])
        code_mixed = bool(lang_analysis.get("is_code_mixed", False))
        transliterated = bool(lang_analysis.get("transliteration_candidate", False))

        language_info = LanguageMetadata(
            primary=primary_lang,
            languages=detected_langs,
            script=scripts,
            code_mixed=code_mixed,
            transliterated=transliterated,
            normalized_text=normalized_text,
            original_text=original_text,
        )

        # 8. Build ImageMetadata
        images_meta = []
        for img in images:
            images_meta.append(
                ImageMetadata(
                    image_id=f"img_{uuid.uuid4().hex[:10]}",
                    image_path=img,
                )
            )

        # 9. Build TrainingPost
        return TrainingPost(
            post_id=post_id,
            platform=platform_val,
            post_type=post_type,
            source_url=payload.get("source_url") or payload.get("url"),
            author=author,
            content=PostContent(
                text=normalized_text,
                title=title,
                hashtags=payload.get("hashtags", []),
                mentions=payload.get("mentions", []),
                urls=payload.get("urls", []),
            ),
            media=PostMedia(images=images),
            timestamp=ts,
            metadata={
                "raw_id": raw_record.raw_id,
                "original_text": original_text,
                "source_format": raw_record.format,
                **payload.get("metadata", {}),
            },
            language_info=language_info,
            provenance=provenance,
            images_meta=images_meta,
            is_example=bool(payload.get("is_example", False)),
        )
