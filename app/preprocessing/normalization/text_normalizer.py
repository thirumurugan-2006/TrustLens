import time
from typing import List, Tuple

from app.vision.ocr.ocr_result import TamilOCRResult
from app.preprocessing.text.text_cleaner import (
    normalize_unicode,
    remove_control_characters,
    normalize_whitespace,
    normalize_punctuation
)

def _sort_regions(texts: List[str], bboxes: List[list], confs: List[float]) -> Tuple[List[str], List[list], List[float]]:
    """Sorts OCR regions from top-to-bottom, left-to-right based on bounding boxes."""
    if not bboxes or len(bboxes) != len(texts):
        return texts, bboxes, confs
        
    regions = []
    for i in range(len(texts)):
        # Calculate center of bounding box for sorting
        if bboxes[i] and len(bboxes[i]) == 4:
            # bbox is typically [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            y_center = (bboxes[i][0][1] + bboxes[i][2][1]) / 2.0
            x_center = (bboxes[i][0][0] + bboxes[i][1][0]) / 2.0
            regions.append((y_center, x_center, texts[i], bboxes[i], confs[i]))
        else:
            # Fallback if bbox is malformed
            regions.append((0, 0, texts[i], bboxes[i], confs[i]))
            
    # Sort by Y (top-to-bottom) with a tolerance, then by X (left-to-right)
    # A tolerance of 10-15 pixels groups items on the same line
    regions.sort(key=lambda r: (round(r[0] / 15.0), r[1]))
    
    return [r[2] for r in regions], [r[3] for r in regions], [r[4] for r in regions]

def _remove_duplicates(texts: List[str], bboxes: List[list], confs: List[float]) -> Tuple[List[str], List[list], List[float]]:
    """Conservatively removes exact duplicate regions."""
    if not texts:
        return texts, bboxes, confs
        
    unique_texts = []
    unique_bboxes = []
    unique_confs = []
    seen = set()
    
    for i, t in enumerate(texts):
        # We only consider exact matches as duplicates for now to be conservative
        t_stripped = t.strip()
        if t_stripped and t_stripped not in seen:
            seen.add(t_stripped)
            unique_texts.append(t)
            unique_bboxes.append(bboxes[i] if i < len(bboxes) else None)
            unique_confs.append(confs[i] if i < len(confs) else 0.0)
            
    return unique_texts, unique_bboxes, unique_confs

def normalize_ocr_result(result: TamilOCRResult) -> TamilOCRResult:
    """
    Main entry point for OCR normalization.
    Preserves raw_text, populates normalized_text.
    """
    started = time.perf_counter()
    
    # 1. Preserve raw text
    # In case it hasn't been set yet
    if not result.raw_text:
        result.raw_text = result.text
        
    if not result.raw_text:
        result.normalized_text = ""
        result.text = ""
        return result
        
    # Re-construct from regions if possible for sorting and deduplication
    # Wait, the regions are not stored in TamilOCRResult (only raw text and region_confs/bboxes)
    # If we don't have region texts, we just process the combined text.
    # Currently CombinedOCR joins texts into `result.text` before saving region texts.
    # We should normalize the combined text directly for now, or update CombinedOCR to pass region texts.
    
    raw_text = result.raw_text
    
    # 2. Unicode normalization
    norm_text = normalize_unicode(raw_text)
    
    # 3. Remove unsafe/control artifacts
    norm_text = remove_control_characters(norm_text)
    
    # 4. Punctuation normalization
    norm_text = normalize_punctuation(norm_text)
    
    # 5. Whitespace normalization
    norm_text = normalize_whitespace(norm_text)
    
    # Update result
    result.normalized_text = norm_text
    result.text = norm_text # Update the main text alias to the normalized one
    
    elapsed = (time.perf_counter() - started) * 1000
    # We don't overwrite processing_time_ms to keep OCR time, but we could add normalization_time_ms
    if not hasattr(result, "normalization_time_ms"):
        setattr(result, "normalization_time_ms", elapsed)
        
    return result
