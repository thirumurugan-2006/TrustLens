"""
TrustLens OCR v2 — Image Preprocessor
=======================================

Generates multiple image preprocessing variants for OCR candidate generation.

Preprocessing variants
-----------------------
1. original          — image as-is (no changes)
2. upscale_2x        — bicubic 2× upscale (cv2.INTER_CUBIC)
3. grayscale         — BGR → GRAY on original
4. grayscale_upscale_2x — BGR → GRAY, then 2× upscale
5. contrast_enhanced — moderate CLAHE contrast on grayscale upscale
6. adaptive_threshold — Gaussian adaptive threshold on grayscale upscale

Rules
-----
- Do NOT apply aggressive morphological operations (erode/dilate).
  Tamil vowel/consonant marks (matras) can be destroyed by morphology.
- 2× is the default upscale factor.  4× / 5× are NOT applied by default.
- Adaptive threshold is an optional candidate; the selector decides if it wins.
- Each variant is independent — no variant overwrites another.
- Variants are returned as (name, numpy_array) tuples.

Configuration
-------------
upscale_factor : float   default 2.0
contrast_clip  : float   default 2.5  (CLAHE clipLimit)
contrast_grid  : int     default 8    (CLAHE tileGridSize)
threshold_block: int     default 31   (adaptive threshold block size — must be odd)
threshold_c    : int     default 11   (adaptive threshold C constant)
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class OCRPreprocessor:
    """
    Produces image preprocessing variants for the OCR candidate pipeline.

    Parameters
    ----------
    upscale_factor : float
        Multiplier applied to width and height during upscaling.
        Default 2.0.
    contrast_clip : float
        CLAHE clipLimit for contrast enhancement.  Lower = less aggressive.
        Default 2.5.
    contrast_grid : int
        CLAHE tileGridSize (square).  Default 8.
    threshold_block : int
        Adaptive threshold block size (must be odd, >= 3).  Default 31.
    threshold_c : int
        Adaptive threshold C constant.  Default 11.
    """

    def __init__(
        self,
        upscale_factor: float = 2.0,
        contrast_clip: float = 2.5,
        contrast_grid: int = 8,
        threshold_block: int = 31,
        threshold_c: int = 11,
    ) -> None:
        self.upscale_factor = upscale_factor
        self.contrast_clip = contrast_clip
        self.contrast_grid = contrast_grid
        self.threshold_block = threshold_block
        self.threshold_c = threshold_c

    # ------------------------------------------------------------------
    # Core building blocks
    # ------------------------------------------------------------------

    def _load_bgr(self, image_path: str | Path) -> np.ndarray:
        """Load image in BGR format.  Raises ValueError if unreadable."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")
        return img

    def _upscale(self, image: np.ndarray) -> np.ndarray:
        """Bicubic 2× upscale (high quality, preserves fine strokes)."""
        h, w = image.shape[:2]
        return cv2.resize(
            image,
            (int(w * self.upscale_factor), int(h * self.upscale_factor)),
            interpolation=cv2.INTER_CUBIC,
        )

    def _to_gray(self, bgr: np.ndarray) -> np.ndarray:
        """BGR → GRAY conversion."""
        if len(bgr.shape) == 2:
            return bgr  # already grayscale
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    def _clahe_enhance(self, gray: np.ndarray) -> np.ndarray:
        """
        CLAHE (Contrast-Limited Adaptive Histogram Equalization).
        Enhances local contrast while limiting noise amplification.
        This is much safer for Tamil glyphs than global histogram stretching.
        """
        clahe = cv2.createCLAHE(
            clipLimit=self.contrast_clip,
            tileGridSize=(self.contrast_grid, self.contrast_grid),
        )
        return clahe.apply(gray)

    def _adaptive_threshold(self, gray: np.ndarray) -> np.ndarray:
        """
        Gaussian adaptive thresholding.

        block_size must be odd — we enforce this.
        Safe for Tamil glyphs when block_size is large enough (>= 21).
        """
        block = self.threshold_block
        if block % 2 == 0:
            block += 1
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block,
            self.threshold_c,
        )

    # ------------------------------------------------------------------
    # Variant generation
    # ------------------------------------------------------------------

    def variants(self, image_path: str | Path) -> list[tuple[str, np.ndarray]]:
        """
        Return a list of (name, numpy_array) preprocessing variants.

        Variants generated:
          1. original
          2. upscale_2x
          3. grayscale
          4. grayscale_upscale_2x
          5. contrast_enhanced
          6. adaptive_threshold
        """
        bgr_orig = self._load_bgr(image_path)
        bgr_up   = self._upscale(bgr_orig)
        gray_up  = self._to_gray(bgr_up)
        gray_orig = self._to_gray(bgr_orig)

        return [
            ("original",              bgr_orig),
            ("upscale_2x",            bgr_up),
            ("grayscale",             gray_orig),
            ("grayscale_upscale_2x",  gray_up),
            ("contrast_enhanced",     self._clahe_enhance(gray_up)),
            ("adaptive_threshold",    self._adaptive_threshold(gray_up)),
        ]

    def variants_as_files(
        self,
        image_path: str | Path,
        tmp_dir: str | Path | None = None,
    ) -> list[tuple[str, Path]]:
        """
        Write preprocessing variants to temporary files and return
        (name, tmp_path) pairs.

        The caller is responsible for deleting the temporary files.

        Parameters
        ----------
        image_path : str | Path
            Source image.
        tmp_dir : str | Path | None
            Directory for temporary files.  Uses system temp if None.
        """
        results: list[tuple[str, Path]] = []
        for name, arr in self.variants(image_path):
            suffix = ".png"
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                dir=str(tmp_dir) if tmp_dir else None,
                delete=False,
            ) as f:
                tmp_path = Path(f.name)
            ok = cv2.imwrite(str(tmp_path), arr)
            if not ok:
                # Fallback: try original path for this variant
                tmp_path.unlink(missing_ok=True)
                continue
            results.append((name, tmp_path))
        return results

    def variant_names(self) -> list[str]:
        """Return the list of variant names (no image required)."""
        return [
            "original",
            "upscale_2x",
            "grayscale",
            "grayscale_upscale_2x",
            "contrast_enhanced",
            "adaptive_threshold",
        ]
