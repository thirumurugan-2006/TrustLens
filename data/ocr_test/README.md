# OCR Benchmark Dataset

Place benchmark images in this directory and create `manifest.json` with one record per image:

```json
[
  {
    "image_id": "tamil_clean_001",
    "image": "images/tamil_clean_001.png",
    "ground_truth_text": "தமிழ் உரை",
    "language": "ta",
    "category": "clean",
    "optional_metadata": {}
  }
]
```

Required fields are `image`, `ground_truth_text`, `language`, and `category`. Do not use OCR output as ground truth. Samples without annotations are reported and excluded from CER, WER, and exact-match metrics.

Supported categories include `clean`, `blurry`, `noisy`, `low_resolution`, `mixed_tamil_english`, `stylized`, `multi_line`, `complex_background`, `numbers_symbols`, and `social_media`.
