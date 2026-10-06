# TrustLens Model Inventory

## 1. Model Summary

| Component | Model/Library | Version | Task | Actually Used? | Trainable? | Current Status |
|---|---|---|---|---|---|---|
| Language Detection | `langdetect` | Default | Primary language detection | YES | NO | Overridden by custom rule baseline |
| Script Detection | Custom Unicode Rules | N/A | Unicode script extraction | YES | NO | Rule-based baseline |
| OCR | `easyocr` | Unknown | Image Text Extraction | NO | NO | Not implemented fully in pipeline |
| Multilingual Embedding | `xlm-roberta-base` | Transformers | Sentence Representation | YES (Benchmark) | YES | Untrained, used for embedding only |
| Multilingual Embedding | `google/muril-base-cased` | Transformers | Sentence Representation | YES (Benchmark) | YES | Untrained, used for embedding only |
| Claim Extraction | Rule-based | N/A | Segmenting coordinated clauses | YES | NO | Rule-based baseline |
| Claim Decomposition | Rule-based | N/A | Extracting semantic frames | YES | NO | Rule-based baseline |
| Query Synthesis | Rule-based | N/A | Constructing structured queries | YES | NO | Rule-based baseline |
| Sentence Embedding | `multilingual-e5` | Unknown | Semantic similarity | NO | NO | NOT IMPLEMENTED |
| Reranker | Unknown | Unknown | Retrieval ranking | NO | NO | NOT IMPLEMENTED |
