# Experiment EXP_001

## 1. Objective
To benchmark the pretrained XLM-R transformer for representation encoding across TrustLens language tiers (en, ta, hi, ta-en, Tanglish, Hinglish) and determine if it can replace the semantic baseline out-of-the-box.

## 2. Model
- Model: xlm-roberta-base
- Version: Pretrained
- Architecture: Transformer Encoder
- Parameters: ~278M
- Trainable parameters: 0 (Currently locked to eval/pretrained)
- Pretrained: YES
- Task: Sentence Representation (Embedding)

## 3. Dataset
- Dataset: evaluation/datasets/semantic_claims.jsonl
- Version: 1.0 (Seed Dataset)
- Total samples: 13
- Languages: en, ta, hi, ta-en, hi-en
- Train: 0
- Validation: 0
- Test: 13
- Class distribution: N/A

## 4. Configuration
- Epochs: N/A
- Batch size: 1
- Learning rate: N/A
- Optimizer: N/A
- Weight decay: N/A
- Scheduler: N/A
- Max sequence length: 512
- Dropout: 0.1
- Random seed: N/A

## 5. Hardware
- CPU: Standard Instance
- GPU: N/A
- RAM: Available
- Device: CPU
- Python: 3.13
- PyTorch: Current
- Transformers: Current

## 6. Training
- Training time: N/A
- Final training loss: N/A
- Validation loss: N/A

## 7. Evaluation Metrics
| Metric | Score |
|---|---:|
| Accuracy | N/A |
| Precision | N/A |
| Recall | N/A |
| Macro F1 | N/A |
| Weighted F1 | N/A |
| ECE | N/A |
| Brier Score | N/A |

*(Model currently only provides embeddings, thus semantic structural metrics cannot be generated directly without a task-specific fine-tuned head.)*

## 8. Per-Language Results
| Language | Samples | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| English | 9 | N/A | N/A | N/A |
| Tamil | 1 | N/A | N/A | N/A |
| Hindi | 1 | N/A | N/A | N/A |
| Tamil-English | 1 | N/A | N/A | N/A |
| Hinglish | 1 | N/A | N/A | N/A |

## 9. Error Analysis
- **false positives:** None (No classification output)
- **language failures:** None (Tokenizes successfully)
- **code-mix failures:** None (Tokenizes successfully)

## 10. Strengths
Successfully loads and processes all tested scripts natively including Devanagari and Tamil without crashing. High dimensional representation available immediately for downstream tasks.

## 11. Limitations
Without a tuned structural parsing head (or Sequence-to-Sequence head), XLM-R cannot decompose claims, extract entity values, or map financial condition relationships (`invest -> receive`). Output is only a 768-d semantic vector. Cannot currently beat the Rule-Based Baseline for structural Query Generation. High latency on CPU (150ms).

## 12. Reproducibility
- Git commit: HEAD
- configuration file: Default
- dataset version: Seed V1
- random seed: 0
- environment: TrustLens Standard Env

## 13. Decision
NEEDS FINE-TUNING
**Reasoning:** XLM-R represents a powerful embedding substrate but fundamentally lacks the structural extraction abilities out-of-the-box needed for Step 12/13. It cannot be promoted to the primary Claim Extractor without a massive annotated dataset (currently nonexistent) and a trained classification/extraction head.
