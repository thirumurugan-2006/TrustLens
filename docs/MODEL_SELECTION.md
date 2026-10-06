# Model Selection

Based on the Model Audit for Steps 11–13, the current model selection relies on a deterministic rule-based baseline.

## 1. Selected Model: Deterministic Baseline

The deterministic baseline was selected over fine-tuned XLM-R / MuRIL transformers for the following reasons:
1. **Semantic Performance:** Explicit rules correctly assign predicates and cross-reference conditions in financial guarantees (e.g., `invest -> receive`), whereas a purely pretrained transformer simply generates unguided embeddings without structural relational extraction.
2. **Multilingual Robustness:** Direct Unicode detection effectively categorizes Code-Mixed data (Tamil/Hindi + English) preventing the common failure mode of false English categorization.
3. **Training Data Deficiency:** We lack a validated, comprehensive dataset of labeled structural claims to appropriately fine-tune XLM-R or MuRIL for sequential semantic parsing.

## 2. XLM-R and MuRIL Context

Both models were profiled successfully.
- `xlm-roberta-base` successfully encodes English, Tamil, and Hindi.
- `google/muril-base-cased` also encodes all samples with slightly lower CPU latency.

However, neither model is selected for the *Claim Extraction and Decomposition* tasks since they are currently just untuned representation encoders, which cannot yet extract structured relations like `{subject: ABC, predicate: guarantees, value: 20}`.

**Conclusion:** The Deterministic Rule-Based Baseline remains the selected engine for Steps 11–13 until a sufficient labeled dataset enables task-specific fine-tuning.
