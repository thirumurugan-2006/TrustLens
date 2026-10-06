# Multilingual Representation Layer

## 1. Why Multilingual Representation is Required
TrustLens is designed to evaluate evidence-based claims across English, Tamil, and Tanglish. Operating purely on string matching or monolingual logic fails when a claim is expressed across different scripts and languages. The multilingual representation layer projects text into a unified, high-dimensional semantic space where meanings align regardless of the source language. This enables downstream claim extraction, verification, and risk modeling to operate on meaning rather than raw text.

## 2. XLM-R Role
`xlm-roberta-base` serves as the primary multilingual encoder. Pretrained on 100 languages, it provides a robust baseline for mapping English and Tamil into the same vector space.

## 3. MuRIL Role
`google/muril-base-cased` serves as the Indic-language-oriented alternative. Pretrained explicitly on Indian languages and their transliterations, MuRIL provides an alternative representation layer optimized for the Indian linguistic context.

## 4. Pooling Strategy
Both models utilize **Masked Mean Pooling**. The output of the models comprises contextual embeddings for each token. By calculating the mean across all tokens (ignoring padding tokens via the attention mask), we derive a fixed-size semantic representation for the entire input sequence. This avoids arbitrary token extraction (like only taking the CLS token) and provides a more comprehensive semantic signal.

## 5. Embedding Dimensions
Both XLM-R and MuRIL produce embeddings with a dimension size of **768**.

## 6. Model Caching
Models are loaded **lazily** via the `MultilingualModelManager` and cached in memory. The models are loaded upon the first encoding request and persist in memory to avoid repetitive initialization delays across API calls.

## 7. CPU/GPU Behavior
The layer seamlessly supports both CPU and CUDA devices. It actively probes for GPU availability using `torch.cuda.is_available()` during model initialization and defaults to CPU execution if no GPU is found, preserving broad compatibility.

## 8. Code-mix Handling
Code-mixed text (e.g., Tamil script intermixed with English words) is processed directly by the models. The input remains untouched from the language detection phase, preserving the raw and normalized texts. The multilingual representation models map this code-mixed syntax into the unified semantic space without requiring enforced translation.

## 9. Tanglish Handling
Transliterated Tamil (Tanglish) remains in its Latin script format during representation. It is processed similarly to code-mix data. 

## 10. Current Limitations
- Pure transliteration normalization (e.g., IndicXlit) is not yet applied before representation.
- Semantic embeddings do not yet differentiate finely between nuanced deception constructs; they only represent textual meaning.

## 11. Why Embeddings are NOT Risk Predictions
Semantic embeddings represent **what** the text means, not its **veracity** or **intent**. High embedding similarity between two claims simply means they discuss the same topic. It does not imply that the text is a scam, nor does it guarantee the information is safe. Risk calibration is the responsibility of downstream evidence verification and risk engines (Step 11+).
