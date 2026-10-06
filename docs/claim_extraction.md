# Claim Extraction Layer (Step 11)

## 1. What is a claim?
In TrustLens, a "claim" is an assertion, instruction, or request found within a social media post that is structured into an auditable representation. Extracting claims allows downstream systems to process independent statements rather than entire noisy paragraphs.

## 2. Claim vs Opinion
- **Claim:** "This company guarantees 30% monthly returns." (Factual, Verifiable)
- **Opinion:** "I think this looks suspicious." (Subjective, Not objectively verifiable)
Opinions are identified and classified differently from factual guarantees.

## 3. Claim vs Request
- **Request:** "DM me for details." (Action-oriented)
- **Claim:** "You will earn money." (Outcome-oriented)
Requests do not represent facts to be verified.

## 4. Claim vs Question
- **Question:** "Is this legitimate?"
Questions are not factual assertions. They are tagged as `QUESTION`.

## 5. Claim vs Scam Classification
**CRITICAL:** The claim extraction layer does **NOT** classify a post as a scam. It does not treat words like "investment", "guarantee", or "money" as fraud indicators. It simply extracts them as financial claims for downstream processing. Extraction confidence is about the extraction itself, not veracity.

## 6. Claim Schema
A extracted `Claim` contains:
- `claim_id`
- `post_id`
- `text` (Original text)
- `normalized_text`
- `claim_type` (e.g. FACTUAL, FINANCIAL, GUARANTEE, REQUEST, INSTRUCTION, OPINION, QUESTION, CONTEXTUAL)
- `language`
- `source_span` (Start and end index in original post)
- `extraction_method` (e.g. `rule_based`)
- `extraction_confidence`
- `verifiable` (Boolean indicating if it can be checked for facts)
- `metadata` (Numbers, currencies, percentages)

## 7. Extraction Pipeline
Text -> Sentence Segmentation -> Metadata Extraction (Numbers/Currencies) -> Type Classification -> Normalization -> Deduplication -> Output Claims

## 8. Multilingual Handling
Claims are extracted exactly as they are written in English or Tamil. The original language is preserved. 

## 9. Code-mix Handling
Code-mixed sentences (Tamil + English) are extracted as single coherent claims. The script variations do not artificially fracture the sentence.

## 10. Provenance
Every claim preserves a `source_span` linking it directly back to the original index in the raw text, ensuring complete audibility for final reports.

## 11. Confidence Meaning
`extraction_confidence` indicates the system's certainty that the extracted string is a well-formed claim. It does NOT mean "probability of being true" or "probability of being a scam".

## 12. Current Limitations
The current `RuleBasedClaimExtractor` uses regex-based classification which might struggle with highly complex, nuanced sarcasm or nested clauses. An LLM-based extractor can be substituted later.
