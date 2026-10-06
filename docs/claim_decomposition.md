# Claim Decomposition (Step 12)

## 1. Why decomposition is required
Step 11 extracts assertions in bulk format from posts. However, a single sentence often contains multiple independent, verifiable claims. Attempting to verify a compound claim in one pass reduces precision. Decomposition breaks these compound claims into independently verifiable "atomic claims", maximizing accurate evidence retrieval.

## 2. Atomic claim definition
An atomic claim is a sub-unit of an extracted claim containing a single semantic assertion (ideally one primary subject and one primary predicate).

## 3. Parent-child relationship
Every `AtomicClaim` stores a `parent_claim_id`, ensuring clear provenance back to the Step 11 extracted claim.

## 4. Subject/predicate/object
Atomic claims aim to abstract the semantic triples (subject, predicate, object). In the rule-based approach, this requires NLP-heavy dependency parsing (like spaCy), so these fields are left empty/extensible to prevent hallucinated data.

## 5. Numbers and units
Atomic claims map inherited `metadata` arrays (e.g. `percentages`, `numbers`) to distinct `value` and `unit` fields when the target text matches.

## 6. Temporal expressions
Tokens like "monthly", "today", "yesterday", or "since X" are parsed and pushed into the `temporal_context` property, preserving time constraints on the claim.

## 7. Conditional claims
Sentences possessing conditional terms ("if") tag `conditional=True` in the metadata. 

## 8. Negation
Negative terms ("not", "never") populate `metadata['negation'] = True` to prevent verification engines from seeking positive confirmation.

## 9. Modality
Expressions of probability or requirement ("may", "must", "guaranteed") are captured in `metadata['modality']` to respect the claim's original strength.

## 10. Attribution
Phrases like "according to" or "claims that" correctly tag `metadata['attribution']`, ensuring down-stream systems know the post is merely reporting a claim, not asserting it as truth.

## 11. Multilingual decomposition
Tamil and Tanglish remain natively preserved. Tamil coordinates like "மற்றும்" (and) and "ஆனால்" (but) trigger splits, ensuring semantic accuracy across scripts.

## 12. Provenance
Because atomic claims carry the `parent_claim_id`, they inherit the exact `source_span` (start/end indexes) and `source_text` from the `Claim`, preserving byte-level traceability.

## 13. Limitations
The `RuleBasedDecomposer` utilizes regex splitting. While highly deterministic and fast, it struggles with highly complex, deeply nested clauses (e.g. comparative relative clauses). A `ModelBasedDecomposer` is supported by the schema for future implementations.

## 14. Why decomposition is not verification
Decomposition ONLY restructures syntax. It performs zero fact-checking, zero web searches, and outputs zero scam/risk scores.
