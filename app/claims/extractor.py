import re
from typing import List, Dict, Any
from app.claims.schemas import Claim, SourceSpan
from app.input.schemas import NormalizedPost

class RuleBasedClaimExtractor:
    def __init__(self):
        self.patterns = {
            "QUESTION": r"(?:\?|என்ன|எப்படி|எங்கே|யார்|\bis\b.*\?|\bdoes\b.*\?|\bdo\b.*\?|\bcan\b.*\?)",
            "GUARANTEE": r"(?i)\b(guarantee|guarantees|assured|100%|sure|உறுதி|கண்டிப்பாக|நிச்சயம்)\b",
            "FINANCIAL": r"(?i)(₹|\$|earn|earns|profit|return|returns|money|pay|fee|income|investment|invest|receive|லாபம்|வருமானம்|பணம்|முதலீடு)",
            "REQUEST": r"(?i)\b(dm me|message me|send|provide|give|whatsapp me|அனுப்பு|தொடர்பு)\b",
            "INSTRUCTION": r"(?i)\b(apply|click|register|join|share|பதிவு|கிளிக்)\b",
            "OPINION": r"(?i)\b(i think|i feel|maybe|probably|நினைக்கிறேன்|தோன்றுகிறது)\b",
            "CONTEXTUAL": r"(?i)\b(received|saw|found|message yesterday|கிடைத்தது|பார்த்தேன்)\b",
            "NUMERICAL": r"(?i)\b(customers|users|people|employees)\b",
            "FACTUAL": r"(?i)\b(founded|headquartered_in|headquarters|approved|registered|located|owns|operates|has)\b"
        }
        self.num_pattern = r"\b\d+(?:,\d+)*(?:\.\d+)?(?:k|m|b)?\b"
        self.currency_pattern = r"(?:₹|\$|rs\.?|rupees?|dollars?)\s*\d+(?:,\d+)*(?:\.\d+)?(?:k|m|b)?|\d+(?:,\d+)*(?:\.\d+)?(?:k|m|b)?\s*(?:₹|\$|rs\.?|rupees?|dollars?)"
        self.percent_pattern = r"\d+(?:\.\d+)?\s*%"

    def _segment_sentences(self, text: str) -> List[str]:
        if not text:
            return []
        
        # Initial sentence split
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        sentences = []
        for line in lines:
            parts = re.split(r'(?<=[.!?])\s+', line)
            for part in parts:
                if part.strip():
                    sentences.append(part.strip())
                    
        # Sub-clause split for coordinated sentences
        # e.g., "A was founded in 2018, has 50,000 customers, is government approved, and guarantees 20% monthly returns."
        clauses = []
        for sent in sentences:
            # We want to preserve the subject if it exists at the start
            subject = ""
            m = re.match(r"(?i)^([A-Z][a-zA-Z\s]+?)\s+(?:was|has|is|guarantees)", sent)
            if m:
                subject = m.group(1).strip()
            
            sub_parts = re.split(r'(?i),\s+(?:and\s+)?|\s+and\s+(?=(?:has|is|guarantees|was)\b)', sent)
            
            # Prevent splitting conditionals or attributions
            if re.match(r"(?i)^(if|according to|due to|because)\b", sent):
                clauses.append(sent)
                continue
                
            if len(sub_parts) > 1:
                for i, p in enumerate(sub_parts):
                    p = p.strip()
                    if not p: continue
                    # Re-inject subject if this clause starts with a naked predicate
                    if i > 0 and subject and re.match(r"(?i)^(has|is|was|guarantees)\b", p):
                        p = f"{subject} {p}"
                    clauses.append(p)
            else:
                clauses.append(sent)
                
        return clauses

    def _extract_metadata(self, text: str) -> Dict[str, Any]:
        numbers = re.findall(self.num_pattern, text, re.IGNORECASE)
        currencies = re.findall(self.currency_pattern, text, re.IGNORECASE)
        percentages = re.findall(self.percent_pattern, text)
        return {
            "numbers": list(set(numbers)),
            "currency": list(set(currencies)),
            "percentages": list(set(percentages))
        }

    def _classify_claim(self, text: str, meta: Dict[str, Any]) -> tuple[str, bool]:
        if re.search(self.patterns["QUESTION"], text):
            return "QUESTION", False
        if re.search(self.patterns["OPINION"], text):
            return "OPINION", False
        if re.search(self.patterns["REQUEST"], text):
            return "REQUEST", False
        if re.search(self.patterns["INSTRUCTION"], text):
            return "INSTRUCTION", False
        if re.search(self.patterns["CONTEXTUAL"], text):
            return "CONTEXTUAL", False
        if re.search(self.patterns["GUARANTEE"], text):
            return "GUARANTEE", True
        if re.search(self.patterns["FINANCIAL"], text):
            return "FINANCIAL", True
        if re.search(self.patterns["NUMERICAL"], text) and (meta["numbers"] or meta["percentages"]):
            return "NUMERICAL", True
        if re.search(self.patterns["FACTUAL"], text):
            return "FACTUAL", True
        if meta["numbers"] or meta["currency"] or meta["percentages"]:
            return "NUMERICAL", True
        return "FACTUAL", True

    def extract_from_post(self, post: NormalizedPost) -> List[Claim]:
        if not post or not post.text or not post.text.strip():
            return []
        sentences = self._segment_sentences(post.text)
        claims = []
        seen_normalized = set()
        start_idx = 0
        for i, sent in enumerate(sentences):
            norm_text = re.sub(r'\s+', ' ', sent).strip()
            if norm_text.lower() in seen_normalized:
                continue
            seen_normalized.add(norm_text.lower())
            sent_start = post.text.find(sent, start_idx)
            if sent_start != -1:
                sent_end = sent_start + len(sent)
                start_idx = sent_end
            else:
                sent_start = 0
                sent_end = len(sent)
                
            meta = self._extract_metadata(sent)
            claim_type, verifiable = self._classify_claim(sent, meta)
            
            # Simple attribution extraction
            attribution = None
            attr_match = re.search(r"(?i)(according to|stated by|claimed by)\s+([^,.]+)", sent)
            if attr_match:
                attribution = {"source": attr_match.group(2).strip(), "type": "attributed_claim"}
                meta["attribution"] = attribution

            lang = post.metadata.get("language_analysis", {}).get("primary_language", "unknown")
            claim = Claim(
                post_id=post.post_id,
                text=sent,
                normalized_text=norm_text,
                claim_type=claim_type,
                language=lang,
                source_span=SourceSpan(start=sent_start, end=sent_end, source_sentence=sent, source_index=i),
                verifiable=verifiable,
                metadata=meta
            )
            claims.append(claim)
        return claims
