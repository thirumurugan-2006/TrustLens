class AbstentionController:
    def decide(self, calibrated_score, evidence_count):
        if evidence_count < 2:
            return "INSUFFICIENT_EVIDENCE"
        if calibrated_score > 0.85:
            return "HIGH"
        if calibrated_score < 0.3:
            return "LOW"
        return "MEDIUM"
