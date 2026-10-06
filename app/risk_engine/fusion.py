class RiskFusion:
    def fuse(self, gat_score, lgbm_score):
        # Average fusion
        return (gat_score + lgbm_score) / 2.0
