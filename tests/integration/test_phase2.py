import sys
import os

# Add to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.retrieval.bm25 import BM25Retriever
from app.retrieval.semantic_search import SemanticSearch
from app.retrieval.reranker import Reranker
from app.risk_engine.graph_model import GATModel
from app.risk_engine.feature_model import LightGBMModel
from app.risk_engine.fusion import RiskFusion
from app.risk_engine.calibration import Calibrator
from app.risk_engine.abstention import AbstentionController

def run_phase2():
    print("Running Phase 2 tests...")
    
    # 1. Retrieval
    bm25 = BM25Retriever()
    semantic = SemanticSearch()
    reranker = Reranker()
    
    candidates = bm25.retrieve("ABC Wealth guaranteed return")
    candidates += semantic.retrieve("ABC Wealth guaranteed return")
    ranked = reranker.rerank(candidates)
    
    assert len(ranked) == 2
    
    # 2. Risk Engine
    gat = GATModel()
    lgbm = LightGBMModel()
    fusion = RiskFusion()
    calibrator = Calibrator()
    abstention = AbstentionController()
    
    gat_score = gat.predict(None)
    lgbm_score = lgbm.predict(None)
    
    fused_score = fusion.fuse(gat_score, lgbm_score)
    calibrated_score = calibrator.calibrate(fused_score)
    
    decision = abstention.decide(calibrated_score, len(ranked))
    
    print(f"GAT Score: {gat_score}")
    print(f"LGBM Score: {lgbm_score}")
    print(f"Fused Score: {fused_score}")
    print(f"Calibrated Score: {calibrated_score}")
    print(f"Decision: {decision}")
    print("PHASE 2 PIPELINE EXECUTED SUCCESSFULLY")

if __name__ == "__main__":
    run_phase2()
