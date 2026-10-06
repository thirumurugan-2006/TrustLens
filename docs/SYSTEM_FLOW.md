# System Flow

USER
 ↓
STREAMLIT (`frontend/streamlit_app.py`) (Status: IMPLEMENTED)
 ↓
INPUT
 ↓
NORMALIZATION (`app/input/schemas.py`)
 ↓
VALIDATION
 ↓
OCR (EasyOCR)
 ↓
LANGUAGE (`app/preprocessing/language_analyzer.py`)
 ↓
CLAIMS (`app/claims/extractor.py`)
 ↓
ATOMIC CLAIMS (`app/claims/decomposer.py`)
 ↓
QUERY (`app/query_synthesis/query_generator.py`)
 ↓
QUALITY GATE (`app/validation/semantic_quality_gate.py`)
 ↓
RETRIEVAL (`app/retrieval/bm25.py`) (Status: PLACEHOLDER)
 ↓
EVIDENCE (`app/evidence/schemas.py`)
 ↓
GRAPH (`app/evidence/graph_schema.py`)
 ↓
GAT (`app/risk_engine/graph_model.py`) (Status: PLACEHOLDER)
 ↓
FEATURE MODEL (`app/risk_engine/feature_model.py`) (Status: PLACEHOLDER)
 ↓
FUSION (`app/risk_engine/fusion.py`) (Status: IMPLEMENTED)
 ↓
CALIBRATION (`app/risk_engine/calibration.py`) (Status: IMPLEMENTED)
 ↓
ABSTENTION (`app/risk_engine/abstention.py`) (Status: IMPLEMENTED)
 ↓
EXPLANATION
 ↓
STREAMLIT RESULT
