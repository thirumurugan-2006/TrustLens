# TrustLens Repository Cleanup Report

## Cleanup Status

COMPLETE

## Reports Removed

33

List of removed report filenames:
- `ARCHITECTURE_AUDIT.md`
- `DATA_SPLIT_AND_LEAKAGE.md`
- `DEPENDENCIES.md`
- `FINAL_TRAINING_READINESS_REPORT.md`
- `IMPLEMENTATION_STATUS.md`
- `L2_VALIDATION_FIX_REPORT.md`
- `LABELING_GUIDELINES.md`
- `MODEL_INVENTORY.md`
- `MODEL_SELECTION.md`
- `PHASE_4A_DATA_PIPELINE.md`
- `PHASE_4A_REPORT.md`
- `PHASE_4B_DATASET_REPORT.md`
- `PHASE_4B_REPORT.md`
- `PHASE_5_REPORT.md`
- `PHASE_6A_REPORT.md`
- `PHASE_6B_REPORT.md`
- `PHASE_6B_RETRIEVAL_DATASET.md`
- `PHASE_6C_EVIDENCE_GRAPH_DATASET.md`
- `PHASE_6C_REPORT.md`
- `PHASE_6D_FEATURE_DATASET.md`
- `PHASE_6D_REPORT.md`
- `RUNNING.md`
- `SEMANTIC_VALIDATION_REPORT.md`
- `SYSTEM_FLOW.md`
- `TRAINING_DATA_SCHEMA.md`
- `TRAINING_EVALUATION_PROTOCOL.md`
- `TRAINING_READINESS.md`
- `TRAINING_READINESS_AUDIT.md`
- `TRAINING_READINESS_REPORT.md`
- `TRAINING_TASK_MAPPING.md`
- `claim_decomposition.md`
- `claim_extraction.md`
- `multilingual_representation.md`

## Temporary Files Removed

- 244 bytecode and temporary files (`*.pyc`, `tmp.png`)
- 60 cache directories (`__pycache__/`, `.pytest_cache/`)
- 2 empty experiment directories (`experiments/configs/`, `experiments/results/`)

## Duplicate Files Removed

0 (All dataset copies across `data/` audited; all files are verified canonical dataset components, version archives, or split partitions).

## Preserved Datasets

### Main Dataset
- **Version**: v0.2.0
- **Path**: `data/trustlens/`
- **Records**: 1,560
- **Train**: 1,092
- **Validation**: 234
- **Test**: 234
- **Risk Classes**: HIGH (990), MEDIUM (240), LOW (180), INSUFFICIENT (150)
- **Archive Version**: `data/trustlens/archive/v0.1.0/` (preserved)

### Retrieval Dataset
- **Version**: v0.1.0
- **Path**: `data/processed/retrieval/`
- **Queries**: 1,200
- **Positives**: 1,200
- **Hard Negatives**: 3,600
- **Cross-Language Pairs**: 800
- **Train**: 846 queries
- **Validation**: 180 queries
- **Test**: 174 queries
- **Evaluation Pool**: `data/processed/retrieval/eval_test_pool.jsonl` (preserved)

### Graph Dataset
- **Version**: v0.1.0
- **Path**: `data/processed/graphs/`
- **Graphs**: 1,560
- **Nodes**: 7,800
- **Edges**: 6,240
- **Train**: 1,092 graphs
- **Validation**: 234 graphs
- **Test**: 234 graphs

### Feature Dataset
- **Version**: v1.0.0
- **Path**: `data/processed/features/`
- **Rows**: 1,560
- **Features**: 92 (68 numerical, 2 categorical, 22 boolean across 10 groups)
- **Train**: 1,092 rows
- **Validation**: 234 rows
- **Test**: 234 rows
- **Target Classes**: HIGH, MEDIUM, LOW, INSUFFICIENT

## Dataset Integrity

### Content Hashes: UNCHANGED
| Dataset File | SHA-256 Digest | Status |
|---|---|---|
| `data/trustlens/posts.jsonl` | `b146e89fd7ef595c6e2ee89535282502fb60fb98964fed1c241c9f8cb4df1469` | UNCHANGED |
| `data/processed/retrieval/train.jsonl` | `581a0db88d4ab516b76b3cc1f311eac87a559299a29c120d5092b7639754160a` | UNCHANGED |
| `data/processed/retrieval/validation.jsonl` | `7c6ffa71b212328ba464765c5e52f782d6e8e32823008094451276eea83c3cbe` | UNCHANGED |
| `data/processed/retrieval/test.jsonl` | `65cd4682f318d6d7a4b8ff17aa4fbb5a8436f4938700f5e40fe73d29d66ea68d` | UNCHANGED |
| `data/processed/graphs/train.json` | `aeb498e3d85a9a4536084d6ca86bcf183e2ff7caa3749f8d7d444707fdec374e` | UNCHANGED |
| `data/processed/graphs/validation.json` | `9094e3402c9b9fb7ee2b44c27c4d372637a6eee552ff820f9367c0072b296971` | UNCHANGED |
| `data/processed/graphs/test.json` | `810d6d714882db652d66975ea95a18f560981ad9afb947a0bbacc3ffd0debb98` | UNCHANGED |
| `data/processed/features/train.csv` | `75817ab3ff3b984165461bab28e77711017b0a5d5b5ad4b9ca67184b48f2a411` | UNCHANGED |
| `data/processed/features/validation.csv` | `9897c435f209143260e0821f430bc08f4706545b61f06032a8bd2fa8a27eb992` | UNCHANGED |
| `data/processed/features/test.csv` | `6142673049a76843aa6e6aaec0ad91859535ba4bf1e4431cfd4f2e9dd200d788` | UNCHANGED |

- **Target leakage**: PASS (Zero leakage columns in feature matrix; verified via `feature_leakage_report.json`)
- **Cross-split leakage**: PASS (Strict campaign/source/translation group isolation verified across all splits)
- **Provenance**: PASS (Full v0.1.0 archive and Phase 6A remediation matrices verified intact)

## Machine-Readable Metadata Preserved

The following machine-readable manifests, schemas, and audit reports are preserved:
- `data/trustlens/dataset_manifest.json`
- `data/trustlens/source_registry.json`
- `data/trustlens/training_readiness_report.json`
- `data/trustlens/audit/domain_risk_matrix.json`
- `data/trustlens/audit/evidence_relation_distribution.json`
- `data/trustlens/audit/language_risk_matrix.json`
- `data/trustlens/audit/medium_language_distribution.json`
- `data/trustlens/audit/medium_risk_audit.json`
- `data/trustlens/audit/neutral_evidence_audit.json`
- `data/trustlens/audit/phase_6a_statistics.json`
- `data/trustlens/archive/v0.1.0/dataset_manifest.json`
- `data/trustlens/archive/v0.1.0/source_registry.json`
- `data/processed/retrieval/dataset_manifest.json`
- `data/processed/graphs/dataset_manifest.json`
- `data/processed/features/feature_schema.json`
- `data/processed/features/feature_manifest.json`
- `data/processed/features/feature_leakage_report.json`
- `data/schema/annotation.schema.json`
- `data/schema/atomic_claim.schema.json`
- `data/schema/claim.schema.json`
- `data/schema/dataset_manifest.schema.json`
- `data/schema/evidence.schema.json`
- `data/schema/examples.json`
- `data/schema/post.schema.json`
- `data/schema/risk.schema.json`
- `data/training/training_readiness_report.json`
- `training_readiness_matrix.json`

## Model Training Data Map

| Model Task | Task Code | Target Dataset | Dataset Modality & Format | Primary Target Labels |
|---|---|---|---|---|
| **Claim Detection** | T-1 | `data/trustlens/` | Text / JSONL (`posts.jsonl`, `annotated/claims.jsonl`) | Binary claim presence |
| **Claim Type Classification** | T-2 | `data/trustlens/` | Text / JSONL (`annotated/claims.jsonl`) | Claim taxonomy categories |
| **Atomic Claim Decomposition** | T-3 | `data/trustlens/` | Text / JSONL (`annotated/claims.jsonl`) | Claim frames & atomic components |
| **Cross-Lingual Evidence Retrieval** | T-4 | `data/processed/retrieval/` | Triplet / Pair JSONL (`pairs_*.jsonl`, `eval_test_pool.jsonl`) | Query-evidence relevance |
| **Stance / Relation Verification** | T-5 | `data/trustlens/` & `data/processed/retrieval/` | Text Pairs / JSONL (`annotated/evidence.jsonl`) | SUPPORTS, CONTRADICTS, NEUTRAL, INSUFFICIENT |
| **Evidence Verification (NLI)** | T-6 | `data/processed/retrieval/` | Dense Triplet Pairs (`pairs_*.jsonl`) | Binary / Triplet verification |
| **Evidence Graph Neural Network** | T-7 | `data/processed/graphs/` | Graph JSON & JSONL (`*.json`, `*_nodes.jsonl`, `*_edges.jsonl`) | Heterogeneous node/edge risk graph |
| **Tabular Risk Classification** | T-8 | `data/processed/features/` | Feature Matrix CSV & JSONL (`train.csv`, `validation.csv`, `test.csv`) | HIGH, MEDIUM, LOW, INSUFFICIENT |
| **Risk Calibration** | T-9 | Validation predictions | Post-training probability vectors | Expected Calibration Error (ECE) |
| **Selective Classification (Abstention)**| T-10 | Validation probabilities | Post-training uncertainty estimates | Risk vs. coverage trade-off |

## Tests

- **Previous Baseline**: 321 passed
- **New Tests**: 0
- **Total Tests**: 321
- **Passed**: 321
- **Failed**: 0
- **Skipped**: 0
- **Warnings**: 2

## Manual Training Readiness

- **Dataset**: READY
- **Model Training**: NOT STARTED

## Final Repository State

```
TrustLens/
├── app/
│   ├── dataset/
│   ├── validation/
│   ├── preprocessing/
│   ├── multilingual/
│   ├── claims/
│   ├── evidence/
│   ├── retrieval_dataset/
│   ├── graph_dataset/
│   ├── feature_dataset/
│   ├── features/
│   ├── risk_engine/
│   └── training/
├── artifacts/
│   ├── experiments/
│   ├── metrics/
│   ├── models/
│   │   ├── claim_detection/
│   │   ├── evidence_verification/
│   │   └── risk/
│   └── predictions/
├── data/
│   ├── processed/
│   │   ├── features/
│   │   ├── graphs/
│   │   └── retrieval/
│   ├── raw/
│   ├── schema/
│   ├── training/
│   └── trustlens/
│       ├── annotated/
│       ├── archive/v0.1.0/
│       ├── audit/
│       ├── test/
│       ├── train/
│       ├── validation/
│       ├── dataset_manifest.json
│       ├── posts.jsonl
│       └── source_registry.json
├── docs/
│   └── REPOSITORY_CLEANUP_REPORT.md
├── evaluation/
│   ├── datasets/
│   └── results/
├── experiments/
│   ├── model_comparison/
│   ├── multilingual/
│   ├── experiment_registry.csv
│   └── README.md
├── models/
│   ├── claim_detection/
│   ├── evidence_verification/
│   ├── graph/
│   ├── retrieval/
│   └── risk/
├── scripts/
│   ├── diagnostics/
│   ├── verify_training_datasets.py
│   └── ...
├── tests/
├── training_readiness_matrix.json
├── requirements.txt
├── requirements-optional.txt
├── .env.example
├── .gitignore
└── README.md
```
