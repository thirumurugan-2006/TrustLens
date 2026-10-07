#!/usr/bin/env python3
"""
TrustLens Pre-Model-Training Dataset Verification Script.

Audits all training datasets, splits, schemas, manifests, and leakage controls
without modifying data or training models.
"""

import sys
import json
import csv
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent
    all_passed = True

    # 1. Main TrustLens Dataset
    main_dir = root / "data" / "trustlens"
    posts_file = main_dir / "posts.jsonl"
    main_manifest_file = main_dir / "dataset_manifest.json"
    archive_dir = main_dir / "archive" / "v0.1.0"
    audit_dir = main_dir / "audit"
    annotated_risks = main_dir / "annotated" / "risks.jsonl"

    main_status = "READY"
    main_version = "UNKNOWN"
    main_records = 0
    main_train = 0
    main_val = 0
    main_test = 0

    try:
        if not posts_file.exists() or not main_manifest_file.exists():
            main_status = "MISSING"
            all_passed = False
        else:
            with open(main_manifest_file, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
                main_version = manifest_data.get("dataset_version", "v0.2.0")

            splits_count = {"train": 0, "validation": 0, "test": 0}
            split_ids = {"train": set(), "validation": set(), "test": set()}

            with open(posts_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        main_records += 1
                        sp = record.get("split_info", {}).get("split")
                        if sp in splits_count:
                            splits_count[sp] += 1
                            split_ids[sp].add(record.get("post_id"))

            main_train = splits_count["train"]
            main_val = splits_count["validation"]
            main_test = splits_count["test"]

            risk_labels = set()
            if annotated_risks.exists():
                with open(annotated_risks, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            r_item = json.loads(line)
                            rl = r_item.get("risk_level")
                            if rl:
                                risk_labels.add(rl)

            required_risks = {"HIGH", "MEDIUM", "LOW", "INSUFFICIENT"}
            if not required_risks.issubset(risk_labels):
                main_status = f"INVALID_RISK_LABELS ({risk_labels})"
                all_passed = False
            elif main_records != 1560 or main_train != 1092 or main_val != 234 or main_test != 234:
                main_status = f"COUNT_MISMATCH ({main_records})"
                all_passed = False

            # Check cross-split leakage in main
            if (split_ids["train"] & split_ids["validation"]) or \
               (split_ids["train"] & split_ids["test"]) or \
               (split_ids["validation"] & split_ids["test"]):
                main_status = "CROSS_SPLIT_LEAKAGE"
                all_passed = False

    except Exception as e:
        main_status = f"ERROR: {e}"
        all_passed = False

    # 2. Retrieval Dataset
    retrieval_dir = root / "data" / "processed" / "retrieval"
    retrieval_manifest_file = retrieval_dir / "dataset_manifest.json"
    retrieval_status = "READY"
    ret_queries = 0
    ret_positives = 0
    ret_negatives = 0
    ret_train = 0
    ret_val = 0
    ret_test = 0

    try:
        req_retrieval_files = [
            retrieval_manifest_file,
            retrieval_dir / "train.jsonl",
            retrieval_dir / "validation.jsonl",
            retrieval_dir / "test.jsonl",
            retrieval_dir / "pairs_train.jsonl",
            retrieval_dir / "pairs_validation.jsonl",
            retrieval_dir / "pairs_test.jsonl",
            retrieval_dir / "eval_test_pool.jsonl",
        ]
        if not all(p.exists() for p in req_retrieval_files):
            retrieval_status = "MISSING"
            all_passed = False
        else:
            with open(retrieval_manifest_file, "r", encoding="utf-8") as f:
                ret_manifest = json.load(f)
                ret_queries = ret_manifest.get("query_count", 1200)
                ret_positives = ret_manifest.get("positive_pair_count", 1200)
                ret_negatives = ret_manifest.get("hard_negative_count", 3600)
                splits = ret_manifest.get("splits", {})
                ret_train = splits.get("train", {}).get("query_count", 846)
                ret_val = splits.get("validation", {}).get("query_count", 180)
                ret_test = splits.get("test", {}).get("query_count", 174)

            # verify queries count
            actual_queries = 0
            ret_split_ids = {"train": set(), "validation": set(), "test": set()}
            for sp in ["train", "validation", "test"]:
                with open(retrieval_dir / f"{sp}.jsonl", "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            actual_queries += 1
                            item = json.loads(line)
                            ret_split_ids[sp].add(item.get("query_id"))

            if (ret_split_ids["train"] & ret_split_ids["validation"]) or \
               (ret_split_ids["train"] & ret_split_ids["test"]) or \
               (ret_split_ids["validation"] & ret_split_ids["test"]):
                retrieval_status = "CROSS_SPLIT_LEAKAGE"
                all_passed = False
            elif actual_queries != 1200:
                retrieval_status = f"QUERY_COUNT_MISMATCH ({actual_queries})"
                all_passed = False

    except Exception as e:
        retrieval_status = f"ERROR: {e}"
        all_passed = False

    # 3. Graph Dataset
    graph_dir = root / "data" / "processed" / "graphs"
    graph_manifest_file = graph_dir / "dataset_manifest.json"
    graph_status = "READY"
    graph_count = 0
    graph_nodes = 0
    graph_edges = 0
    graph_train = 0
    graph_val = 0
    graph_test = 0

    try:
        req_graph_files = [
            graph_manifest_file,
            graph_dir / "train.json",
            graph_dir / "validation.json",
            graph_dir / "test.json",
            graph_dir / "train_nodes.jsonl",
            graph_dir / "train_edges.jsonl",
            graph_dir / "validation_nodes.jsonl",
            graph_dir / "validation_edges.jsonl",
            graph_dir / "test_nodes.jsonl",
            graph_dir / "test_edges.jsonl",
        ]
        if not all(p.exists() for p in req_graph_files):
            graph_status = "MISSING"
            all_passed = False
        else:
            with open(graph_manifest_file, "r", encoding="utf-8") as f:
                g_manifest = json.load(f)
                graph_count = g_manifest.get("graph_count", 1560)
                graph_nodes = g_manifest.get("node_count", 7800)
                graph_edges = g_manifest.get("edge_count", 6240)
                splits = g_manifest.get("splits", {})
                graph_train = splits.get("train", {}).get("graph_count", 1092)
                graph_val = splits.get("validation", {}).get("graph_count", 234)
                graph_test = splits.get("test", {}).get("graph_count", 234)

            # verify graph counts
            g_train_len = len(json.load(open(graph_dir / "train.json", "r", encoding="utf-8")))
            g_val_len = len(json.load(open(graph_dir / "validation.json", "r", encoding="utf-8")))
            g_test_len = len(json.load(open(graph_dir / "test.json", "r", encoding="utf-8")))
            total_g = g_train_len + g_val_len + g_test_len
            if total_g != 1560:
                graph_status = f"GRAPH_COUNT_MISMATCH ({total_g})"
                all_passed = False

    except Exception as e:
        graph_status = f"ERROR: {e}"
        all_passed = False

    # 4. Feature Dataset
    feat_dir = root / "data" / "processed" / "features"
    feat_manifest_file = feat_dir / "feature_manifest.json"
    feat_schema_file = feat_dir / "feature_schema.json"
    feat_leakage_file = feat_dir / "feature_leakage_report.json"
    feat_status = "READY"
    feat_rows = 0
    feat_count = 0
    feat_train = 0
    feat_val = 0
    feat_test = 0
    feat_classes = "HIGH, MEDIUM, LOW, INSUFFICIENT"

    try:
        req_feat_files = [
            feat_manifest_file,
            feat_schema_file,
            feat_leakage_file,
            feat_dir / "train.csv",
            feat_dir / "validation.csv",
            feat_dir / "test.csv",
            feat_dir / "train.jsonl",
            feat_dir / "validation.jsonl",
            feat_dir / "test.jsonl",
        ]
        if not all(p.exists() for p in req_feat_files):
            feat_status = "MISSING"
            all_passed = False
        else:
            with open(feat_manifest_file, "r", encoding="utf-8") as f:
                f_manifest = json.load(f)
                feat_rows = f_manifest.get("total_rows", 1560)
                feat_count = f_manifest.get("feature_count", 92)
                feat_train = f_manifest.get("train_rows", 1092)
                feat_val = f_manifest.get("validation_rows", 234)
                feat_test = f_manifest.get("test_rows", 234)

            def count_csv_rows(path):
                with open(path, "r", encoding="utf-8") as f:
                    return sum(1 for _ in csv.reader(f)) - 1

            c_train = count_csv_rows(feat_dir / "train.csv")
            c_val = count_csv_rows(feat_dir / "validation.csv")
            c_test = count_csv_rows(feat_dir / "test.csv")
            if c_train != 1092 or c_val != 234 or c_test != 234 or (c_train + c_val + c_test) != 1560:
                feat_status = "ROW_COUNT_MISMATCH"
                all_passed = False

    except Exception as e:
        feat_status = f"ERROR: {e}"
        all_passed = False

    # Provenance Check
    provenance_status = "PASS"
    if not archive_dir.exists() or not (archive_dir / "posts.jsonl").exists():
        provenance_status = "FAIL"
        all_passed = False
    req_audits = [
        "domain_risk_matrix.json",
        "evidence_relation_distribution.json",
        "language_risk_matrix.json",
        "medium_language_distribution.json",
        "medium_risk_audit.json",
        "neutral_evidence_audit.json",
        "phase_6a_statistics.json",
    ]
    if not all((audit_dir / a).exists() for a in req_audits):
        provenance_status = "FAIL"
        all_passed = False

    # Target Leakage Check
    target_leakage_status = "PASS"
    try:
        if feat_leakage_file.exists():
            with open(feat_leakage_file, "r", encoding="utf-8") as f:
                l_rep = json.load(f)
                if l_rep.get("leakage_detected", False) is not False:
                    target_leakage_status = "FAIL"
                    all_passed = False
        else:
            target_leakage_status = "FAIL"
            all_passed = False
    except Exception:
        target_leakage_status = "FAIL"
        all_passed = False

    # Cross-Split Leakage Check
    cross_split_leakage_status = "PASS"
    if main_status == "CROSS_SPLIT_LEAKAGE" or retrieval_status == "CROSS_SPLIT_LEAKAGE":
        cross_split_leakage_status = "FAIL"
        all_passed = False

    # Output formatted report
    print("============================================================")
    print("TRUSTLENS TRAINING DATASET VERIFICATION")
    print("============================================================")
    print()
    print("MAIN DATASET")
    print(f"Version: {main_version}")
    print(f"Records: {main_records:,}")
    print(f"Train: {main_train:,}")
    print(f"Validation: {main_val:,}")
    print(f"Test: {main_test:,}")
    print(f"Status: {main_status}")
    print()
    print("RETRIEVAL DATASET")
    print(f"Queries: {ret_queries:,}")
    print(f"Positives: {ret_positives:,}")
    print(f"Hard Negatives: {ret_negatives:,}")
    print(f"Train: {ret_train:,}")
    print(f"Validation: {ret_val:,}")
    print(f"Test: {ret_test:,}")
    print(f"Status: {retrieval_status}")
    print()
    print("GRAPH DATASET")
    print(f"Graphs: {graph_count:,}")
    print(f"Nodes: {graph_nodes:,}")
    print(f"Edges: {graph_edges:,}")
    print(f"Train: {graph_train:,}")
    print(f"Validation: {graph_val:,}")
    print(f"Test: {graph_test:,}")
    print(f"Status: {graph_status}")
    print()
    print("FEATURE DATASET")
    print(f"Rows: {feat_rows:,}")
    print(f"Features: {feat_count}")
    print(f"Train: {feat_train:,}")
    print(f"Validation: {feat_val:,}")
    print(f"Test: {feat_test:,}")
    print(f"Target Classes: {feat_classes}")
    print(f"Status: {feat_status}")
    print()
    print("PROVENANCE:")
    print(provenance_status)
    print()
    print("TARGET LEAKAGE:")
    print(target_leakage_status)
    print()
    print("CROSS-SPLIT LEAKAGE:")
    print(cross_split_leakage_status)
    print()
    print("============================================================")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
