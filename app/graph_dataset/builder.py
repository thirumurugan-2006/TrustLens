"""
TrustLens Graph Dataset Builder CLI (Phase 6C).
Constructs and serializes heterogeneous evidence graphs from TrustLens data with dry-run support.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from app.graph_dataset.cluster_manager import GraphClusterManager
from app.graph_dataset.graph_builder import EvidenceGraphBuilder
from app.graph_dataset.manifest import GraphManifestBuilder
from app.graph_dataset.schemas import (
    EdgeType,
    HeterogeneousEvidenceGraph,
    NodeType,
)
from app.graph_dataset.serializer import GraphSerializer
from app.graph_dataset.splitter import GraphSplitter
from app.graph_dataset.validator import GraphDatasetValidator
from app.training.schemas import (
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


class GraphDatasetPipeline:
    """
    Orchestrates ingestion of TrustLens posts, claims, evidence, and risks to construct,
    split, validate, and serialize heterogeneous evidence graphs.
    """

    def __init__(self, data_dir: str = "data/trustlens", output_dir: str = "data/processed/graphs"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.builder = EvidenceGraphBuilder()
        self.serializer = GraphSerializer(str(self.output_dir))
        self.splitter = GraphSplitter()
        self.cluster_manager = GraphClusterManager()

    def load_data(self) -> Dict[str, Any]:
        """Loads posts, claims, evidence, and risks from the source dataset directory."""
        posts: List[TrainingPost] = []
        claims: List[TrainingClaim] = []
        evidence_list: List[TrainingEvidence] = []
        risks_by_post: Dict[str, TrainingRisk] = {}

        # 1. Load posts
        posts_file = self.data_dir / "posts.jsonl"
        if not posts_file.is_file():
            # Check train/val/test subdirectories
            for sname in ["train", "validation", "test"]:
                sp = self.data_dir / sname / "posts.jsonl"
                if sp.is_file():
                    with sp.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                posts.append(TrainingPost.model_validate(json.loads(line.strip())))
        else:
            with posts_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        posts.append(TrainingPost.model_validate(json.loads(line.strip())))

        # 2. Load claims
        claims_file = self.data_dir / "annotated" / "claims.jsonl"
        if not claims_file.is_file():
            for sname in ["train", "validation", "test"]:
                sc = self.data_dir / sname / "claims.jsonl"
                if sc.is_file():
                    with sc.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                claims.append(TrainingClaim.model_validate(json.loads(line.strip())))
        else:
            with claims_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        claims.append(TrainingClaim.model_validate(json.loads(line.strip())))

        # 3. Load evidence
        ev_file = self.data_dir / "annotated" / "evidence.jsonl"
        if not ev_file.is_file():
            for sname in ["train", "validation", "test"]:
                se = self.data_dir / sname / "evidence.jsonl"
                if se.is_file():
                    with se.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                evidence_list.append(TrainingEvidence.model_validate(json.loads(line.strip())))
        else:
            with ev_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        evidence_list.append(TrainingEvidence.model_validate(json.loads(line.strip())))

        # 4. Load risks
        risk_file = self.data_dir / "annotated" / "risks.jsonl"
        if not risk_file.is_file():
            for sname in ["train", "validation", "test"]:
                sr = self.data_dir / sname / "risks.jsonl"
                if sr.is_file():
                    with sr.open("r", encoding="utf-8") as f:
                        for line in f:
                            if line.strip():
                                r = TrainingRisk.model_validate(json.loads(line.strip()))
                                risks_by_post[r.post_id] = r
        else:
            with risk_file.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = TrainingRisk.model_validate(json.loads(line.strip()))
                        risks_by_post[r.post_id] = r

        return {
            "posts": posts,
            "claims": claims,
            "evidence": evidence_list,
            "risks_by_post": risks_by_post,
        }

    def build_all_graphs(self) -> List[HeterogeneousEvidenceGraph]:
        """Constructs discrete HeterogeneousEvidenceGraph instances for all posts."""
        data = self.load_data()
        posts = data["posts"]
        claims = data["claims"]
        evidence_list = data["evidence"]
        risks_by_post = data["risks_by_post"]

        claims_by_post: Dict[str, List[TrainingClaim]] = {}
        for c in claims:
            claims_by_post.setdefault(c.post_id, []).append(c)

        ev_by_claim: Dict[str, List[TrainingEvidence]] = {}
        for e in evidence_list:
            ev_by_claim.setdefault(e.claim_id, []).append(e)

        graphs: List[HeterogeneousEvidenceGraph] = []

        for p in posts:
            p_claims = claims_by_post.get(p.post_id, [])
            p_evidence: List[TrainingEvidence] = []
            for c in p_claims:
                p_evidence.extend(ev_by_claim.get(c.claim_id, []))

            risk_record = risks_by_post.get(p.post_id)
            g = self.builder.build_graph_for_post(p, p_claims, p_evidence, risk_record)
            graphs.append(g)

        return graphs

    def run(self, dry_run: bool = False) -> Dict[str, Any]:
        """Runs the graph dataset construction pipeline."""
        graphs = self.build_all_graphs()
        splits = self.splitter.partition_graphs(graphs)

        # Compute statistics
        node_counts = {t.value: 0 for t in NodeType}
        edge_counts = {e.value: 0 for e in EdgeType}
        clusters: Set[str] = set()

        total_nodes = 0
        total_edges = 0

        for g in graphs:
            clusters.add(g.cluster_id)
            for n in g.nodes:
                total_nodes += 1
                node_counts[n.node_type.value] = node_counts.get(n.node_type.value, 0) + 1
            for e in g.edges:
                total_edges += 1
                edge_counts[e.edge_type.value] = edge_counts.get(e.edge_type.value, 0) + 1

        print("\n================ TRUSTLENS GRAPH DATASET BUILDER ================")
        print(f"Mode:                 {'DRY RUN (No files written)' if dry_run else 'LIVE EXECUTION'}")
        print(f"Total Graphs:         {len(graphs)}")
        print(f"Total Nodes:          {total_nodes}")
        print(f"Total Edges:          {total_edges}")
        print(f"Canonical Clusters:   {len(clusters)}")
        print(f"Splits:               Train={len(splits['train'])}, Val={len(splits['validation'])}, Test={len(splits['test'])}")
        print("\nNode Counts by Type:")
        for t, c in node_counts.items():
            print(f"  - {t}: {c}")
        print("\nEdge Counts by Type:")
        for e, c in edge_counts.items():
            print(f"  - {e}: {c}")

        if dry_run:
            print("=================================================================\n")
            return {
                "dry_run": True,
                "graphs": len(graphs),
                "nodes": total_nodes,
                "edges": total_edges,
                "node_counts": node_counts,
                "edge_counts": edge_counts,
                "splits": {k: len(v) for k, v in splits.items()},
            }

        # Live Execution: serialize files
        self.serializer.serialize_all(splits)

        # Validate
        validator = GraphDatasetValidator(str(self.output_dir))
        val_report = validator.validate()

        # Build and write manifest
        manifest_builder = GraphManifestBuilder()
        manifest = manifest_builder.build_manifest(graphs, val_report)
        manifest_path = self.output_dir / "dataset_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            f.write(manifest.model_dump_json(indent=2))

        print(f"\nValidation Result:    {'PASSED (VALID)' if val_report['is_valid'] else 'FAILED (INVALID)'}")
        print(f"Cross-split leakage:  {val_report.get('cross_split_leakage', 0)}")
        print(f"Manifest written:     {manifest_path}")
        print("=================================================================\n")

        return {
            "dry_run": False,
            "graphs": len(graphs),
            "nodes": total_nodes,
            "edges": total_edges,
            "manifest": manifest.model_dump(),
            "validation": val_report,
        }


def main():
    parser = argparse.ArgumentParser(description="TrustLens Evidence Graph Dataset Builder")
    parser.add_argument("--data-dir", default="data/trustlens", help="Source TrustLens directory")
    parser.add_argument("--output-dir", default="data/processed/graphs", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without writing files")
    args = parser.parse_args()

    pipeline = GraphDatasetPipeline(data_dir=args.data_dir, output_dir=args.output_dir)
    res = pipeline.run(dry_run=args.dry_run)
    sys.exit(0)


if __name__ == "__main__":
    main()
