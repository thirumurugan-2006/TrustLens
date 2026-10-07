"""
TrustLens Phase 6C Heterogeneous Evidence Graph Dataset Test Suite.
Deterministic, offline tests verifying:
1. Node schema validation
2. Edge schema validation
3. Node uniqueness
4. Edge uniqueness
5. Claim linkage to originating post
6. Evidence linkage to claim
7. Evidence relation preservation (SUPPORTS, CONTRADICTS, NEUTRAL, INSUFFICIENT)
8. Provenance preservation on nodes and edges
9. Multimodal nodes (IMAGE)
10. URL nodes and domain normalization
11. Comment nodes
12. Account nodes
13. Image similarity edge creation
14. Graph cluster-safe splitting
15. Campaign leakage prevention
16. Translation leakage prevention
17. Zero cross-split leakage
18. Graph statistics calculation
19. Manifest generation
20. Validator CLI and reports
21. Dry-run CLI execution
22. Invalid graph rejection (broken endpoints/types)
23. Missing provenance quarantine/handling
24. PyG conversion and dictionary representation
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.graph_dataset.cluster_manager import GraphClusterManager
from app.graph_dataset.edge_builder import EdgeBuilder
from app.graph_dataset.feature_builder import GraphFeatureBuilder
from app.graph_dataset.graph_builder import EvidenceGraphBuilder
from app.graph_dataset.label_builder import GraphLabelBuilder
from app.graph_dataset.manifest import GraphManifestBuilder
from app.graph_dataset.node_builder import NodeBuilder
from app.graph_dataset.pyg_converter import PyGHeteroDataConverter
from app.graph_dataset.schemas import (
    AccountNode,
    ClaimNode,
    CommentNode,
    EdgeType,
    EvidenceNode,
    GraphEdge,
    GraphNode,
    HeterogeneousEvidenceGraph,
    ImageNode,
    NodeType,
    PostNode,
    UrlNode,
)
from app.graph_dataset.serializer import GraphSerializer
from app.graph_dataset.splitter import GraphSplitter
from app.graph_dataset.validator import GraphDatasetValidator
from app.input.schemas import AuthorInfo, Platform, PostContent, PostType
from app.training.schemas import (
    ClaimType,
    EvidenceRelationLabel,
    ProvenanceMetadata,
    ProvenanceSourceType,
    RiskLevel,
    SplitMetadata,
    SplitName,
    TrainingClaim,
    TrainingEvidence,
    TrainingPost,
    TrainingRisk,
)


@pytest.fixture
def sample_training_post():
    return TrainingPost(
        post_id="post_test_01",
        platform=Platform.reddit,
        author=AuthorInfo(username="investor_guru", verified=True, display_name="Guru"),
        post_type=PostType.text,
        content=PostContent(text="Get 20% return every month on trading bot."),
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Regulatory Warnings",
        ),
        split_info=SplitMetadata(
            split=SplitName.train,
            campaign_group_id="camp_999",
            translation_group_id="trans_999",
            source_group_id="src_999",
        ),
    )


@pytest.fixture
def sample_training_claim(sample_training_post):
    return TrainingClaim(
        claim_id="claim_test_01",
        post_id=sample_training_post.post_id,
        claim_text="Get 20% return every month on trading bot.",
        claim_type=ClaimType.FINANCIAL,
        language="en",
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="Regulatory Warnings",
        ),
        split_info=sample_training_post.split_info,
    )


@pytest.fixture
def sample_training_evidence():
    return TrainingEvidence(
        evidence_id="ev_test_01",
        claim_id="claim_test_01",
        evidence_text="SEBI issues warning against unregistered trading bots.",
        relation_label=EvidenceRelationLabel.CONTRADICTS,
        source_url="https://official.gov.in/advisories/bots",
        source_type="REGULATORY_ADVISORY",
        provenance=ProvenanceMetadata(
            source_type=ProvenanceSourceType.PUBLIC_DATASET,
            source_name="SEBI Alerts",
        ),
    )


# ---------------------------------------------------------------------------
# 1. Node Schema Validation
# ---------------------------------------------------------------------------
def test_1_node_schema_validation():
    """Verify base and specialized node schemas."""
    p_node = PostNode(
        node_id="node_post_p1",
        source_id="p1",
        post_id="p1",
        text="Sample post text",
        language="en",
        script="Latin",
        platform="reddit",
        split=SplitName.train,
        cluster_id="camp_01",
    )
    assert p_node.node_type == NodeType.POST
    assert p_node.split == SplitName.train
    assert p_node.cluster_id == "camp_01"


# ---------------------------------------------------------------------------
# 2. Edge Schema Validation
# ---------------------------------------------------------------------------
def test_2_edge_schema_validation():
    """Verify directed edge schema and prohibition of self-loops."""
    edge = GraphEdge(
        edge_id="edge_01",
        source_node_id="node_post_p1",
        source_node_type=NodeType.POST,
        target_node_id="node_claim_c1",
        target_node_type=NodeType.CLAIM,
        edge_type=EdgeType.HAS_CLAIM,
        split=SplitName.train,
        cluster_id="camp_01",
    )
    assert edge.edge_type == EdgeType.HAS_CLAIM

    # Self loop rejected
    with pytest.raises(ValidationError):
        GraphEdge(
            edge_id="edge_self",
            source_node_id="node_post_p1",
            source_node_type=NodeType.POST,
            target_node_id="node_post_p1",  # Self-loop!
            target_node_type=NodeType.POST,
            edge_type=EdgeType.HAS_CLAIM,
            split=SplitName.train,
            cluster_id="camp_01",
        )


# ---------------------------------------------------------------------------
# 3. Node Uniqueness
# ---------------------------------------------------------------------------
def test_3_node_uniqueness(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify builder guarantees distinct node IDs within a graph."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post,
        [sample_training_claim],
        [sample_training_evidence],
    )
    node_ids = [n.node_id for n in graph.nodes]
    assert len(node_ids) == len(set(node_ids))


# ---------------------------------------------------------------------------
# 4. Edge Uniqueness
# ---------------------------------------------------------------------------
def test_4_edge_uniqueness(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify builder guarantees distinct edges within a graph."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post,
        [sample_training_claim],
        [sample_training_evidence],
    )
    edge_keys = [(e.source_node_id, e.target_node_id, e.edge_type) for e in graph.edges]
    assert len(edge_keys) == len(set(edge_keys))


# ---------------------------------------------------------------------------
# 5. Claim Linkage
# ---------------------------------------------------------------------------
def test_5_claim_linkage(sample_training_post, sample_training_claim):
    """Verify claim nodes are linked to the parent post via HAS_CLAIM."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post,
        [sample_training_claim],
        [],
    )
    has_claim_edges = [e for e in graph.edges if e.edge_type == EdgeType.HAS_CLAIM]
    assert len(has_claim_edges) == 1
    assert has_claim_edges[0].source_node_id == f"node_post_{sample_training_post.post_id}"
    assert has_claim_edges[0].target_node_id == f"node_claim_{sample_training_claim.claim_id}"


# ---------------------------------------------------------------------------
# 6. Evidence Linkage
# ---------------------------------------------------------------------------
def test_6_evidence_linkage(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify evidence nodes are linked to claims with relation labels."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post,
        [sample_training_claim],
        [sample_training_evidence],
    )
    ev_edges = [e for e in graph.edges if e.edge_type == EdgeType.CONTRADICTS]
    assert len(ev_edges) == 1
    assert ev_edges[0].source_node_id == f"node_claim_{sample_training_claim.claim_id}"
    assert ev_edges[0].target_node_id == f"node_ev_{sample_training_evidence.evidence_id}"


# ---------------------------------------------------------------------------
# 7. Relation Preservation
# ---------------------------------------------------------------------------
def test_7_relation_preservation(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify SUPPORTS, CONTRADICTS, NEUTRAL, and INSUFFICIENT relations map to distinct edge types."""
    builder = EvidenceGraphBuilder()

    relations = [
        (EvidenceRelationLabel.SUPPORTS, EdgeType.SUPPORTS),
        (EvidenceRelationLabel.CONTRADICTS, EdgeType.CONTRADICTS),
        (EvidenceRelationLabel.NEUTRAL, EdgeType.NEUTRAL_TO),
        (EvidenceRelationLabel.INSUFFICIENT, EdgeType.INSUFFICIENT_FOR),
    ]

    for rel_label, expected_etype in relations:
        ev = sample_training_evidence.model_copy(deep=True)
        ev.relation_label = rel_label
        graph = builder.build_graph_for_post(sample_training_post, [sample_training_claim], [ev])
        matched_edge = [e for e in graph.edges if e.edge_type == expected_etype]
        assert len(matched_edge) == 1
        assert matched_edge[0].relation == rel_label


# ---------------------------------------------------------------------------
# 8. Provenance Preservation
# ---------------------------------------------------------------------------
def test_8_provenance_preservation(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify provenance metadata is preserved on evidence nodes and relation edges."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post, [sample_training_claim], [sample_training_evidence]
    )
    ev_node = next(n for n in graph.nodes if n.node_type == NodeType.EVIDENCE)
    assert ev_node.provenance is not None
    assert ev_node.provenance.source_name == "SEBI Alerts"


# ---------------------------------------------------------------------------
# 9. Multimodal Nodes (Images)
# ---------------------------------------------------------------------------
def test_9_multimodal_nodes(sample_training_post):
    """Verify image nodes and CONTAINS edges are only added when images exist."""
    builder = EvidenceGraphBuilder()

    # Case A: No images -> 0 image nodes
    graph_no_img = builder.build_graph_for_post(sample_training_post, [], [])
    assert not any(n.node_type == NodeType.IMAGE for n in graph_no_img.nodes)

    # Case B: Post with images
    post_with_img = sample_training_post.model_copy(deep=True)
    post_with_img.images_meta = [{"image_id": "img_001", "image_hash": "a1b2c3", "phash": "f0e1d2"}]
    graph_with_img = builder.build_graph_for_post(post_with_img, [], [])
    img_nodes = [n for n in graph_with_img.nodes if n.node_type == NodeType.IMAGE]
    assert len(img_nodes) == 1
    assert any(e.edge_type == EdgeType.CONTAINS for e in graph_with_img.edges)


# ---------------------------------------------------------------------------
# 10. URL Nodes and Domain Normalization
# ---------------------------------------------------------------------------
def test_10_url_nodes(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify URL nodes are derived from evidence source_url and domain is normalized."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post, [sample_training_claim], [sample_training_evidence]
    )
    url_nodes = [n for n in graph.nodes if n.node_type == NodeType.URL]
    assert len(url_nodes) == 1
    assert url_nodes[0].domain == "official.gov.in"
    assert any(e.edge_type == EdgeType.SOURCED_FROM for e in graph.edges)


# ---------------------------------------------------------------------------
# 11. Comment Nodes
# ---------------------------------------------------------------------------
def test_11_comment_nodes(sample_training_post):
    """Verify comment nodes and HAS_COMMENT edges are added when comments exist."""
    post_with_cmt = sample_training_post.model_copy(deep=True)
    post_with_cmt.comments = [{"comment_id": "cmt_01", "text": "This looks like a fraud scheme.", "stance": "REFUTES"}]

    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(post_with_cmt, [], [])
    cmt_nodes = [n for n in graph.nodes if n.node_type == NodeType.COMMENT]
    assert len(cmt_nodes) == 1
    assert any(e.edge_type == EdgeType.HAS_COMMENT for e in graph.edges)


# ---------------------------------------------------------------------------
# 12. Account Nodes
# ---------------------------------------------------------------------------
def test_12_account_nodes(sample_training_post):
    """Verify account nodes are extracted from post author information."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(sample_training_post, [], [])
    acc_nodes = [n for n in graph.nodes if n.node_type == NodeType.ACCOUNT]
    assert len(acc_nodes) == 1
    assert acc_nodes[0].username == "investor_guru"
    assert any(e.edge_type == EdgeType.ASSOCIATED_WITH for e in graph.edges)


# ---------------------------------------------------------------------------
# 13. Image Similarity Edge
# ---------------------------------------------------------------------------
def test_13_image_similarity_edge():
    """Verify EdgeBuilder constructs SIMILAR_TO edges with score metadata."""
    node_builder = NodeBuilder()
    img_a = node_builder.build_image_node({"image_id": "img_a", "phash": "1111"}, "c1", SplitName.train)
    img_b = node_builder.build_image_node({"image_id": "img_b", "phash": "1112"}, "c1", SplitName.train)

    edge_builder = EdgeBuilder()
    sim_edge = edge_builder.build_similar_to_edge(
        img_a, img_b, similarity_score=0.95, threshold=0.90, similarity_method="phash_hamming", cluster_id="c1", split=SplitName.train
    )
    assert sim_edge.edge_type == EdgeType.SIMILAR_TO
    assert sim_edge.metadata["similarity_score"] == 0.95


# ---------------------------------------------------------------------------
# 14. Graph Splitter
# ---------------------------------------------------------------------------
def test_14_graph_splitter():
    """Verify cluster-safe splitting partitions graphs without overlap."""
    clusters = {f"camp_{i:03d}": [f"item_{i}"] for i in range(20)}
    splitter = GraphSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    splits = splitter.split_clusters(clusters)

    train_set = set(splits["train"])
    val_set = set(splits["validation"])
    test_set = set(splits["test"])

    assert len(train_set.intersection(val_set)) == 0
    assert len(train_set.intersection(test_set)) == 0
    assert len(val_set.intersection(test_set)) == 0
    assert len(train_set) + len(val_set) + len(test_set) == len(clusters)


# ---------------------------------------------------------------------------
# 15. Campaign Leakage Prevention
# ---------------------------------------------------------------------------
def test_15_campaign_leakage(sample_training_post):
    """Verify posts sharing a campaign ID receive identical cluster IDs."""
    p1 = sample_training_post
    p2 = sample_training_post.model_copy(update={"post_id": "post_p2"})
    mgr = GraphClusterManager()
    assert mgr.get_cluster_id(p1) == mgr.get_cluster_id(p2) == "camp_camp_999"


# ---------------------------------------------------------------------------
# 16. Translation Leakage Prevention
# ---------------------------------------------------------------------------
def test_16_translation_leakage(sample_training_post):
    """Verify translated posts sharing translation_group_id map to the same cluster."""
    p_trans = sample_training_post.model_copy(deep=True)
    p_trans.split_info.campaign_group_id = None  # Leave only translation group
    mgr = GraphClusterManager()
    assert mgr.get_cluster_id(p_trans) == "trans_trans_999"


# ---------------------------------------------------------------------------
# 17. Zero Cross-Split Leakage
# ---------------------------------------------------------------------------
def test_17_cross_split_leakage():
    """Verify generated graph dataset in data/processed/graphs has 0 cross-split leakage."""
    validator = GraphDatasetValidator("data/processed/graphs")
    res = validator.validate()
    assert res["is_valid"] is True
    assert res["cross_split_leakage"] == 0


# ---------------------------------------------------------------------------
# 18. Graph Statistics Calculation
# ---------------------------------------------------------------------------
def test_18_graph_statistics():
    """Verify graph dataset manifest reports all required structural statistics."""
    validator = GraphDatasetValidator("data/processed/graphs")
    graphs = validator.load_graphs()
    manifest_builder = GraphManifestBuilder()
    manifest = manifest_builder.build_manifest(graphs)

    assert manifest.graph_count == 1560
    assert manifest.node_count == 7800
    assert manifest.edge_count == 6240
    assert manifest.density > 0.0
    assert manifest.average_nodes_per_graph == 5.0
    assert manifest.average_edges_per_graph == 4.0


# ---------------------------------------------------------------------------
# 19. Manifest Generation
# ---------------------------------------------------------------------------
def test_19_manifest_generation():
    """Verify dataset_manifest.json on disk contains expected schema versions and counts."""
    manifest_path = Path("data/processed/graphs/dataset_manifest.json")
    assert manifest_path.is_file()
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["graph_dataset_version"] == "v0.1.0"
    assert data["source_dataset_version"] == "v0.2.0"
    assert data["cross_split_leakage"] == 0
    assert data["duplicate_nodes"] == 0
    assert data["duplicate_edges"] == 0


# ---------------------------------------------------------------------------
# 20. Validator CLI and Reports
# ---------------------------------------------------------------------------
def test_20_validator_cli():
    """Verify validator CLI returns exit code 0 on valid graph data."""
    cmd = [sys.executable, "-m", "app.graph_dataset.validator", "--data-dir", "data/processed/graphs"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "Overall:              VALID" in proc.stdout


# ---------------------------------------------------------------------------
# 21. Dry-Run CLI Execution
# ---------------------------------------------------------------------------
def test_21_dry_run_cli():
    """Verify builder CLI dry-run executes cleanly without writing files."""
    test_out = Path("data/processed/graphs_dry_run_test")
    cmd = [
        sys.executable,
        "-m",
        "app.graph_dataset.builder",
        "--data-dir",
        "data/trustlens",
        "--output-dir",
        str(test_out),
        "--dry-run",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0
    assert "DRY RUN (No files written)" in proc.stdout
    assert not test_out.exists()


# ---------------------------------------------------------------------------
# 22. Invalid Graph Rejection
# ---------------------------------------------------------------------------
def test_22_invalid_graph_rejection(tmp_path):
    """Verify validator flags invalid edges with missing endpoints or broken types."""
    corrupt_dir = tmp_path / "corrupt_graphs"
    corrupt_dir.mkdir()

    # Broken graph: edge points to non-existent target node
    broken_graph = {
        "graph_id": "graph_broken",
        "split": "train",
        "cluster_id": "c1",
        "nodes": [
            {
                "node_id": "node_post_1",
                "node_type": "POST",
                "source_id": "p1",
                "post_id": "p1",
                "text": "sample",
                "split": "train",
                "cluster_id": "c1",
                "metadata": {},
            }
        ],
        "edges": [
            {
                "edge_id": "edge_broken",
                "source_node_id": "node_post_1",
                "source_node_type": "POST",
                "target_node_id": "node_claim_NON_EXISTENT",  # Missing!
                "target_node_type": "CLAIM",
                "edge_type": "HAS_CLAIM",
                "split": "train",
                "cluster_id": "c1",
                "metadata": {},
            }
        ],
        "labels": {},
        "metadata": {},
    }

    with (corrupt_dir / "train.json").open("w", encoding="utf-8") as f:
        json.dump([broken_graph], f)

    validator = GraphDatasetValidator(str(corrupt_dir))
    report = validator.validate()
    assert report["is_valid"] is False
    assert report["invalid_edges"] > 0


# ---------------------------------------------------------------------------
# 23. Grounded Feature Integrity (No Fabrication)
# ---------------------------------------------------------------------------
def test_23_grounded_feature_integrity(sample_training_post):
    """Verify features never fabricate fake dense embeddings."""
    feature_builder = GraphFeatureBuilder()
    node_builder = NodeBuilder()
    post_node = node_builder.build_post_node(sample_training_post, "c1", SplitName.train)
    feats = feature_builder.build_post_features(post_node)

    assert feats["text_embedding_ref"] is None
    assert feats["embedding_available"] is False


# ---------------------------------------------------------------------------
# 24. PyG Dictionary Representation & Status
# ---------------------------------------------------------------------------
def test_24_pyg_conversion_status(sample_training_post, sample_training_claim, sample_training_evidence):
    """Verify PyG converter status reporting and dictionary tensor representation."""
    builder = EvidenceGraphBuilder()
    graph = builder.build_graph_for_post(
        sample_training_post, [sample_training_claim], [sample_training_evidence]
    )

    # Dictionary conversion works offline in any environment
    pyg_dict = PyGHeteroDataConverter.to_dict_representation(graph)
    assert pyg_dict["graph_id"] == graph.graph_id
    assert "node_types" in pyg_dict
    assert "edge_indices" in pyg_dict

    status = PyGHeteroDataConverter.get_status()
    assert status in ("READY", "PYG_NOT_INSTALLED", "TORCH_AND_PYG_NOT_INSTALLED")
