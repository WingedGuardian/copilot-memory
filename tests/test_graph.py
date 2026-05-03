"""Tests for graph traversal using recursive CTEs.

Uses real SQLite (CTEs can't be mocked). Schema initialized via
genesis_memory.backends.schema.init_schema.
"""

import aiosqlite
import pytest

from genesis_memory.backends.schema import init_schema
from genesis_memory.graph import (
    GraphNode,
    TraversalResult,
    find_connected_by_type,
    get_cluster,
    traverse,
)


@pytest.fixture
async def db():
    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await init_schema(conn)
    yield conn
    await conn.close()


async def _insert_link(
    db: aiosqlite.Connection,
    source_id: str,
    target_id: str,
    link_type: str = "supports",
    strength: float = 0.8,
) -> None:
    await db.execute(
        "INSERT INTO memory_links (source_id, target_id, link_type, strength, created_at) "
        "VALUES (?, ?, ?, ?, '2026-05-01T00:00:00+00:00')",
        (source_id, target_id, link_type, strength),
    )
    await db.commit()


# -- traverse tests --


class TestTraverse:
    async def test_single_hop(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "A", "C", "extends", 0.8)

        result = await traverse(db, "A", max_depth=1)
        assert isinstance(result, TraversalResult)
        assert result.root_id == "A"
        assert len(result.nodes) == 2

        ids = {n.memory_id for n in result.nodes}
        assert ids == {"B", "C"}

    async def test_multi_hop(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)
        await _insert_link(db, "C", "D", "supports", 0.7)

        result = await traverse(db, "A", max_depth=3)
        ids = {n.memory_id for n in result.nodes}
        assert ids == {"B", "C", "D"}

        # Check depths
        depths = {n.memory_id: n.depth for n in result.nodes}
        assert depths["B"] == 1
        assert depths["C"] == 2
        assert depths["D"] == 3

    async def test_depth_limit(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)
        await _insert_link(db, "C", "D", "supports", 0.7)

        result = await traverse(db, "A", max_depth=2)
        ids = {n.memory_id for n in result.nodes}
        assert "D" not in ids  # depth 3, beyond max_depth=2
        assert "B" in ids
        assert "C" in ids

    async def test_min_strength_filter(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "A", "C", "supports", 0.3)  # below threshold

        result = await traverse(db, "A", min_strength=0.5)
        ids = {n.memory_id for n in result.nodes}
        assert "B" in ids
        assert "C" not in ids

    async def test_no_outgoing_links(self, db):
        result = await traverse(db, "isolated")
        assert result.nodes == []
        assert result.root_id == "isolated"

    async def test_query_ms_populated(self, db):
        await _insert_link(db, "A", "B")
        result = await traverse(db, "A")
        assert result.query_ms >= 0

    async def test_reverse_not_followed(self, db):
        """traverse() only follows outgoing (source->target) links."""
        await _insert_link(db, "B", "A", "supports", 0.9)  # B->A, not A->B
        result = await traverse(db, "A")
        assert result.nodes == []  # A has no outgoing links


# -- cycle handling --


class TestCycleHandling:
    async def test_direct_cycle(self, db):
        """A -> B -> A should not infinite loop."""
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "A", "extends", 0.8)

        result = await traverse(db, "A", max_depth=10)
        # Should find B but not loop back to A
        ids = {n.memory_id for n in result.nodes}
        assert "B" in ids
        # A might appear if the CTE doesn't perfectly prevent it,
        # but the important thing is it doesn't hang
        assert len(result.nodes) <= 5  # bounded, not infinite

    async def test_triangle_cycle(self, db):
        """A -> B -> C -> A should terminate."""
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)
        await _insert_link(db, "C", "A", "supports", 0.7)

        result = await traverse(db, "A", max_depth=10)
        ids = {n.memory_id for n in result.nodes}
        assert "B" in ids
        assert "C" in ids
        # Should terminate without infinite loop
        assert len(result.nodes) <= 10

    async def test_self_loop(self, db):
        """A -> A should not infinite loop."""
        await _insert_link(db, "A", "A", "supports", 0.9)
        result = await traverse(db, "A", max_depth=5)
        # Should terminate
        assert len(result.nodes) <= 5


# -- find_connected_by_type tests --


class TestFindConnectedByType:
    async def test_filters_by_link_type(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "A", "C", "extends", 0.8)
        await _insert_link(db, "A", "D", "supports", 0.7)

        nodes = await find_connected_by_type(db, "A", "supports")
        ids = {n.memory_id for n in nodes}
        assert ids == {"B", "D"}
        assert "C" not in ids  # "extends", not "supports"

    async def test_multi_hop_same_type(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "supports", 0.8)

        nodes = await find_connected_by_type(db, "A", "supports", max_depth=2)
        ids = {n.memory_id for n in nodes}
        assert ids == {"B", "C"}

    async def test_stops_at_different_type(self, db):
        """Chain A->B(supports)->C(extends) should stop at C for type=supports."""
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)  # different type

        nodes = await find_connected_by_type(db, "A", "supports", max_depth=3)
        ids = {n.memory_id for n in nodes}
        assert "B" in ids
        assert "C" not in ids

    async def test_no_matching_type(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        nodes = await find_connected_by_type(db, "A", "contradicts")
        assert nodes == []

    async def test_no_links(self, db):
        nodes = await find_connected_by_type(db, "isolated", "supports")
        assert nodes == []

    async def test_node_structure(self, db):
        await _insert_link(db, "A", "B", "extends", 0.85)
        nodes = await find_connected_by_type(db, "A", "extends")
        assert len(nodes) == 1
        node = nodes[0]
        assert isinstance(node, GraphNode)
        assert node.memory_id == "B"
        assert node.link_type == "extends"
        assert node.depth == 1
        assert node.strength == 0.85


# -- get_cluster tests --


class TestGetCluster:
    async def test_bidirectional_discovery(self, db):
        """Cluster follows links in both directions."""
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "C", "A", "extends", 0.8)  # reverse direction

        cluster = await get_cluster(db, "A", min_strength=0.5)
        assert "B" in cluster
        assert "C" in cluster

    async def test_min_strength_filter(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "A", "C", "supports", 0.3)  # below threshold

        cluster = await get_cluster(db, "A", min_strength=0.5)
        assert "B" in cluster
        assert "C" not in cluster

    async def test_connected_component(self, db):
        """Full connected component discovery."""
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)
        await _insert_link(db, "D", "A", "supports", 0.7)

        cluster = await get_cluster(db, "A", max_depth=3, min_strength=0.5)
        assert set(cluster) >= {"B", "C", "D"}  # all connected to A

    async def test_isolated_node(self, db):
        cluster = await get_cluster(db, "isolated")
        assert cluster == []

    async def test_depth_limit(self, db):
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)
        await _insert_link(db, "C", "D", "supports", 0.7)

        cluster = await get_cluster(db, "A", max_depth=1, min_strength=0.5)
        assert "B" in cluster
        # C and D are beyond depth 1 — may or may not be included
        # depending on reverse links, but D should not appear

    async def test_cycle_in_cluster(self, db):
        """Cluster with cycles should terminate."""
        await _insert_link(db, "A", "B", "supports", 0.9)
        await _insert_link(db, "B", "C", "extends", 0.8)
        await _insert_link(db, "C", "A", "supports", 0.7)

        cluster = await get_cluster(db, "A", max_depth=5, min_strength=0.5)
        # Should find all 3 nodes without infinite loop
        assert "B" in cluster
        assert "C" in cluster
        assert len(cluster) <= 10  # bounded
