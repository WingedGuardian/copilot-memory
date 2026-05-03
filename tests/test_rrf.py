"""Tests for RRF (Reciprocal Rank Fusion) — isolated unit tests."""

from genesis_memory.retrieval import _rrf_fuse


class TestRRFFuse:
    def test_single_list(self):
        fused = _rrf_fuse([["a", "b", "c"]])
        assert fused["a"] > fused["b"] > fused["c"]

    def test_two_lists_same_order(self):
        fused = _rrf_fuse([["a", "b", "c"], ["a", "b", "c"]])
        # Same order, so "a" gets highest score (rank 1 in both)
        assert fused["a"] > fused["b"] > fused["c"]

    def test_two_lists_reversed(self):
        fused = _rrf_fuse([["a", "b", "c"], ["c", "b", "a"]])
        # "a" gets rank 1+3 = 1/61 + 1/63
        # "b" gets rank 2+2 = 2/62
        # "c" gets rank 3+1 = 1/63 + 1/61
        # a == c (symmetric), both slightly above b (rank 1 is disproportionately valuable)
        assert abs(fused["a"] - fused["c"]) < 1e-10
        assert fused["a"] >= fused["b"]

    def test_disjoint_lists(self):
        fused = _rrf_fuse([["a", "b"], ["c", "d"]])
        assert len(fused) == 4
        # Rank 1 items from each list have equal scores
        assert fused["a"] == fused["c"]
        assert fused["b"] == fused["d"]

    def test_partial_overlap(self):
        fused = _rrf_fuse([["a", "b", "c"], ["b", "d"]])
        # "b" appears in both lists → higher than "a" or "d"?
        # b: 1/(60+2) + 1/(60+1) = 1/62 + 1/61
        # a: 1/(60+1) = 1/61
        # d: 1/(60+2) = 1/62
        assert fused["b"] > fused["a"]
        assert fused["b"] > fused["d"]

    def test_empty_list_ignored(self):
        fused = _rrf_fuse([["a", "b"], []])
        assert len(fused) == 2

    def test_k_parameter(self):
        fused_default = _rrf_fuse([["a", "b"]], k=60)
        fused_small = _rrf_fuse([["a", "b"]], k=1)
        # Smaller k amplifies rank differences
        gap_default = fused_default["a"] - fused_default["b"]
        gap_small = fused_small["a"] - fused_small["b"]
        assert gap_small > gap_default

    def test_three_lists(self):
        fused = _rrf_fuse([
            ["a", "b", "c"],
            ["c", "a", "b"],
            ["b", "c", "a"],
        ])
        # Each item appears at rank 1, 2, 3 across lists → all equal
        assert abs(fused["a"] - fused["b"]) < 1e-10
        assert abs(fused["b"] - fused["c"]) < 1e-10

    def test_ordinal_only(self):
        """RRF must be ordinal-only — absolute scores should not matter."""
        # Same ranked lists, same results regardless of "source scores"
        list1 = ["x", "y", "z"]
        list2 = ["y", "z", "x"]
        fused = _rrf_fuse([list1, list2])
        # "y" is rank 1+2, "x" is rank 1+3, "z" is rank 2+3
        assert fused["y"] > fused["x"]
        assert fused["x"] > fused["z"]
