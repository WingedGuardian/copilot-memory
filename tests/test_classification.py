"""Tests for memory classification heuristics."""

from genesis_memory.classification import CLASS_WEIGHTS, classify_memory


class TestClassifyMemory:
    def test_rule_always(self):
        assert classify_memory("You ALWAYS need to check permissions") == "rule"

    def test_rule_never(self):
        assert classify_memory("NEVER push to main without a PR") == "rule"

    def test_rule_must_not(self):
        assert classify_memory("You MUST NOT share API keys") == "rule"

    def test_rule_must(self):
        assert classify_memory("MUST run tests before committing") == "rule"

    def test_rule_do_not(self):
        assert classify_memory("DO NOT delete the database") == "rule"

    def test_rule_critical_colon(self):
        assert classify_memory("CRITICAL: the server is overloaded") == "rule"

    def test_reference_url(self):
        assert classify_memory("See https://example.com/docs/very-long-path") == "reference"

    def test_reference_tracked_in(self):
        assert classify_memory("Pipeline bugs are tracked in Linear project INGEST") == "reference"

    def test_reference_see_also(self):
        assert classify_memory("See also the deployment runbook") == "reference"

    def test_reference_grafana(self):
        assert classify_memory("Check the Grafana dashboard for latency") == "reference"

    def test_fact_default(self):
        assert classify_memory("The routing system uses circuit breakers") == "fact"

    def test_fact_neutral(self):
        assert classify_memory("Session discussed memory retrieval improvements") == "fact"

    def test_cc_memory_type_feedback(self):
        assert classify_memory("some content", cc_memory_type="feedback") == "rule"

    def test_cc_memory_type_user(self):
        assert classify_memory("some content", cc_memory_type="user") == "rule"

    def test_cc_memory_type_project(self):
        assert classify_memory("some content", cc_memory_type="project") == "fact"

    def test_cc_memory_type_reference(self):
        assert classify_memory("some content", cc_memory_type="reference") == "reference"

    def test_cc_memory_type_overrides_heuristic(self):
        # Even though content has NEVER, cc_memory_type="project" wins
        assert classify_memory("NEVER do this", cc_memory_type="project") == "fact"

    def test_cc_memory_type_unknown_falls_through(self):
        # Unknown cc_memory_type falls through to heuristics
        assert classify_memory("ALWAYS check", cc_memory_type="unknown") == "rule"

    def test_lowercase_always_is_fact(self):
        # Only ALL-CAPS triggers rule classification
        assert classify_memory("You should always be careful") == "fact"


class TestClassWeights:
    def test_rule_weight(self):
        assert CLASS_WEIGHTS["rule"] == 1.3

    def test_fact_weight(self):
        assert CLASS_WEIGHTS["fact"] == 1.0

    def test_reference_weight(self):
        assert CLASS_WEIGHTS["reference"] == 0.7
