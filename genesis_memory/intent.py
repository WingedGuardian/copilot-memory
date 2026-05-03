"""Query intent classification and expansion for memory retrieval.

Classifies queries into intent categories (WHAT/WHY/HOW/WHEN/WHERE/STATUS)
and biases retrieval toward intent-appropriate memories. Also performs
query expansion via tag co-occurrence for improved FTS5 recall.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from genesis_memory.protocols import VectorBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryIntent:
    """Classified query intent."""

    category: str  # WHAT, WHY, HOW, WHEN, WHERE, STATUS, GENERAL
    confidence: float  # 0.0-1.0
    matched_pattern: str  # pattern that triggered (debug/logging)


@dataclass(frozen=True)
class IntentProfile:
    """Scoring profile for an intent category."""

    boosted_sources: frozenset[str]
    boosted_tags: frozenset[str]
    content_signals: tuple[str, ...]


# Priority order matters: more specific intents first.
_INTENT_PATTERNS: list[tuple[str, re.Pattern[str], float]] = [
    ("WHY", re.compile(
        r"^(why\b|what.{0,20}reason|rationale\b|motivation\b|justification\b)",
        re.IGNORECASE,
    ), 0.85),
    ("HOW", re.compile(
        r"^(how\b|steps?\s+to\b|procedure\b|process\s+for\b|instructions?\b)",
        re.IGNORECASE,
    ), 0.85),
    ("WHEN", re.compile(
        r"^(when\b|timeline\b|history\s+of\b|last\s+time\b|first\s+time\b)",
        re.IGNORECASE,
    ), 0.85),
    ("WHERE", re.compile(
        r"^(where\b|location\s+of\b|which\s+file\b|find\s+the\b)",
        re.IGNORECASE,
    ), 0.85),
    ("STATUS", re.compile(
        r"(status\s+of\b|progress\s+on\b|current\s+state\b|update\s+on\b|is\s+it\s+done\b)",
        re.IGNORECASE,
    ), 0.80),
    ("WHAT", re.compile(
        r"^(what\b|define\b|describe\b|explain\s+what\b|tell\s+me\s+about\b)",
        re.IGNORECASE,
    ), 0.80),
]

INTENT_PROFILES: dict[str, IntentProfile] = {
    "WHAT": IntentProfile(
        boosted_sources=frozenset({"session_extraction"}),
        boosted_tags=frozenset({"entity", "concept"}),
        content_signals=(),
    ),
    "WHY": IntentProfile(
        boosted_sources=frozenset({"deep_reflection", "retrospective"}),
        boosted_tags=frozenset({"decision", "evaluation"}),
        content_signals=("because", "decided", "rationale", "reason", "chose"),
    ),
    "HOW": IntentProfile(
        boosted_sources=frozenset({"auto_memory_harvest", "session_extraction"}),
        boosted_tags=frozenset({"action_item", "concept"}),
        content_signals=("step", "run", "execute", "command", "install", "configure"),
    ),
    "WHEN": IntentProfile(
        boosted_sources=frozenset({"session_extraction", "retrospective"}),
        boosted_tags=frozenset(),
        content_signals=(),
    ),
    "WHERE": IntentProfile(
        boosted_sources=frozenset({"session_extraction"}),
        boosted_tags=frozenset({"entity"}),
        content_signals=(),
    ),
    "STATUS": IntentProfile(
        boosted_sources=frozenset({"retrospective", "reflection"}),
        boosted_tags=frozenset({"action_item"}),
        content_signals=("status", "progress", "done", "pending", "complete"),
    ),
}


def classify_intent(query: str) -> QueryIntent:
    """Classify query intent using compiled regex patterns.

    Returns GENERAL with confidence 0.0 if no pattern matches.
    """
    cleaned = query.strip()
    if not cleaned:
        return QueryIntent(category="GENERAL", confidence=0.0, matched_pattern="")

    for category, pattern, confidence in _INTENT_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return QueryIntent(
                category=category,
                confidence=confidence,
                matched_pattern=match.group(0),
            )

    return QueryIntent(category="GENERAL", confidence=0.0, matched_pattern="")


def compute_intent_affinity(
    intent: QueryIntent,
    source: str,
    tags: list[str],
    content: str,
) -> float:
    """Compute intent affinity score for a single memory."""
    if intent.category == "GENERAL":
        return 0.0

    profile = INTENT_PROFILES.get(intent.category)
    if profile is None:
        return 0.0

    score = 0.0
    if source in profile.boosted_sources:
        score += 2.0
    if tags and profile.boosted_tags and (profile.boosted_tags & set(tags)):
        score += 1.5
    if profile.content_signals and content:
        content_lower = content.lower()
        if any(sig in content_lower for sig in profile.content_signals):
            score += 1.0

    return score


def rank_by_intent(
    intent: QueryIntent,
    candidates: dict[str, dict],
) -> list[str]:
    """Rank candidate memory IDs by intent affinity.

    Returns empty list for GENERAL intent (no bias applied).
    """
    if intent.category == "GENERAL":
        return []

    scored: list[tuple[str, float]] = []
    for mid, meta in candidates.items():
        affinity = compute_intent_affinity(
            intent,
            source=meta.get("source", ""),
            tags=meta.get("tags") or [],
            content=meta.get("content", ""),
        )
        scored.append((mid, affinity))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [mid for mid, _ in scored]


# ---------------------------------------------------------------------------
# Query expansion via tag co-occurrence
# ---------------------------------------------------------------------------

@dataclass
class TagCooccurrenceIndex:
    """Lazily-built, cached index of tag co-occurrence from vector payloads."""

    _cooccurrence: dict[str, dict[str, int]] = field(default_factory=dict)
    _memory_count: int = 0
    _built_at: float = 0.0
    _stale_threshold: float = 0.10

    def is_stale(self, current_count: int) -> bool:
        if self._memory_count == 0:
            return True
        delta = abs(current_count - self._memory_count) / max(self._memory_count, 1)
        return delta > self._stale_threshold

    def build(self, tag_lists: list[list[str]], memory_count: int) -> None:
        """Build co-occurrence index from tag lists across all memories."""
        cooc: dict[str, dict[str, int]] = {}
        for tags in tag_lists:
            if len(tags) < 2:
                continue
            normalized = list({t.lower() for t in tags if t and not t.startswith("obs:")})
            for i, tag_a in enumerate(normalized):
                if tag_a not in cooc:
                    cooc[tag_a] = {}
                for tag_b in normalized[i + 1:]:
                    cooc[tag_a][tag_b] = cooc[tag_a].get(tag_b, 0) + 1
                    if tag_b not in cooc:
                        cooc[tag_b] = {}
                    cooc[tag_b][tag_a] = cooc[tag_b].get(tag_a, 0) + 1

        self._cooccurrence = cooc
        self._memory_count = memory_count
        self._built_at = time.monotonic()
        logger.info(
            "Tag co-occurrence index built: %d unique tags, %d memories",
            len(cooc), memory_count,
        )

    def expand(self, keywords: list[str], max_expansions: int = 5) -> list[str]:
        """Find tags that co-occur with the given keywords."""
        if not self._cooccurrence or not keywords:
            return []

        keyword_set = {k.lower() for k in keywords}
        expansion_scores: dict[str, int] = {}

        for kw in keyword_set:
            neighbors = self._cooccurrence.get(kw, {})
            for tag, count in neighbors.items():
                if tag not in keyword_set:
                    expansion_scores[tag] = expansion_scores.get(tag, 0) + count

        if not expansion_scores:
            return []

        ranked = sorted(expansion_scores.items(), key=lambda x: -x[1])
        return [tag for tag, _ in ranked[:max_expansions]]


# Module-level singleton
_tag_index = TagCooccurrenceIndex()


def _tokenize_query(query: str) -> list[str]:
    """Extract meaningful keywords from a query string."""
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "do", "does", "did", "have", "has", "had", "will", "would",
        "can", "could", "should", "may", "might", "shall",
        "in", "on", "at", "to", "for", "of", "with", "by", "from",
        "it", "its", "this", "that", "these", "those",
        "i", "we", "you", "he", "she", "they", "me", "us", "my",
        "what", "why", "how", "when", "where", "which", "who",
        "about", "and", "or", "not", "but", "so", "if", "then",
    }
    tokens = re.findall(r"\w+", query.lower())
    return [t for t in tokens if t not in stop_words and len(t) > 1]


async def expand_query(
    query: str,
    vector_backend: VectorBackend,
    collections: list[str],
    *,
    max_expansions: int = 5,
) -> str:
    """Expand a query with co-occurring tags for improved FTS5 recall.

    Uses VectorBackend.scroll_tags() to build/refresh the tag co-occurrence
    index, then appends related terms to the query string.
    """
    global _tag_index  # noqa: PLW0603

    try:
        # Build/refresh index via VectorBackend protocol
        tag_lists, total_count = await vector_backend.scroll_tags(collections)

        if _tag_index.is_stale(total_count) and total_count > 0:
            _tag_index.build(tag_lists, total_count)

        # Expand query
        keywords = _tokenize_query(query)
        if not keywords:
            return query

        expansions = _tag_index.expand(keywords, max_expansions=max_expansions)
        if not expansions:
            return query

        original_and = " AND ".join(keywords)
        parts = [f"({original_and})"] + expansions
        expanded = " OR ".join(parts)
        logger.debug("Query expanded: %r → %r", query, expanded)
        return expanded

    except Exception:
        logger.error("Query expansion failed, using original query", exc_info=True)
        return query
