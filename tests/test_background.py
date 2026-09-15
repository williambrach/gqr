"""Offline unit tests for the GQR-Bench v2 background builder.

Run: uv run --with pytest pytest tests
"""

import random

import pytest

from gqr.core.background import (
    OverlapIndex,
    fingerprint,
    id_topic_hit,
    select_background,
    smallest_eligible,
)
from gqr.core.dataloader import load_train_dataset


def pools(n: int) -> dict[str, list[str]]:
    # fewer than 8 words each, so overlap means an exact match, never a shared shingle
    return {
        name: [f"{name} item {i} about gardens" for i in range(n)]
        for name in ("wikitext", "dolly", "yahoo")
    }


def test_id_topic_hit_flags_domain_keywords() -> None:
    assert id_topic_hit("My landlord kept the deposit") == "law"
    assert id_topic_hit("Should I refinance my mortgage?") == "finance"
    assert id_topic_hit("A persistent cough after the flu") == "healthcare"
    assert id_topic_hit("The cathedral was completed in 1248") is None


def test_overlap_index_catches_exact_and_shingle_overlap() -> None:
    index = OverlapIndex(["The quick brown fox jumps over the lazy dog near the river bank"])
    assert index.hit("the QUICK brown fox jumps over the lazy dog near the river bank!")
    assert index.hit("Yesterday the quick brown fox jumps over the lazy dog near a barn")
    assert not index.hit("A completely unrelated sentence about sailing boats")


def test_smallest_eligible_ignores_input_order_and_filters_keywords() -> None:
    texts = [f"passage {i} about sailing" for i in range(200)] + ["Can I sue my landlord?"] * 3
    shuffled = texts[:]
    random.Random(1).shuffle(shuffled)
    first = smallest_eligible(texts, 20, seed=42, salt="dolly")
    assert first == smallest_eligible(shuffled, 20, seed=42, salt="dolly")
    assert len(first) == len(set(first)) == 20
    assert all(id_topic_hit(t) is None for t in first)
    assert first != smallest_eligible(texts, 20, seed=43, salt="dolly")


def test_select_background_is_balanced_sized_and_order_invariant() -> None:
    a = pools(30)
    b = {name: list(reversed(pool)) for name, pool in a.items()}
    texts, sources, stats = select_background(a, OverlapIndex(), 30, seed=7)
    assert (texts, sources) == select_background(b, OverlapIndex(), 30, seed=7)[:2]
    assert len(texts) == 30
    assert {s: sources.count(s) for s in set(sources)} == {"wikitext": 10, "dolly": 10, "yahoo": 10}
    assert stats["dolly_used"] == 10


def test_select_background_drops_keywords_and_test_overlap_locally() -> None:
    candidates = pools(12)
    candidates["dolly"].append("How do I sue my employer for unpaid overtime?")
    leaked = candidates["yahoo"][0]
    texts, _, stats = select_background(candidates, OverlapIndex([leaked]), 27, seed=0)
    assert "How do I sue my employer for unpaid overtime?" not in texts
    assert leaked not in texts
    assert stats["yahoo_test_overlap_dropped"] == 1
    assert stats["dolly_test_overlap_dropped"] == stats["wikitext_test_overlap_dropped"] == 0
    clean, _, _ = select_background(pools(12), OverlapIndex(), 27, seed=0)
    assert len(set(clean) - set(texts)) <= 1  # one dropped candidate changes at most one row


def test_select_background_raises_when_a_source_runs_out() -> None:
    with pytest.raises(ValueError, match="clean candidates"):
        select_background(pools(3), OverlapIndex(), 30)


def test_fingerprint_is_order_sensitive() -> None:
    assert fingerprint(["a", "b"]) != fingerprint(["b", "a"])


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown GQR-Bench version"):
        load_train_dataset("v3")
