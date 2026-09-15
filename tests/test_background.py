"""Offline unit tests for the GQR-Bench v2 background builder.

Run: uv run --with pytest pytest tests
"""

import numpy as np
import pytest

from gqr.core.background import (
    OverlapIndex,
    fingerprint,
    id_topic_hit,
    select_background,
)
from gqr.core.dataloader import load_train_dataset


def never_in_domain(texts: list[str]) -> np.ndarray:
    return np.zeros(len(texts))


def pools(n: int) -> dict[str, list[str]]:
    return {
        name: [f"{name} passage number {i} about gardens rivers and old maps" for i in range(n)]
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


def test_select_background_is_balanced_deterministic_and_sized() -> None:
    args = (pools(30), never_in_domain, OverlapIndex(), 30)
    texts, sources, stats = select_background(*args, seed=7)
    again, _, _ = select_background(*args, seed=7)
    assert len(texts) == 30
    assert texts == again
    assert {s: sources.count(s) for s in set(sources)} == {"wikitext": 10, "dolly": 10, "yahoo": 10}
    assert stats["dolly_used"] == 10


def test_select_background_drops_keywords_classifier_hits_and_test_overlap() -> None:
    candidates = pools(12)
    candidates["dolly"].append("How do I sue my employer for unpaid overtime?")
    leaked = candidates["yahoo"][0]

    def flags_rivers(texts: list[str]) -> np.ndarray:
        return np.array([1.0 if "number 1 " in t else 0.0 for t in texts])

    texts, _, stats = select_background(
        candidates, flags_rivers, OverlapIndex([leaked]), 27, seed=0
    )
    assert "How do I sue my employer for unpaid overtime?" not in texts
    assert leaked not in texts
    assert stats["yahoo_test_overlap_dropped"] == 1
    assert stats["wikitext_classifier_dropped"] == 1


def test_select_background_raises_when_a_source_runs_out() -> None:
    with pytest.raises(ValueError, match="clean candidates"):
        select_background(pools(3), never_in_domain, OverlapIndex(), 30)


def test_fingerprint_is_order_sensitive() -> None:
    assert fingerprint(["a", "b"]) != fingerprint(["b", "a"])


def test_unknown_version_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown GQR-Bench version"):
        load_train_dataset("v3")
