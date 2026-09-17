"""Offline unit tests for the GQR-Bench v2 background builder.

Run: uv run --with pytest pytest tests
"""

import os
import random
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pandas as pd
import pytest

from gqr.core import background
from gqr.core.background import (
    OverlapIndex,
    detokenize_wikitext,
    fingerprint,
    id_topic_hit,
    normalize,
    select_background,
    smallest_eligible,
)
from gqr.core.dataloader import DataLoader, load_train_dataset


def pools(n: int, **sizes: int) -> dict[str, list[str]]:
    # fewer than 8 words each, so overlap means an exact match, never a shared shingle
    return {
        name: [f"{name} item {i} about gardens" for i in range(sizes.get(name, n))]
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


def test_smallest_eligible_collapses_normalized_duplicates_order_invariantly() -> None:
    variants = ["What is photosynthesis?", "What is Photosynthesis", "what is photosynthesis!"]
    texts = variants + [f"passage {i} about sailing" for i in range(50)]
    first = smallest_eligible(texts, 51, seed=42, salt="dolly")
    assert first == smallest_eligible(list(reversed(texts)), 51, seed=42, salt="dolly")
    assert len({normalize(t) for t in first}) == len(first) == 51
    assert "What is Photosynthesis" in first  # smallest variant represents the group


def test_select_background_deduplicates_normalized_text_within_and_across_sources() -> None:
    # every source has exactly 12 distinct clean texts, so all of them are selected
    candidates = pools(12, dolly=10)
    candidates["dolly"] += ["What is photosynthesis?", "What is Photosynthesis"]
    candidates["dolly"] += ["What is the meaning of life?", "what is the meaning of life?"]
    candidates["yahoo"] += ["WHAT IS THE MEANING OF LIFE", "yahoo item 3 about gardens!"]
    texts, sources, stats = select_background(candidates, OverlapIndex(), 36, seed=0)
    norms = [normalize(t) for t in texts]
    assert len(set(norms)) == len(norms) == 36
    assert "what is photosynthesis" in norms and "what is the meaning of life" in norms
    assert sources[norms.index("what is the meaning of life")] == "dolly"
    assert stats["yahoo_cross_source_duplicates_dropped"] == 1
    assert stats["yahoo_candidates"] == 13  # "yahoo item 3" variants collapse to one


def test_build_puts_no_normalized_duplicate_in_both_splits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # 10 distinct clean texts per source for 30 rows, so every duplicate group is used
    base = pools(10, dolly=8)
    extra = {
        "dolly": ["What is photosynthesis?", "What is Photosynthesis",
                  "What is the meaning of life?", "what is the meaning of life?"],
        "yahoo": ["What is the Meaning of Life", "what is photosynthesis ?"],
    }

    def fake_candidates(source: str, limit: int, seed: int) -> list[str]:
        texts = base[source] + extra.get(source, [])
        return smallest_eligible(texts, limit, seed, source)

    monkeypatch.setenv("GQR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(background, "_candidates", fake_candidates)
    monkeypatch.setattr(DataLoader, "load_ood_test_dataset", staticmethod(dict))
    id_test = pd.DataFrame({"text": ["an unrelated test question"]})
    train_bg, eval_bg = background._build_or_load(id_test, 24, 6, check=False)
    train_norms = set(train_bg["text"].map(normalize))
    eval_norms = set(eval_bg["text"].map(normalize))
    assert len(train_norms) == len(train_bg) == 24 and len(eval_norms) == len(eval_bg) == 6
    assert not train_norms & eval_norms
    assert {"what is photosynthesis", "what is the meaning of life"} <= train_norms | eval_norms


def test_build_warns_and_does_not_cache_when_an_ood_test_set_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = pools(10)
    empty = pd.DataFrame(columns=["text", "label", "domain"])
    ood = {"dkhate": empty, "olid": pd.DataFrame({"text": ["an unrelated tweet"]})}
    monkeypatch.setenv("GQR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(background, "_candidates", lambda source, limit, seed: base[source])
    monkeypatch.setattr(DataLoader, "load_ood_test_dataset", staticmethod(lambda: ood))
    id_test = pd.DataFrame({"text": ["an unrelated test question"]})
    with pytest.warns(UserWarning, match=r"\['dkhate'\].*not cached"):
        train_bg, eval_bg = background._build_or_load(id_test, 24, 6, check=True)
    assert len(train_bg) == 24 and len(eval_bg) == 6
    assert not list(tmp_path.iterdir())

    ood["dkhate"] = pd.DataFrame({"text": ["en dansk tekst"]})
    background._build_or_load(id_test, 24, 6, check=False)
    assert len(list(tmp_path.glob("*.parquet"))) == 1


def test_build_cache_survives_interrupted_writes_and_corrupt_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    base = pools(10)
    monkeypatch.setenv("GQR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(background, "_candidates", lambda source, limit, seed: base[source])
    monkeypatch.setattr(DataLoader, "load_ood_test_dataset", staticmethod(dict))
    id_test = pd.DataFrame({"text": ["an unrelated test question"]})
    to_parquet = pd.DataFrame.to_parquet

    def interrupted(self: pd.DataFrame, path: Path, **kwargs: object) -> None:
        Path(path).write_bytes(b"PAR1 partial")
        raise KeyboardInterrupt

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted)
    with pytest.raises(KeyboardInterrupt):
        background._build_or_load(id_test, 24, 6, check=False)
    assert not list(tmp_path.glob("*.parquet")) and not list(tmp_path.glob(".*.tmp"))

    monkeypatch.setattr(pd.DataFrame, "to_parquet", to_parquet)
    expected = background._build_or_load(id_test, 24, 6, check=False)
    (cache,) = tmp_path.glob("*.parquet")
    cache.write_bytes(b"PAR1 truncated")  # e.g. a file left by an older version
    with pytest.warns(UserWarning, match="unreadable background cache"):
        rebuilt = background._build_or_load(id_test, 24, 6, check=False)
    assert all(a.equals(b) for a, b in zip(rebuilt, expected, strict=True))
    assert background._build_or_load(id_test, 24, 6, check=False)[0].equals(expected[0])


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_build_cache_files_get_default_permissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mask: int
) -> None:
    base = pools(10)
    monkeypatch.setenv("GQR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(background, "_candidates", lambda source, limit, seed: base[source])
    monkeypatch.setattr(DataLoader, "load_ood_test_dataset", staticmethod(dict))
    previous = os.umask(mask)
    try:
        background._build_or_load(pd.DataFrame({"text": ["an unrelated test question"]}), 24, 6, check=False)
    finally:
        os.umask(previous)
    files = list(tmp_path.iterdir())
    assert len(files) == 2
    assert all(stat.S_IMODE(f.stat().st_mode) == 0o666 & ~mask for f in files)


def test_atomic_write_preserves_other_threads_file_permissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = tmp_path / "cache.json"
    unrelated = tmp_path / "private.txt"
    create_unrelated = Event()
    real_umask = os.umask

    def write_unrelated() -> None:
        assert create_unrelated.wait(timeout=5)
        unrelated.write_text("private application data")

    previous = real_umask(0o077)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            other_write = executor.submit(write_unrelated)

            def interleave_umask(mask: int) -> int:
                old = real_umask(mask)
                if mask == 0:
                    # Let the other thread create its file while permissions
                    # are relaxed, making the former race deterministic.
                    create_unrelated.set()
                    other_write.result(timeout=5)
                return old

            monkeypatch.setattr(background.os, "umask", interleave_umask)

            def write_cache(path: Path) -> None:
                create_unrelated.set()
                other_write.result(timeout=5)
                path.write_text("complete cache")

            background._atomic_write(cache, write_cache)
        assert real_umask(0o077) == 0o077
    finally:
        create_unrelated.set()
        real_umask(previous)

    assert cache.read_text() == "complete cache"
    assert stat.S_IMODE(cache.stat().st_mode) == 0o600
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o600
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cache.json", "private.txt"]


def test_detokenize_wikitext_removes_tokenization_artifacts() -> None:
    raw = (
        ' " Finn the Human " is the b @-@ side of CryoSat @-@ 2 ( 2012 ) , '
        "costing $ 1 @,@ 000 or 3 @.@ 5 % of Tanzler 's budget ; it didn 't sell ... "
    )
    text = detokenize_wikitext(raw)
    assert text == (
        '"Finn the Human" is the b-side of CryoSat-2 (2012), '
        "costing $1,000 or 3.5% of Tanzler's budget; it didn't sell..."
    )
    assert normalize(text) == normalize(raw)


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
