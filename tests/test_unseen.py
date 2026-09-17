"""Offline unit tests for GQR-unseen (no downloads).

Run: uv run --with pytest pytest tests
"""

import math
import random
from pathlib import Path

import pandas as pd
import pytest

from gqr.core.background import OverlapIndex
from gqr.core.dataloader import FINANCE_SOURCE
from gqr.core.unseen import (
    UNSEEN_ID_SETS,
    is_finance_question,
    legal_qa_question,
    reddit_post,
    select_unseen,
    unseen_scores,
)


def test_sets_are_balanced_pinned_and_unseen() -> None:
    domains = [domain for domain, *_ in UNSEEN_ID_SETS.values()]
    assert sorted(domains) == sorted(["law", "finance", "healthcare"] * 3)
    assert all(len(revision) == 40 for *_, revision in UNSEEN_ID_SETS.values())
    # the GQR-Bench finance training source must never be an unseen set
    assert all(repo != FINANCE_SOURCE for _, repo, *_ in UNSEEN_ID_SETS.values())


def test_select_unseen_screens_dedups_and_ignores_order() -> None:
    seen = OverlapIndex(["What is the statute of limitations for breach of contract in Texas"])
    law = [f"can my landlord keep deposit number {i} after I moved out" for i in range(30)]
    law += [
        "what is the statute of limitations for breach of contract in texas today",  # shingle overlap
        "hi",  # too short
        law[0].upper(),  # duplicate after normalization
    ]
    finance = [f"why was my card declined at shop number {i}" for i in range(30)] + [law[1]]
    shuffled = law[:]
    random.Random(3).shuffle(shuffled)

    df, stats = select_unseen({"legal_reddit": law, "banking77": finance}, seen, per_set=10, seed=42)
    df2, _ = select_unseen({"legal_reddit": shuffled, "banking77": finance}, seen, per_set=10, seed=42)

    assert df["text"].tolist() == df2["text"].tolist()
    assert stats["legal_reddit"] == {"raw": 33, "too_short": 1, "duplicate": 1, "gqr_overlap": 1, "used": 10}
    assert set(df[df["dataset"] == "legal_reddit"]["label"]) == {0}
    assert set(df[df["dataset"] == "banking77"]["label"]) == {1}
    assert len(df) == 20
    assert df["text"].str.lower().duplicated().sum() == 0


def test_select_unseen_drops_texts_claimed_by_an_earlier_set() -> None:
    shared = "how do I appeal a parking ticket in my city"
    df, stats = select_unseen(
        {"banking77": [shared], "legal_reddit": [shared, "can my employer read my private messages"]},
        OverlapIndex(),
        per_set=10,
    )
    assert stats["legal_reddit"]["duplicate"] == 1
    assert df[df["dataset"] == "legal_reddit"]["text"].tolist() == ["can my employer read my private messages"]


def test_unseen_scores_macro_average_over_sets() -> None:
    unseen_id = pd.DataFrame(
        {"dataset": ["a", "a", "b"], "label": [0, 0, 1], "predictions": [0, 3, 1]}
    )
    ood = pd.DataFrame(
        {"dataset": ["x", "x", "y"], "label": [3, 3, 3], "predictions": [3, 2, 3]}
    )
    s = unseen_scores(unseen_id, ood)
    assert math.isclose(s["unseen_id_accuracy"], 0.75)  # (0.5 + 1.0) / 2, not pooled 2/3
    assert math.isclose(s["ood_accuracy"], 0.75)
    assert math.isclose(s["gqr_unseen_score"], 0.75)
    assert s["unseen_id_per_dataset"] == {"a": 0.5, "b": 1.0}
    assert s["ood_per_dataset"] == {"x": 0.5, "y": 1.0}


def test_select_unseen_representative_does_not_depend_on_row_order() -> None:
    group = ["Why was my card declined at the shop?", "WHY WAS MY CARD DECLINED AT THE SHOP!"]
    others = [f"how do I order a new card number {i}" for i in range(5)]
    forward, _ = select_unseen({"banking77": group + others}, OverlapIndex(), per_set=100)
    backward, _ = select_unseen({"banking77": others + group[::-1]}, OverlapIndex(), per_set=100)
    # the duplicate group is certainly selected (per_set exceeds the candidates)
    assert len(forward) == 6
    assert forward["text"].tolist() == backward["text"].tolist()
    assert min(group) in forward["text"].tolist()


def test_unavailable_ood_set_is_screened_after_it_becomes_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from gqr.core import unseen

    leaking = "please help my landlord kept the whole security deposit after I moved out"
    clean = [f"can my employer read my private messages at work number {i}" for i in range(5)]
    empty = pd.DataFrame({"text": pd.Series(dtype=str)})
    ood_available = {"flag": False}

    def fake_ood() -> dict[str, pd.DataFrame]:
        dkhate = pd.DataFrame({"text": [leaking]}) if ood_available["flag"] else empty
        return {"jigsaw": pd.DataFrame({"text": ["some toxic comment here"]}), "dkhate": dkhate}

    monkeypatch.setenv("GQR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(unseen.DataLoader, "load_train_dataset", staticmethod(lambda: (empty, empty, empty)))
    monkeypatch.setattr(unseen.DataLoader, "load_ood_test_dataset", staticmethod(fake_ood))
    monkeypatch.setattr(unseen, "load_background_dataset", lambda: (empty, empty))
    monkeypatch.setattr(unseen, "_texts", lambda name: [leaking, *clean] if name == "legal_reddit" else [])

    with pytest.warns(UserWarning, match="dkhate"):
        first = unseen.load_unseen_id_test_dataset(per_set=10)
    assert leaking in first["text"].tolist()  # could not be screened ...
    assert not list(tmp_path.glob("*.parquet"))  # ... so nothing was cached

    ood_available["flag"] = True
    second = unseen.load_unseen_id_test_dataset(per_set=10)
    assert leaking not in second["text"].tolist()
    assert sorted(second["text"]) == sorted(clean)
    assert list(tmp_path.glob("*.parquet"))  # a complete build is cached


def test_source_specific_cleaning() -> None:
    assert is_finance_question("How much cash did the company pay for income taxes in 2023?")
    assert not is_finance_question("What are the primary pillars of FedEx's community engagement program?")
    assert legal_qa_question("Q: Can I sue the hospital for negligence?") == "Can I sue the hospital for negligence?"
    assert legal_qa_question("Three Features of a Kangaroo Court") is None


def test_reddit_post_unescapes_and_drops_scheduled_threads() -> None:
    assert reddit_post("Rent &amp;amp; deposit", "Landlord said &amp;gt;30 days&amp;amp;#x200B;") == (
        "Rent & deposit\nLandlord said >30 days"
    )
    assert reddit_post("Evicted without notice", None) == "Evicted without notice"
    for title in (
        "Daily General Discussion - November 10, 2017",
        "Premarket Thread for General Trading and Plans for Friday, April 30, 2021",
        "GME Megathread for March 04, 2021",
        "Weekend Thread for General Discussion and Plans for Saturday",
    ):
        assert reddit_post(title, "body") is None
    for title in ("21 year old with 5k to invest - any advice welcomed", "The biggest mistake, repeated daily"):
        assert reddit_post(title, "body") is not None


def test_cached_build_is_checked_against_the_reference(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from gqr.core import unseen

    monkeypatch.setenv("GQR_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(unseen, "EXPECTED_SHA256", "0" * 64)
    cached = pd.DataFrame({"text": ["a cached query here"], "label": [0], "domain": ["law"], "dataset": ["legal_reddit"]})
    cached.to_parquet(tmp_path / f"{unseen.BUILD_ID}_seed42_per1000.parquet", index=False)
    with pytest.warns(UserWarning, match="differs from the reference"):
        unseen.load_unseen_id_test_dataset()
