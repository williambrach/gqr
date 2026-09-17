"""Offline unit tests for GQR-unseen (no downloads).

Run: uv run --with pytest pytest tests
"""

import math
import random

import pandas as pd

from gqr.core.background import OverlapIndex
from gqr.core.dataloader import FINANCE_SOURCE
from gqr.core.unseen import UNSEEN_ID_SETS, select_unseen, unseen_scores


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
