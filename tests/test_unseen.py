"""Offline unit tests for GQR-unseen (no downloads).

Run: uv run --with pytest pytest tests
"""

import math
import random

import pandas as pd

from gqr.core.background import OverlapIndex
from gqr.core.unseen import UNSEEN_SETS, select_unseen, unseen_scores


def test_sets_are_balanced_and_pinned() -> None:
    kinds = [kind for kind, *_ in UNSEEN_SETS.values()]
    assert kinds.count("id") == 9 and kinds.count("ood") == 7
    for domain in ("law", "finance", "healthcare"):
        assert sum(1 for k, d, *_ in UNSEEN_SETS.values() if k == "id" and d == domain) == 3
    assert all(len(revision) == 40 for *_, revision in UNSEEN_SETS.values())
    # the GQR-Bench finance training source must never be an unseen set
    assert all(repo != "DeividasM/financial-instruction-aq22" for _, _, repo, *_ in UNSEEN_SETS.values())


def test_select_unseen_screens_dedups_and_ignores_order() -> None:
    seen = OverlapIndex(["What is the statute of limitations for breach of contract in Texas"])
    law = [f"can my landlord keep deposit number {i} after I moved out" for i in range(30)]
    law += [
        "what is the statute of limitations for breach of contract in texas today",  # shingle overlap
        "hi",  # too short
        law[0].upper(),  # duplicate after normalization
    ]
    ood = [f"who won the football match on sunday number {i}" for i in range(30)] + [law[1]]
    shuffled = law[:]
    random.Random(3).shuffle(shuffled)

    df, stats = select_unseen({"legal_reddit": law, "trec": ood}, seen, per_set=10, seed=42)
    df2, _ = select_unseen({"legal_reddit": shuffled, "trec": ood}, seen, per_set=10, seed=42)

    assert df["text"].tolist() == df2["text"].tolist()
    assert stats["legal_reddit"] == {"raw": 33, "too_short": 1, "duplicate": 1, "gqr_overlap": 1, "used": 10}
    assert set(df[df["dataset"] == "legal_reddit"]["label"]) == {0}
    assert set(df[df["dataset"] == "trec"]["label"]) == {3}
    assert len(df) == 20
    assert df["text"].str.lower().duplicated().sum() == 0


def test_select_unseen_drops_texts_claimed_by_an_earlier_set() -> None:
    shared = "how do I appeal a parking ticket in my city"
    df, stats = select_unseen(
        {"legal_reddit": [shared], "trec": [shared, "who painted the mona lisa originally"]},
        OverlapIndex(),
        per_set=10,
    )
    assert stats["trec"]["duplicate"] == 1
    assert df[df["dataset"] == "trec"]["text"].tolist() == ["who painted the mona lisa originally"]


def test_unseen_scores_macro_average_over_sets() -> None:
    data = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "c", "c"],
            "kind": ["id", "id", "id", "ood", "ood"],
            "label": [0, 0, 1, 3, 3],
            "predictions": [0, 3, 1, 3, 2],
        }
    )
    s = unseen_scores(data)
    assert math.isclose(s["id_accuracy"], 0.75)  # (0.5 + 1.0) / 2, not pooled 2/3
    assert math.isclose(s["ood_accuracy"], 0.5)
    assert math.isclose(s["gqr_unseen_score"], 2 * 0.75 * 0.5 / 1.25)
    assert s["per_dataset"] == {"a": 0.5, "b": 1.0, "c": 0.5}
