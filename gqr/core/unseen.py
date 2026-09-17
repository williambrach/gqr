"""GQR-unseen: in-domain test sets from sources GQR-Bench never uses.

The GQR-Bench ID test set comes from the same three sources as the training data,
so ID accuracy there says little about how routing carries over to queries from
other sources. GQR-unseen replaces the ID half of the benchmark with nine
in-domain test sets from new sources, three per domain, and keeps the existing
seven OOD test sets:

* finance: banking77 customer queries, financial-qa-10K questions about annual
  reports (finance-keyword questions only), finance and investing Reddit posts
* healthcare: iCliniq patient questions, MedQuAD definitional questions, medical
  flashcards
* law: r/legaladvice posts, legal-qa-v1 user questions, MMLU professional-law
  fact patterns

None of these sources is used by GQR-Bench v1 or v2 training, evaluation or test
data, and each dataset is pinned to a commit revision. Every candidate must pass
a leakage screen: no exact or 8-word-shingle overlap with GQR-Bench train, eval,
ID test, OOD test, or the v2 background class. Candidates are deduplicated on
normalized text across all sets. Each set keeps at most ``PER_SET`` texts with the
smallest seeded content hash, so the build does not depend on row order or
library versions, and ``EXPECTED_SHA256`` pins it.

The GQR-unseen score is the harmonic mean of unseen ID accuracy (macro-averaged
over the nine sets, so every source weighs the same) and OOD accuracy on the
existing GQR-Bench OOD test sets (macro-averaged over datasets, as in the GQR
score). The existing test sets are unchanged. The built sets are cached as
parquet under ``$GQR_CACHE_DIR`` (default ``~/.cache/gqr``).
"""

from __future__ import annotations

import json
import re
import warnings
from collections.abc import Callable, Iterable

import pandas as pd

from .background import (
    ID_TOPIC_PATTERNS,
    OverlapIndex,
    _atomic_write,
    _cache_dir,
    fingerprint,
    load_background_dataset,
    normalize,
    rank_key,
)
from .dataloader import SEED, DataLoader, domain2label, load_ood_test_dataset

BUILD_ID = "unseen-id-audited"
PER_SET = 1_000
MIN_WORDS = 3
MAX_CHARS = 4_000

# name -> (domain, repo, config, split, revision)
UNSEEN_ID_SETS: dict[str, tuple[str, str, str | None, str, str]] = {
    "banking77": ("finance", "mteb/banking77", None, "test",
                  "18072d2685ea682290f7b8924d94c62acc19c0b2"),
    "financial_qa_10k": ("finance", "virattt/financial-qa-10K", None, "train",
                         "30e48b49ef40389cce6488b0bddef7d6f00f4c7a"),
    "reddit_finance": ("finance", "winddude/reddit_finance_43_250k", None, "train",
                       "fc980546bdf17618646cc114690cc89fa6dfbf55"),
    "icliniq": ("healthcare", "lavita/ChatDoctor-iCliniq", None, "train",
                "f2220c8483eb01c0e962ee37b5d32f4879fb0b74"),
    "medquad": ("healthcare", "keivalya/MedQuad-MedicalQnADataset", None, "train",
                "5b0961fbaa6d7f9c344c5d59c29943fb900c2eca"),
    "med_flashcards": ("healthcare", "medalpaca/medical_meadow_medical_flashcards", None,
                       "train", "7597b32036d67c731cb91bae4f49717fcfe5d5f0"),
    "legal_reddit": ("law", "jonathanli/legal-advice-reddit", None, "test",
                     "f105b9d763743e20d2f3b8e33f73055ad414e7c5"),
    "legal_qa_v1": ("law", "dzunggg/legal-qa-v1", None, "train",
                    "6280beb74faf5b4dfd1f63adbf7d18908b377b93"),
    "mmlu_professional_law": ("law", "cais/mmlu", "professional_law", "test",
                              "c30699e8356da336a370243923dbaf21066bb9fe"),
}  # fmt: skip
REDDIT_FINANCE_ROWS = 40_000  # the first rows of the pinned revision; the full set is ~250k posts
_FINANCE_RE = re.compile(ID_TOPIC_PATTERNS["finance"], re.IGNORECASE)


def is_finance_question(text: str) -> bool:
    """financial-qa-10K asks about every part of an annual report (store counts, ESG programs,
    litigation); keep only questions with a finance keyword, so its label is actually finance."""
    return bool(_FINANCE_RE.search(str(text)))


def legal_qa_question(text: str) -> str | None:
    """legal-qa-v1 mixes user questions ("Q: ...") with article titles; keep the questions and
    drop the "Q:" marker, which would identify the source."""
    text = str(text or "").strip()
    return text[2:].strip() if text.startswith("Q:") else None

# sha256 over the texts of the default build (set order of UNSEEN_ID_SETS, then hash order).
EXPECTED_SHA256: str | None = None


def _texts(name: str) -> Iterable[str]:
    """Raw query texts of one unseen set at its pinned revision."""
    from datasets import load_dataset

    _, repo, config, split, revision = UNSEEN_ID_SETS[name]
    ds = load_dataset(repo, config, split=split, revision=revision)
    if name == "reddit_finance":
        ds = ds.select(range(min(REDDIT_FINANCE_ROWS, len(ds))))
        return (f"{r['title']}\n{r['selftext'] or ''}" for r in ds)
    if name == "legal_reddit":
        return (f"{r['title']}\n{r['body']}" for r in ds)
    if name == "financial_qa_10k":
        return (q for q in ds["question"] if is_finance_question(q))
    if name == "legal_qa_v1":
        return (q for q in map(legal_qa_question, ds["question"]) if q)
    column = {
        "banking77": "text",
        "icliniq": "input",
        "medquad": "Question",
        "med_flashcards": "input",
        "mmlu_professional_law": "question",
    }[name]
    return ds[column]


def select_unseen(
    raw: dict[str, Iterable[str]],
    seen: OverlapIndex,
    per_set: int = PER_SET,
    seed: int = SEED,
) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Clean, screen and sample the unseen ID sets.

    ``raw`` maps set name -> texts (in any order); sets are processed in
    ``UNSEEN_ID_SETS`` order and a text claimed by an earlier set is dropped from
    later ones. Returns the rows (text, label, domain, dataset) and per-set filter
    statistics.
    """
    claimed: set[str] = set()
    frames, stats = [], {}
    for name, (domain, *_) in UNSEEN_ID_SETS.items():
        if name not in raw:
            continue
        s = {"raw": 0, "too_short": 0, "duplicate": 0, "gqr_overlap": 0}
        candidates: dict[str, str] = {}
        leaked: set[str] = set()
        for text in raw[name]:
            s["raw"] += 1
            text = str(text or "").strip()[:MAX_CHARS]
            norm = normalize(text)
            if len(norm.split()) < MIN_WORDS:
                s["too_short"] += 1
                continue
            if norm in candidates or norm in leaked or norm in claimed:
                s["duplicate"] += 1
                if norm in candidates:
                    # spellings that normalize identically: keep the lexicographically smallest,
                    # so the emitted query does not depend on row order
                    candidates[norm] = min(candidates[norm], text)
                continue
            if seen.hit(text):  # overlap is decided on normalized text, so any spelling gives the same answer
                s["gqr_overlap"] += 1
                leaked.add(norm)
                continue
            candidates[norm] = text
        ranked = sorted(candidates, key=lambda n: rank_key(n, seed, f"unseen:{name}"))[:per_set]
        claimed.update(ranked)
        s["used"] = len(ranked)
        stats[name] = s
        frames.append(
            pd.DataFrame(
                {
                    "text": [candidates[n] for n in ranked],
                    "label": domain2label[domain],
                    "domain": domain,
                    "dataset": name,
                }
            )
        )
    columns = ["text", "label", "domain", "dataset"]
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)), stats


def _seen_index() -> tuple[OverlapIndex, list[str]]:
    """Everything a GQR-Bench v1/v2 router may have seen, plus both test sets.

    Returns the index and the names of OOD test sets that could not be loaded
    (DDSC/dkhate is gated); an index with missing sets screens incompletely.
    """
    train_df, eval_df, id_test_df = DataLoader.load_train_dataset()
    background_train, background_eval = load_background_dataset()
    index = OverlapIndex()
    for frame in (train_df, eval_df, id_test_df, background_train, background_eval):
        index.add(frame["text"].astype(str))
    missing = []
    for name, frame in DataLoader.load_ood_test_dataset().items():
        if frame.empty:
            missing.append(name.strip())
        index.add(frame["text"].astype(str))
    return index, missing


def load_unseen_id_test_dataset(seed: int = SEED, per_set: int = PER_SET) -> pd.DataFrame:
    """Unseen in-domain test sets, built and cached on first use.

    Columns: text, label (0 law, 1 finance, 2 healthcare), domain, dataset.
    """
    cache = _cache_dir() / f"{BUILD_ID}_seed{seed}_per{per_set}.parquet"
    if cache.exists():
        try:
            df = pd.read_parquet(cache)
            if df.empty or not {"text", "label", "domain", "dataset"} <= set(df.columns):
                raise ValueError("expected text, label, domain and dataset columns")
            return df
        except Exception as e:
            warnings.warn(f"Ignoring unreadable GQR-unseen cache {cache} ({e}); rebuilding.", stacklevel=2)
    seen, missing = _seen_index()
    raw = {name: _texts(name) for name in UNSEEN_ID_SETS}
    df, stats = select_unseen(raw, seen, per_set, seed)
    if missing:
        # Not cached, so the next call (for example after logging in to the Hugging Face
        # Hub) rebuilds with the full overlap check.
        warnings.warn(
            f"GQR-unseen was not screened for overlap with the OOD test set(s) {missing}, "
            f"which could not be loaded (DDSC/dkhate is gated: log in to the Hugging Face Hub "
            f"and accept its terms). The sets are used for this call but not cached, and "
            f"their fingerprint is not the reference one.",
            stacklevel=2,
        )
        return df
    digest = fingerprint(df["text"])
    if seed == SEED and per_set == PER_SET and EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        warnings.warn(
            f"GQR-unseen fingerprint {digest[:12]} differs from the reference "
            f"{EXPECTED_SHA256[:12]}; an upstream dataset has changed.",
            stacklevel=2,
        )
    stats_json = json.dumps({"sha256": digest, "rows": len(df), "sets": stats}, indent=2)
    _atomic_write(cache.with_suffix(".stats.json"), lambda path: path.write_text(stats_json))
    _atomic_write(cache, lambda path: df.to_parquet(path, index=False))
    return df


def unseen_scores(
    unseen_id: pd.DataFrame, ood: pd.DataFrame, pred_col: str = "predictions"
) -> dict:
    """GQR-unseen from predictions on the unseen ID sets and on the GQR-Bench OOD test sets.

    Both frames need label, dataset and prediction columns.
    """

    def per_dataset(frame: pd.DataFrame) -> pd.Series:
        correct = frame[pred_col].astype(int) == frame["label"].astype(int)
        return correct.groupby(frame["dataset"]).mean()

    id_sets, ood_sets = per_dataset(unseen_id), per_dataset(ood)
    id_accuracy, ood_accuracy = float(id_sets.mean()), float(ood_sets.mean())
    total = id_accuracy + ood_accuracy
    return {
        "unseen_id_accuracy": id_accuracy,
        "ood_accuracy": ood_accuracy,
        "gqr_unseen_score": 2 * id_accuracy * ood_accuracy / total if total > 0 else 0.0,
        "unseen_id_per_dataset": {name: float(acc) for name, acc in id_sets.items()},
        "ood_per_dataset": {name: float(acc) for name, acc in ood_sets.items()},
    }


def score_unseen_batch(
    model_fn: Callable[[list[str]], list[int]], batch_size: int = 32
) -> dict:
    """GQR-unseen score of a batched model_fn (list of texts -> labels in {0, 1, 2, 3})."""

    def predict(frame: pd.DataFrame, name: str) -> pd.DataFrame:
        texts = frame["text"].astype(str).tolist()
        print(f"[GQR-unseen] Running model on {len(texts)} {name} texts in batches of {batch_size}...")
        predictions: list[int] = []
        for i in range(0, len(texts), batch_size):
            predictions.extend(model_fn(texts[i : i + batch_size]))
        return frame.assign(predictions=predictions)

    scores = unseen_scores(
        predict(load_unseen_id_test_dataset(), "unseen ID"),
        predict(load_ood_test_dataset(), "OOD test"),
    )
    summary = {k: v for k, v in scores.items() if not k.endswith("per_dataset")}
    print(f"[GQR-unseen] Final scores: {summary}")
    return scores


def score_unseen(model_fn: Callable[[str], int]) -> dict:
    """GQR-unseen score of a per-text model_fn (text -> label in {0, 1, 2, 3})."""
    return score_unseen_batch(lambda batch: [model_fn(text) for text in batch], batch_size=256)
