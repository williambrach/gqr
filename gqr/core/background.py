"""GQR-Bench v2: background (4th-class) training data.

GQR-Bench v1 ships training data for the three in-distribution domains only, so
a router can learn to reject out-of-distribution queries only indirectly (for
example by thresholding confidence). v2 adds a fourth *background* class
(label 3, domain "ood") to the train and eval splits. It is built only from
general-purpose corpora that are disjoint from every GQR-Bench test set:

* wikitext-103 prose (``Salesforce/wikitext``, ``wikitext-103-raw-v1``, train)
* dolly-15k instructions (``databricks/databricks-dolly-15k``, train)
* Yahoo Answers questions from topics outside the three domains
  (``community-datasets/yahoo_answers_topics``, train; Society & Culture,
  Science & Mathematics, Education & Reference, Computers & Internet)

Every candidate passes three filters:

1. it contains no law / finance / healthcare keyword (``ID_TOPIC_PATTERNS``),
   so the background class never teaches the router to reject on-topic text;
2. it is not confidently in-domain (max probability <= 0.99) under a TF-IDF +
   logistic-regression classifier trained on the v1 training split;
3. it has no exact or 8-word-shingle overlap with the ID or OOD test sets.

The three sources contribute equally, and the class is sized like one
in-distribution domain in each split. Test sets are unchanged, so v1 and v2
scores are directly comparable. The built corpus is cached as parquet under
``$GQR_CACHE_DIR`` (default ``~/.cache/gqr``).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .dataloader import SEED, DataLoader, domain2label

BACKGROUND_DOMAIN = "ood"
BACKGROUND_LABEL = domain2label[BACKGROUND_DOMAIN]
SOURCES = ("wikitext", "dolly", "yahoo")
YAHOO_TOPICS = (0, 1, 3, 4)
MAX_CHARS = 2_000
MIN_CHARS = 40
SHINGLE_WORDS = 8
CLF_MAX_PROBA = 0.99
OVERSAMPLE = 2.5
STREAM_BUFFER = 50_000

# sha256 over the default v2 background texts (train split, then eval split).
# A mismatch means an upstream corpus changed and the build is not the
# reference GQR-Bench v2 background.
EXPECTED_SHA256: str | None = (
    "8cb99a014bcdd9d8340cd4c1973ae63c65c75143f7ba70395a66389265fea98c"
)

ID_TOPIC_PATTERNS = {
    "finance": r"\b(stocks?|invest\w*|bank\w*|loans?|tax\w*|mortgage|credit|insurance|salary|401k|ira|dividends?|mutual funds?|hedge funds?|budget\w*|debts?|interest rates?|currenc\w*|bitcoin|crypto\w*|accounting|revenue|profits?|inflation|portfolio|equity|bonds?|shares?|trading|broker\w*|pension|retire\w*|payroll|wages?|financ\w*|econom\w*|gdp|paycheck|refinanc\w*|fico|debit|money|price\w*|cost\w*|pay\w*|income|fund\w*|cash|dollars?|euros?)\b",
    "healthcare": r"\b(doctors?|symptoms?|pain|diseases?|medic\w*|hospital\w*|patients?|diagnos\w*|treatments?|pills?|drugs?|cancer|diabet\w*|infection\w*|blood|surger\w*|pregnan\w*|health\w*|therap\w*|vaccin\w*|virus\w*|fever|rash|heart|tumou?r|clinic\w*|prescri\w*|dosage|antibiotic\w*|allerg\w*|asthma|injur\w*|nurse\w*|physician\w*|pharmac\w*|disorder\w*|anxiety|depress\w*|ache|cough|flu|covid|dental|dentist|kidney|liver|lungs?|skin|stomach|bones?|muscles?|diet|weight loss|obes\w*|sick\w*|ill\w*|headache|migraine|cholesterol|pressure|vitamin\w*|supplement\w*|sleep\w*|insomnia|periods?|menstru\w*|sex\w*|std|hiv|aids|body|brain|nerv\w*|swell\w*|bleed\w*|eye|ear|throat|nose|chest|back pain|knee|hip|shoulder|bowel|urin\w*|penis|vagina|breast|testicle\w*)\b",
    "law": r"\b(laws?|legal\w*|courts?|judges?|attorney\w*|lawyers?|sue[ds]?|lawsuits?|contracts?|police|arrest\w*|crim\w*|illegal|statute\w*|copyright\w*|patents?|tenants?|landlords?|custody|divorce\w*|visas?|immigration|constitution\w*|jurisdiction|liabilit\w*|plaintiff|defendant|felony|misdemeanor|prosecut\w*|warrant|legislat\w*|regulation\w*|rights|leases?|evict\w*|trademark\w*|licens\w*|fine[ds]?|penalt\w*|jail|prison|bail|charges?|charged|testif\w*|witness\w*|notary|will|inherit\w*|estate|trust|alimony|harass\w*|discriminat\w*|employer|fired|terminated|hoa|dui|dwi|ticket\w*|speeding|permit\w*|zoning|lawful|unlawful|agreement\w*|clause\w*|terms of service|gdpr|privacy)\b",
}
_ID_TOPIC_RE = {k: re.compile(v, re.IGNORECASE) for k, v in ID_TOPIC_PATTERNS.items()}
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")


def id_topic_hit(text: str) -> str | None:
    """Return the first in-distribution domain whose keywords occur in text."""
    for domain, pattern in _ID_TOPIC_RE.items():
        if pattern.search(text):
            return domain
    return None


def normalize(text: str) -> str:
    return _WS.sub(" ", _PUNCT.sub(" ", str(text).lower())).strip()


def shingles(text: str, k: int = SHINGLE_WORDS) -> set[str]:
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class OverlapIndex:
    """Exact (normalized) and k-word-shingle overlap lookup."""

    def __init__(self, texts: Iterable[str] = ()) -> None:
        self.exact: set[str] = set()
        self.shingles: set[int] = set()
        self.add(texts)

    def add(self, texts: Iterable[str]) -> None:
        for text in texts:
            norm = normalize(text)
            if not norm:
                continue
            self.exact.add(norm)
            self.shingles.update(_digest(s) for s in shingles(text))

    def hit(self, text: str) -> bool:
        if normalize(text) in self.exact:
            return True
        return any(_digest(s) in self.shingles for s in shingles(text))


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest())


def fingerprint(texts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


def select_background(
    pools: dict[str, list[str]],
    id_max_proba: Callable[[list[str]], np.ndarray],
    test_index: OverlapIndex,
    n_total: int,
    seed: int = SEED,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Filter the keyword-clean candidate pools and draw a source-balanced sample.

    Returns (texts, sources, stats). Sources contribute equally; the result is
    shuffled with ``seed``. Raises ValueError if a source runs out of candidates.
    """
    per_source = math.ceil(n_total / len(pools))
    rng = np.random.default_rng(seed)
    stats: dict[str, int] = {}
    texts: list[str] = []
    sources: list[str] = []
    for name, pool in pools.items():
        pool = [t for t in dict.fromkeys(pool) if id_topic_hit(t) is None]
        stats[f"{name}_keyword_clean"] = len(pool)
        if pool:
            keep = np.asarray(id_max_proba(pool)) <= CLF_MAX_PROBA
            stats[f"{name}_classifier_dropped"] = int((~keep).sum())
            pool = [t for t, k in zip(pool, keep, strict=True) if k]
        leaked = [test_index.hit(t) for t in pool]
        stats[f"{name}_test_overlap_dropped"] = int(sum(leaked))
        pool = [t for t, bad in zip(pool, leaked, strict=True) if not bad]
        if len(pool) < per_source:
            raise ValueError(
                f"background source {name!r} has {len(pool)} clean candidates, "
                f"needs {per_source}; stats so far: {stats}"
            )
        order = rng.permutation(len(pool))[:per_source]
        texts.extend(pool[i] for i in order)
        sources.extend([name] * per_source)
        stats[f"{name}_used"] = per_source
    order = rng.permutation(len(texts))[:n_total]
    return [texts[i] for i in order], [sources[i] for i in order], stats


def _wikitext_candidates(limit: int, seed: int) -> list[str]:
    from datasets import load_dataset

    stream = load_dataset(
        "Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True
    ).shuffle(seed=seed, buffer_size=STREAM_BUFFER)
    out: list[str] = []
    for row in stream:
        text = row["text"].strip()
        if len(text) >= MIN_CHARS and not text.startswith("=") and id_topic_hit(text) is None:
            out.append(text[:MAX_CHARS])
            if len(out) >= limit:
                break
    return out


def _dolly_candidates(seed: int) -> list[str]:
    from datasets import load_dataset

    dolly = load_dataset("databricks/databricks-dolly-15k", split="train").shuffle(seed=seed)
    return [
        text[:MAX_CHARS]
        for text in (str(t).strip() for t in dolly["instruction"])
        if len(text) >= 15 and id_topic_hit(text) is None
    ]


def _yahoo_candidates(limit: int, seed: int) -> list[str]:
    from datasets import load_dataset

    stream = load_dataset(
        "community-datasets/yahoo_answers_topics", split="train", streaming=True
    ).shuffle(seed=seed, buffer_size=STREAM_BUFFER)
    out: list[str] = []
    for row in stream:
        if row["topic"] not in YAHOO_TOPICS:
            continue
        text = f"{row['question_title']} {row['question_content']}".strip()
        if len(text.split()) >= 4 and id_topic_hit(text) is None:
            out.append(text[:MAX_CHARS])
            if len(out) >= limit:
                break
    return out


def _id_classifier(train_df: pd.DataFrame) -> Callable[[list[str]], np.ndarray]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    clf = make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_features=200_000),
        LogisticRegression(max_iter=1_000),
    )
    clf.fit(train_df["text"].astype(str).tolist(), train_df["label"].tolist())
    return lambda texts: clf.predict_proba(texts).max(axis=1)


def _cache_dir() -> Path:
    return Path(os.environ.get("GQR_CACHE_DIR", Path.home() / ".cache" / "gqr"))


def _background_frame(texts: list[str], sources: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": texts,
            "domain": BACKGROUND_DOMAIN,
            "label": BACKGROUND_LABEL,
            "source": sources,
        }
    )


def _build_or_load(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    id_test_df: pd.DataFrame,
    n_train: int,
    n_eval: int,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = _cache_dir() / f"background_v2_seed{seed}_train{n_train}_eval{n_eval}.parquet"
    if cache.exists():
        df = pd.read_parquet(cache)
    else:
        n_total = n_train + n_eval
        limit = math.ceil(math.ceil(n_total / len(SOURCES)) * OVERSAMPLE)
        pools = {
            "wikitext": _wikitext_candidates(limit, seed),
            "dolly": _dolly_candidates(seed),
            "yahoo": _yahoo_candidates(limit, seed),
        }
        ood_tests = DataLoader.load_ood_test_dataset()
        test_index = OverlapIndex(id_test_df["text"].astype(str))
        for frame in ood_tests.values():
            test_index.add(frame["text"].astype(str))
        texts, sources, stats = select_background(
            pools, _id_classifier(train_df), test_index, n_total, seed
        )
        df = _background_frame(texts, sources)
        df["split"] = ["train"] * n_train + ["eval"] * n_eval
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache, index=False)
        stats["sha256"] = fingerprint(texts)
        cache.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
    digest = fingerprint(df["text"])
    default_size = (n_train, n_eval) == _default_sizes(train_df, eval_df)
    if default_size and EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        warnings.warn(
            f"GQR-Bench v2 background fingerprint {digest[:12]} differs from the "
            f"reference {EXPECTED_SHA256[:12]}; an upstream corpus has changed.",
            stacklevel=3,
        )
    train_bg = df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)
    eval_bg = df[df["split"] == "eval"].drop(columns="split").reset_index(drop=True)
    return train_bg, eval_bg


def _default_sizes(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[int, int]:
    n_domains = train_df["domain"].nunique()
    return round(len(train_df) / n_domains), round(len(eval_df) / n_domains)


def load_background_dataset(
    n_train: int | None = None, n_eval: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (building and caching on first use) the v2 background class.

    By default each split gets as many background rows as one in-distribution
    domain has. Columns: text, domain ("ood"), label (3), source.
    """
    train_df, eval_df, id_test_df = DataLoader.load_train_dataset()
    default_train, default_eval = _default_sizes(train_df, eval_df)
    return _build_or_load(
        train_df,
        eval_df,
        id_test_df,
        default_train if n_train is None else n_train,
        default_eval if n_eval is None else n_eval,
    )


def load_train_dataset_v2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GQR-Bench v2 train and eval splits: v1 splits plus the background class."""
    train_df, eval_df, id_test_df = DataLoader.load_train_dataset()
    train_bg, eval_bg = _build_or_load(
        train_df, eval_df, id_test_df, *_default_sizes(train_df, eval_df)
    )
    out = []
    for split, background in ((train_df, train_bg), (eval_df, eval_bg)):
        split = split.assign(source="gqr")
        combined = pd.concat([split, background], ignore_index=True)
        out.append(combined.sample(frac=1, random_state=SEED).reset_index(drop=True))
    return out[0], out[1]
