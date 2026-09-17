"""GQR-Bench v2: background (4th-class) training data.

GQR-Bench v1 ships training data for the three in-distribution domains only, so
a router can learn to reject out-of-distribution queries only indirectly (for
example by thresholding confidence). v2 adds a fourth *background* class
(label 3, domain "ood") to the train and eval splits. It is built only from
general-purpose corpora, at pinned revisions, that no GQR-Bench test set uses:

* wikitext-103 prose (``Salesforce/wikitext``, ``wikitext-103-raw-v1``, train),
  detokenized so its ``@-@`` markers and padded punctuation do not give it away
* dolly-15k instructions (``databricks/databricks-dolly-15k``, train)
* Yahoo Answers questions from topics outside the three domains
  (``community-datasets/yahoo_answers_topics``, train; Society & Culture,
  Science & Mathematics, Education & Reference, Computers & Internet)

Candidates are deduplicated on normalized text (case, punctuation and
whitespace ignored) within and across sources before the train / eval split, so
the same question never lands in both splits. Every candidate must also pass two
filters:

1. it contains no law / finance / healthcare keyword (``ID_TOPIC_PATTERNS``),
   so the background class never teaches a router to reject on-topic text;
2. it has no exact or 8-word-shingle overlap with the ID or OOD test sets.

Selection is content-addressed: each source keeps the eligible passages with
the smallest seeded BLAKE2b hash, and the final order is again a hash order.
The result therefore does not depend on shard order, streaming or shuffling
implementations, or library versions, and ``EXPECTED_SHA256`` pins it.

The three sources contribute equally, and the class is sized like one
in-distribution domain in each split. Test sets are unchanged, so v1 and v2
scores are directly comparable. The built corpus is cached as parquet under
``$GQR_CACHE_DIR`` (default ``~/.cache/gqr``).
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import tempfile
import warnings
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

from .dataloader import SEED, DataLoader, domain2label

BACKGROUND_DOMAIN = "ood"
BACKGROUND_LABEL = domain2label[BACKGROUND_DOMAIN]
BUILD_ID = "v2-hashrank-dedup-detok"
SOURCES = {
    "wikitext": ("Salesforce/wikitext", "wikitext-103-raw-v1",
                 "b08601e04326c79dfdd32d625aee71d232d685c3"),
    "dolly": ("databricks/databricks-dolly-15k", None,
              "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"),
    "yahoo": ("community-datasets/yahoo_answers_topics", None,
              "6652a1e7c94f7260a0bfd0c9092dd48e2d536ea1"),
}
YAHOO_TOPICS = (0, 1, 3, 4)
MAX_CHARS = 2_000
MIN_CHARS = 40
MIN_DOLLY_CHARS = 15
SHINGLE_WORDS = 8
OVERSAMPLE = 1.25

# sha256 over the default v2 background texts (train split, then eval split).
# A mismatch means an upstream corpus changed and the build is not the
# reference GQR-Bench v2 background.
EXPECTED_SHA256: str | None = (
    "f4f76f57fe34649e2ad97c9374bb04a4ae157eb2826bb14e5df381a570ad8102"
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


_WIKITEXT_DETOKENIZE = (
    (re.compile(r" @(\S)@ "), r"\1"),  # "b @-@ side", "1 @,@ 000", "3 @.@ 5"
    (re.compile(r'"\s*([^"]*?)\s*"'), r'"\1"'),  # '" Finn the Human "'
    (re.compile(r"([(\[$#]) "), r"\1"),
    (re.compile(r" (\.\.\.|[.,;:!?%)\]]|'(?:s|t|re|ve|ll|d|m)\b)"), r"\1"),  # "it 's", "don 't"
    (_WS, " "),
)


def detokenize_wikitext(text: str) -> str:
    """Undo wikitext-103's Moses-style tokenization.

    Raw wikitext splits hyphens and number separators into ``@-@``, ``@,@`` and
    ``@.@`` and pads punctuation with spaces, which no GQR-Bench query does. Left
    in, these artifacts would let a router recognize a background passage by its
    formatting instead of its content. The normalized text is unaffected.
    """
    text = str(text)
    for pattern, replacement in _WIKITEXT_DETOKENIZE:
        text = pattern.sub(replacement, text)
    return text.strip()


def shingles(text: str, k: int = SHINGLE_WORDS) -> set[str]:
    words = normalize(text).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def _digest(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")


def rank_key(text: str, seed: int, salt: str) -> int:
    """Seeded, content-addressed sort key: independent of row order and platform."""
    return _digest(f"{seed}\0{salt}\0{text}")


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


def fingerprint(texts: Iterable[str]) -> str:
    h = hashlib.sha256()
    for text in texts:
        h.update(text.encode())
        h.update(b"\0")
    return h.hexdigest()


def smallest_eligible(texts: Iterable[str], limit: int, seed: int, salt: str) -> list[str]:
    """The ``limit`` keyword-clean texts, distinct after normalization, with the
    smallest rank keys.

    Texts that normalize identically share one rank key and are represented by
    the lexicographically smallest variant. The result depends only on the
    multiset of input texts, not on their order.
    """
    heap: list[tuple[int, str]] = []  # max-heap via negated keys, over normalized texts
    representative: dict[str, str] = {}
    for raw in texts:
        text = str(raw).strip()[:MAX_CHARS]
        norm = normalize(text)
        if not norm:
            continue
        if norm in representative:
            representative[norm] = min(representative[norm], text)
            continue
        key = rank_key(norm, seed, salt)
        if len(heap) >= limit and key >= -heap[0][0]:
            continue
        if id_topic_hit(norm) is not None:
            continue
        if len(heap) < limit:
            heapq.heappush(heap, (-key, norm))
        else:
            _, dropped = heapq.heapreplace(heap, (-key, norm))
            del representative[dropped]
        representative[norm] = text
    return [representative[norm] for _, norm in sorted(heap, key=lambda item: (-item[0], item[1]))]


def select_background(
    pools: dict[str, list[str]],
    test_index: OverlapIndex,
    n_total: int,
    seed: int = SEED,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Filter the candidate pools and draw a source-balanced, hash-ordered sample.

    Returns (texts, sources, stats). Each source contributes the same number of
    passages, and no two returned texts are equal after normalization, so no
    question can appear in both the train and the eval split. A text found in
    several sources belongs to the alphabetically first one. Raises ValueError if
    a source runs out of clean candidates.
    """
    per_source = math.ceil(n_total / len(pools))
    stats: dict[str, int] = {}
    groups: dict[str, dict[str, str]] = {}  # source -> normalized text -> representative
    owner: dict[str, str] = {}  # normalized text -> source
    for name in sorted(pools):
        groups[name] = {}
        for raw in pools[name]:
            text = str(raw)
            norm = normalize(text)
            if not norm or id_topic_hit(norm) is not None:
                continue
            groups[name][norm] = min(groups[name].get(norm, text), text)
            owner.setdefault(norm, name)
    chosen: list[tuple[int, str, str]] = []
    for name, group in groups.items():
        stats[f"{name}_candidates"] = len(group)
        clean = {norm: text for norm, text in group.items() if owner[norm] == name}
        stats[f"{name}_cross_source_duplicates_dropped"] = len(group) - len(clean)
        leaked = {norm for norm, text in clean.items() if test_index.hit(text)}
        stats[f"{name}_test_overlap_dropped"] = len(leaked)
        ranked = sorted((n for n in clean if n not in leaked), key=lambda n: rank_key(n, seed, name))
        clean = [clean[n] for n in ranked]
        if len(clean) < per_source:
            raise ValueError(
                f"background source {name!r} has {len(clean)} clean candidates, "
                f"needs {per_source}; stats so far: {stats}"
            )
        chosen.extend((rank_key(normalize(t), seed, "order"), t, name) for t in clean[:per_source])
        stats[f"{name}_used"] = per_source
    chosen.sort()
    chosen = chosen[:n_total]
    return [t for _, t, _ in chosen], [s for _, _, s in chosen], stats


def _stream(source: str) -> Iterable[dict]:
    from datasets import load_dataset

    repo, config, revision = SOURCES[source]
    return load_dataset(repo, config, split="train", revision=revision, streaming=True)


def _candidates(source: str, limit: int, seed: int) -> list[str]:
    rows = _stream(source)
    if source == "wikitext":
        texts = (
            detokenize_wikitext(r["text"]) for r in rows
            if len(r["text"].strip()) >= MIN_CHARS and not r["text"].strip().startswith("=")
        )
    elif source == "dolly":
        texts = (r["instruction"] for r in rows if len(str(r["instruction"]).strip()) >= MIN_DOLLY_CHARS)
    else:
        texts = (
            f"{r['question_title']} {r['question_content']}".strip() for r in rows
            if r["topic"] in YAHOO_TOPICS
            and len(f"{r['question_title']} {r['question_content']}".split()) >= 4
        )
    return smallest_eligible(texts, limit, seed, source)


def _cache_dir() -> Path:
    return Path(os.environ.get("GQR_CACHE_DIR", Path.home() / ".cache" / "gqr"))


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    """Write on the same filesystem as ``path``, then rename into place.

    An interrupted or concurrent build never leaves a partial file at ``path``.
    The file gets the permissions a plain write would give it, so a shared
    ``$GQR_CACHE_DIR`` stays readable by other users.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # A private directory lets the writer create a new file with normal
    # permissions without reading or changing the process-wide umask.
    with tempfile.TemporaryDirectory(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    ) as directory:
        tmp = Path(directory) / path.name
        write(tmp)
        os.replace(tmp, path)


def _read_cache(cache: Path, n_rows: int) -> pd.DataFrame | None:
    """The cached corpus, or None if there is none or it is unreadable."""
    if not cache.exists():
        return None
    try:
        df = pd.read_parquet(cache)
        if len(df) != n_rows or not {"text", "source", "split"} <= set(df.columns):
            raise ValueError(f"expected {n_rows} rows with text, source and split columns")
    except Exception as e:
        warnings.warn(f"Ignoring unreadable background cache {cache} ({e}); rebuilding.", stacklevel=4)
        return None
    return df


def _default_sizes(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> tuple[int, int]:
    n_domains = train_df["domain"].nunique()
    return round(len(train_df) / n_domains), round(len(eval_df) / n_domains)


def _build_or_load(
    id_test_df: pd.DataFrame, n_train: int, n_eval: int, check: bool, seed: int = SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = _cache_dir() / f"background_{BUILD_ID}_seed{seed}_train{n_train}_eval{n_eval}.parquet"
    missing: list[str] = []
    df = _read_cache(cache, n_train + n_eval)
    if df is None:
        n_total = n_train + n_eval
        limit = math.ceil(math.ceil(n_total / len(SOURCES)) * OVERSAMPLE)
        pools = {source: _candidates(source, limit, seed) for source in SOURCES}
        test_index = OverlapIndex(id_test_df["text"].astype(str))
        for name, frame in DataLoader.load_ood_test_dataset().items():
            if frame.empty:
                missing.append(name.strip())
            test_index.add(frame["text"].astype(str))
        texts, sources, stats = select_background(pools, test_index, n_total, seed)
        df = pd.DataFrame(
            {
                "text": texts,
                "domain": BACKGROUND_DOMAIN,
                "label": BACKGROUND_LABEL,
                "source": sources,
                "split": ["train"] * n_train + ["eval"] * n_eval,
            }
        )
        if missing:
            # Not cached, so the next call (for example after logging in to the
            # Hugging Face Hub) rebuilds with the full overlap check.
            warnings.warn(
                f"GQR-Bench v2 background was not screened for overlap with the OOD "
                f"test set(s) {missing}, which could not be loaded (DDSC/dkhate is "
                f"gated: log in to the Hugging Face Hub and accept its terms). The "
                f"corpus is used for this call but not cached.",
                stacklevel=3,
            )
        else:
            stats["sha256"] = fingerprint(texts)
            stats_json = json.dumps(stats, indent=2)
            _atomic_write(cache.with_suffix(".stats.json"), lambda path: path.write_text(stats_json))
            # the parquet file marks a finished build, so it is written last
            _atomic_write(cache, lambda path: df.to_parquet(path, index=False))
    digest = fingerprint(df["text"])
    if check and not missing and EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        warnings.warn(
            f"GQR-Bench v2 background fingerprint {digest[:12]} differs from the "
            f"reference {EXPECTED_SHA256[:12]}; an upstream corpus has changed.",
            stacklevel=3,
        )
    train_bg = df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)
    eval_bg = df[df["split"] == "eval"].drop(columns="split").reset_index(drop=True)
    return train_bg, eval_bg


def load_background_dataset(
    n_train: int | None = None, n_eval: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (building and caching on first use) the v2 background class.

    By default each split gets as many background rows as one in-distribution
    domain has. Columns: text, domain ("ood"), label (3), source.
    """
    train_df, eval_df, id_test_df = DataLoader.load_train_dataset()
    default = _default_sizes(train_df, eval_df)
    sizes = (default[0] if n_train is None else n_train, default[1] if n_eval is None else n_eval)
    return _build_or_load(id_test_df, *sizes, check=sizes == default)


def load_train_dataset_v2() -> tuple[pd.DataFrame, pd.DataFrame]:
    """GQR-Bench v2 train and eval splits: v1 splits plus the background class."""
    train_df, eval_df, id_test_df = DataLoader.load_train_dataset()
    train_bg, eval_bg = _build_or_load(id_test_df, *_default_sizes(train_df, eval_df), check=True)
    out = []
    for split, background in ((train_df, train_bg), (eval_df, eval_bg)):
        combined = pd.concat([split.assign(source="gqr"), background], ignore_index=True)
        out.append(combined.sample(frac=1, random_state=SEED).reset_index(drop=True))
    return out[0], out[1]
