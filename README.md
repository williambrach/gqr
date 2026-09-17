# GQR-Bench (Guarded Query Routing Benchmark)

A benchmark and evaluation toolkit for developing and testing guarded query routing models for AI systems.

## Installation

```bash
pip install gqr
```

## Quick Start

```python
import gqr

# Load development dataset for initial experimentation
dev_train_data, dev_eval_data = gqr.load_dev_dataset()

# Load training dataset for model development
train_data, eval_data = gqr.load_train_dataset()

# Load test datasets for final evaluation
domain_test_data = gqr.load_id_test_dataset()  # In-domain test data
ood_test_data = gqr.load_ood_test_dataset()    # Out-of-domain test data

# Score the model on gqr-bench
def scoring_function(text: str) -> int:
    # Scoring function takes text input (str) and returns predicted domain label (int)
    # Implement your classification logic here
    return 0  # Replace with actual domain prediction

# Evaluate model performance
score = gqr.score(scoring_function)
```

## GQR-Bench v2: background class for training

GQR-Bench v1 provides training data for the three in-distribution domains only.
v2 adds a fourth **background** class (label `3`, domain `"ood"`) to the train and
eval splits, so a router can learn rejection directly instead of only through a
confidence threshold. **Test sets are identical in v1 and v2**, so scores stay comparable.

```python
import gqr

train_data, eval_data = gqr.load_train_dataset(version="v2")  # adds a `source` column
background_train, background_eval = gqr.load_background_dataset()  # background rows only
```

The background class is sized like one in-distribution domain and drawn equally from
three general-purpose corpora at pinned revisions, none of which is used by any test
set: wikitext-103 prose (detokenized), dolly-15k instructions, and Yahoo Answers
questions from topics outside law, finance, and healthcare. Candidates are deduplicated on normalized
text (case, punctuation, and whitespace ignored) within and across sources before the
train/eval split, so no question appears in both splits. Each candidate must also pass
two filters:

1. no law, finance, or healthcare keyword;
2. no exact or 8-word-shingle overlap with the ID or OOD test sets.

Selection is content-addressed: each source keeps the eligible passages with the
smallest seeded hash. The build is therefore identical across machines and library
versions, and a fingerprint check warns if an upstream corpus ever changes.

The first call builds the corpus (one streamed pass over each source) and caches it
under `$GQR_CACHE_DIR` (default `~/.cache/gqr`), with per-source filter statistics.

## Data sources

The original finance source `4DR1455/finance_questions` was removed from the Hugging
Face Hub. It was a re-upload of `DeividasM/financial-instruction-aq22`, and the
instructions GQR-Bench uses are identical, so the loader now reads the upstream
dataset and the benchmark splits are unchanged. `Stanford/web_questions` is loaded
under its current name, `stanfordnlp/web_questions`.

## Domain Labels

The repository provides mappings between numerical labels and domain names:

```python
# Get label mappings
print(gqr.label2domain)  # Maps numerical labels to domain names
print(gqr.domain2label)  # Maps domain names to numerical labels
```

## Score

```python
import gqr

def scoring_function(text: str) -> int:
    # Scoring function takes text input (str) and returns predicted domain label (int)
    # Implement your classification logic here
    return 0  # Replace with actual domain prediction

# Evaluate model performance
score = gqr.score(scoring_function)
```

## Contributing

```
git clone git@github.com:williambrach/gqr.git
```

```
uv venv --python 3.12
```

```
uv sync 
```

```
uv run --with pytest pytest tests
```

## Paper and Citations

If you use GQR-Bench in your research, please cite our paper:

```
@incollection{gqrbench2025,
      title={Guarded Query Routing for Large Language Models}, 
      author={Richard Šléher and William Brach and Tibor Sloboda and Kristián Košťál and Lukas Galke},
      booktitle={ECAI 2025},
      year={2025},
      pages={4129-4136},
      publisher={IOS Press}
}
```
