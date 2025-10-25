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
