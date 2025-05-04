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
```

## Domain Labels

The repository provides mappings between numerical labels and domain names:

```python
# Get label mappings
print(gqr.label2domain)  # Maps numerical labels to domain names
print(gqr.domain2label)  # Maps domain names to numerical labels
```

## Evaluation

**Important**: When using the `evaluate` functions, ensure that the prediction and ground truth values are strings, not numerical labels.
The module offers comprehensive evaluation functions:

```python
# Evaluate on in-domain test set

results = gqr.evaluate(
    predictions=pred_id_labels,
    ground_truth=true_id_labels
)

# Evaluate on out-of-domain test set
ood_results = gqr.evaluate(
    predictions=pred_ood_labels,
    ground_truth=true_ood_labels
)

# Evaluate by dataset (grouped evaluation)
dataset_results = gqr.evaluate_by_dataset(
    ood_test_data,
    pred_col='pred',
    true_col='true',
    dataset_col='dataset'
)
```


## Paper and Citations

If you use GQR-Bench in your research, please cite our paper:

```
```


## Contributing

Contributions to GQR-Bench are welcome! Please feel free to submit a Pull Request with improvements, additional evaluation metrics, or dataset enhancements.
