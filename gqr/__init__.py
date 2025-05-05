from collections.abc import Callable

from .core.dataloader import (
    DataLoader,
    domain2label,
    label2domain,
    load_dev_dataset,
    load_id_test_dataset,
    load_ood_test_dataset,
    load_train_dataset,
)
from .core.evaluator import evaluate, evaluate_by_dataset


def score(model_fn: Callable[[str], int]) -> dict:
    """
    model_fn should be a callable that takes a string and returns a class label in {0, 1, 2, 3},
    where 3 is the out-of-distribution class and 0, 1, 2 correspond to the three
    target domains: law, finance, and health, respectively.
    """
    print("[GQR-Score] Loading ID test dataset...")
    id_test_data = load_id_test_dataset()
    print("[GQR-Score] Running model on ID test dataset...")
    id_test_data["predictions"] = [model_fn(doc) for doc in id_test_data["text"].values]
    id_scores = evaluate(
        id_test_data["predictions"], ground_truth=id_test_data["label"]
    )
    print("[GQR-Score] ID scores: ", id_scores)

    print("[GQR-Score] Loading ID test dataset...")
    ood_test_data = load_ood_test_dataset()
    print("[GQR-Score] Running model on OOD test dataset...")
    ood_test_data["predictions"] = [
        model_fn(doc) for doc in ood_test_data["text"].values
    ]
    ood_scores_df = evaluate_by_dataset(
        ood_test_data, pred_col="predictions", true_col="label", dataset_col="dataset"
    )
    print("[GQR-Score] OOD scores:", ood_scores_df, sep="\n")

    id_accuracy = id_scores["accuracy"]
    mean_ood_accuracy = ood_scores_df["accuracy"].mean()

    gqr_score = (
        2 * (id_accuracy * mean_ood_accuracy) / (id_accuracy + mean_ood_accuracy)
    )

    scores = {
        "id_accuracy": id_accuracy,
        "ood_accuracy": mean_ood_accuracy,
        "gqr_score": gqr_score,
    }
    print("[GQR-Score] Final scores: ", scores)

    return scores
