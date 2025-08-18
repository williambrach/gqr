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


def score_batch(
    model_fn: Callable[[list[str]], list[int]], batch_size: int = 32
) -> dict:
    """
    Scores a model based on in-distribution (ID) and out-of-distribution (OOD) performance.
    This function processes data in batches for improved efficiency.

    Args:
        model_fn: A callable that takes a list of strings (a batch of documents) and
                  returns a list of corresponding class labels in {0, 1, 2, 3}.
                  Class 3 is for OOD, while 0, 1, 2 correspond to the law,
                  finance, and health domains, respectively.
        batch_size: The number of documents to process in a single batch.

    Returns:
        A dictionary containing the ID accuracy, mean OOD accuracy, and the final GQR score.
    """
    # --- In-Distribution (ID) Scoring ---
    print("[GQR-Score] Loading ID test dataset...")
    id_test_data = load_id_test_dataset()
    id_texts = id_test_data["text"].values
    id_predictions = []

    print(f"[GQR-Score] Running model on ID test dataset in batches of {batch_size}...")
    # Process texts in batches
    for i in range(0, len(id_texts), batch_size):
        batch_texts = id_texts[i : i + batch_size]
        # The model function is now called with a list of texts
        batch_preds = model_fn(list(batch_texts))
        id_predictions.extend(batch_preds)

    id_test_data["predictions"] = id_predictions
    id_scores = evaluate(
        id_test_data["predictions"], ground_truth=id_test_data["label"]
    )
    print(f"[GQR-Score] ID scores: {id_scores}")

    # --- Out-of-Distribution (OOD) Scoring ---
    print("[GQR-Score] Loading OOD test dataset...")
    ood_test_data = load_ood_test_dataset()
    ood_texts = ood_test_data["text"].values
    ood_predictions = []

    print(
        f"[GQR-Score] Running model on OOD test dataset in batches of {batch_size}..."
    )
    # Process OOD texts in batches
    for i in range(0, len(ood_texts), batch_size):
        batch_texts = ood_texts[i : i + batch_size]
        batch_preds = model_fn(list(batch_texts))
        ood_predictions.extend(batch_preds)

    ood_test_data["predictions"] = ood_predictions
    ood_scores_df = evaluate_by_dataset(
        ood_test_data, pred_col="predictions", true_col="label", dataset_col="dataset"
    )
    print("[GQR-Score] OOD scores:", ood_scores_df, sep="\n")

    # --- Final Score Calculation ---
    id_accuracy = id_scores["accuracy"]
    mean_ood_accuracy = ood_scores_df["accuracy"].mean()

    # Calculate the harmonic mean (GQR score), handling division by zero
    if id_accuracy + mean_ood_accuracy == 0:
        gqr_score = 0.0
    else:
        gqr_score = (
            2 * (id_accuracy * mean_ood_accuracy) / (id_accuracy + mean_ood_accuracy)
        )

    scores = {
        "id_accuracy": id_accuracy,
        "ood_accuracy": mean_ood_accuracy,
        "gqr_score": gqr_score,
    }
    print(f"[GQR-Score] Final scores: {scores}")

    return scores
