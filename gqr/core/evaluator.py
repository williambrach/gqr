from typing import Union
from collections.abc import Callable

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Evaluator:
    """Evaluation utilities for the GQRBench package."""

    @staticmethod
    def evaluate(
        predicted_labels: Union[list, pd.DataFrame],
        true_labels: Union[list, pd.DataFrame],
    ) -> dict[str, float]:
        """
        Evaluate predictions against ground truth.

        Args:
            predictions: Model predictions in the form of a DataFrame or list
            ground_truth: Ground truth data in the form of a DataFrame or list
        Returns:
            dict[str, float]: dictionary mapping metric names to scores
        """
        if len(true_labels) != len(predicted_labels):
            raise ValueError(
                f"Length of true labels ({len(true_labels)}) does not match length of predicted labels ({len(predicted_labels)})"
            )
        accuracy = accuracy_score(true_labels, predicted_labels)

        precision = precision_score(
            true_labels, predicted_labels, average="weighted", zero_division=0
        )
        recall = recall_score(
                true_labels, predicted_labels, average="weighted", zero_division=0
        )
        f1 = f1_score(
            true_labels, predicted_labels, average="weighted", zero_division=0
        )

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    @staticmethod
    def evaluate_ood(
        df : pd.DataFrame, pred_col : str = "pred", true_col : str = "label", dataset_col : str = "dataset"
    ) -> dict[str, float]:
        """
        Evaluate predictions on the out-of-distribution test set.
        Automatically loads the OOD test set for evaluation.

        Args:
            predictions: Model predictions in the form of a DataFrame or list
            metrics: list of metrics to compute. If None, computes all available metrics.

        Returns:
            dict[str, float]: dictionary mapping metric names to scores
        """

        results = []
        for dataset in df[dataset_col].unique():
            x = df[df[dataset_col] == dataset]
            true_labels = x[true_col].tolist()
            predicted_labels = x[pred_col].tolist()
            scores = Evaluator.evaluate(
                predicted_labels=predicted_labels,
                true_labels=true_labels,
            )
            scores['dataset'] = dataset
            results.append(scores)
        results = pd.DataFrame(results)
        return results



def evaluate(
    predictions: Union[list, pd.DataFrame],
    ground_truth: Union[list, pd.DataFrame],
) -> dict[str, float]:
    """Convenience function for Evaluator.evaluate()"""
    return Evaluator.evaluate(predictions, ground_truth)


def evaluate_by_dataset(
    data : pd.DataFrame, pred_col : str = "pred", true_col : str = "label", dataset_col : str = "dataset"
) -> dict[str, float]:
    """Convenience function for Evaluator.evaluate_ood()"""
    return Evaluator.evaluate_ood(df=data, pred_col=pred_col, true_col=true_col, dataset_col=dataset_col)



def score(model_fn: Callable[[str], int]) -> dict:
    """ 
    model_fn should be a callable that takes a string and returns a class label in {0, 1, 2, 3},
    where 3 is the out-of-distribution class and 0, 1, 2 correspond to the three
    target domains: law, finance, and health, respectively.
    """
    print("[GQR-Score] Loading ID test dataset...")
    id_test_data = gqr.load_id_test_dataset()
    print("[GQR-Score] Running model on ID test dataset...")
    id_test_data["predictions"] = [model_fn(doc) for doc in id_test_data["text"].values]
    id_scores = gqr.evaluate(id_test_data["predictions"], ground_truth=id_test_data["label"])
    print("[GQR-Score] ID scores: ", id_scores)

    print("[GQR-Score] Loading ID test dataset...")
    ood_test_data = gqr.load_ood_test_dataset()
    print("[GQR-Score] Running model on OOD test dataset...")
    ood_test_data["predictions"] = [model_fn(doc) for doc in ood_test_data["text"].values]
    ood_scores_df = gqr.evaluate_by_dataset(ood_test_data, pred_col="predictions", true_col="label", dataset_col="dataset")
    print("[GQR-Score] OOD scores:", ood_scores_df, sep='\n')

    id_accuracy = id_scores["accuracy"]
    mean_ood_accuracy = ood_scores_df['accuracy'].mean()

    gqr_score =  2 * (id_accuracy * mean_ood_accuracy) / (id_accuracy + mean_ood_accuracy)

    scores = {
        "id_accuracy": id_accuracy,
        "ood_accuracy": mean_ood_accuracy,
        "gqr_score": gqr_score,
    }
    print("[GQR-Score] Final scores: ", scores)

    return scores
