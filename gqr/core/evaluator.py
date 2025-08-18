from typing import Union

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
        df: pd.DataFrame,
        pred_col: str = "pred",
        true_col: str = "label",
        dataset_col: str = "dataset",
    ) -> pd.DataFrame:
        """
        Evaluate predictions on the out-of-distribution test set.
        Automatically loads the OOD test set for evaluation.

        Args:
            predictions: Model predictions in the form of a DataFrame or list
            metrics: list of metrics to compute. If None, computes all available metrics.

        Returns:
            pd.DataFrame: DataFrame mapping metric names to scores
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
            scores["dataset"] = dataset
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
    data: pd.DataFrame,
    pred_col: str = "pred",
    true_col: str = "label",
    dataset_col: str = "dataset",
) -> pd.DataFrame:
    """Convenience function for Evaluator.evaluate_ood()"""
    return Evaluator.evaluate_ood(
        df=data, pred_col=pred_col, true_col=true_col, dataset_col=dataset_col
    )
