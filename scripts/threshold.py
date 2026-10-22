import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from utils.helpers import load_csv_data, split_indices
from utils.implementations import _sigmoid, logistic_regression
from utils.preprocessing import fit_preprocessor, transform_features


def select_threshold(y, probabilities):
    """Choose the highest threshold attaining maximum positive-class F1."""
    order = np.argsort(-probabilities)
    scores = probabilities[order]
    # A threshold includes all samples with the same score.
    ends = np.r_[np.flatnonzero(scores[:-1] != scores[1:]), len(scores) - 1]
    true_positives = np.cumsum(y[order] == 1)[ends]
    f1 = 2 * true_positives / (np.count_nonzero(y == 1) + ends + 1)
    best = np.argmax(f1)
    return float(scores[ends[best]]), float(f1[best])


def classification_metrics(y, predictions):
    true_positives = np.count_nonzero((y == 1) & (predictions == 1))
    f1 = 2 * true_positives / (
        np.count_nonzero(y == 1) + np.count_nonzero(predictions == 1)
    )
    return {"accuracy": float(np.mean(y == predictions)), "f1": float(f1)}


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "dataset"
    with (data_path / "x_train.csv").open(newline="") as file:
        column_names = next(csv.reader(file))[1:]
    print("Loading data...", flush=True)
    x_train, x_test, y, train_ids, test_ids = load_csv_data(data_path)
    train, validation = split_indices(y)
    print("Preprocessing...", flush=True)
    tx_train, state = fit_preprocessor(x_train[train], column_names)
    tx_validation = transform_features(x_train[validation], column_names, state)
    tx_test = transform_features(x_test, column_names, state)
    del x_train, x_test

    print(f"Training: {tx_train.shape}; validation: {tx_validation.shape}", flush=True)
    start = perf_counter()
    weights, loss = logistic_regression(
        (y[train] + 1) / 2, tx_train, np.zeros(tx_train.shape[1]), 5000, 0.19
    )
    probabilities = _sigmoid(tx_validation @ weights)
    threshold, _ = select_threshold(y[validation], probabilities)
    predicted = np.where(probabilities >= threshold, 1, -1)
    metrics = classification_metrics(y[validation], predicted)
    print(f"Threshold: {threshold!r}", flush=True)
    print(f"Accuracy: {metrics['accuracy']:.10f}; F1: {metrics['f1']:.10f}", flush=True)
    print(f"Training seconds: {perf_counter() - start:.2f}; loss: {loss:.10f}")
    for actual in (-1, 1):
        print(
            [
                int(np.count_nonzero((y[validation] == actual) & (predicted == label)))
                for label in (-1, 1)
            ]
        )

    output = root / "results" / "threshold_80_20.npz"
    output.parent.mkdir(exist_ok=True)
    np.savez_compressed(
        output,
        threshold=threshold,
        weights=weights,
        validation_probabilities=probabilities,
        y_validation=y[validation],
        train_ids=train_ids[train],
        validation_ids=train_ids[validation],
        test_probabilities=_sigmoid(tx_test @ weights),
        test_ids=test_ids,
        feature_names=state["feature_names"],
        train_loss=loss,
        **metrics,
    )
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
