import csv
from pathlib import Path
from time import perf_counter

import numpy as np

from scripts.threshold import classification_metrics, select_threshold
from utils.helpers import load_csv_data
from utils.implementations import _sigmoid, logistic_regression
from utils.preprocessing import fit_preprocessor, transform_features


def stratified_folds(y):
    rng = np.random.default_rng(42)
    folds = [[] for _ in range(5)]
    for label in np.unique(y):
        indices = rng.permutation(np.flatnonzero(y == label))
        for fold, part in zip(folds, np.array_split(indices, 5)):
            fold.append(part)
    return [rng.permutation(np.concatenate(fold)) for fold in folds]


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "dataset"
    with (data_path / "x_train.csv").open(newline="") as file:
        column_names = next(csv.reader(file))[1:]
    print("Loading data...", flush=True)
    x, x_test, y, ids, _ = load_csv_data(data_path)
    del x_test
    folds = stratified_folds(y)
    probabilities = np.empty(len(y))
    fold_numbers = np.empty(len(y), dtype=int)
    results = []

    for index, validation in enumerate(folds):
        train = np.concatenate(folds[:index] + folds[index + 1:])
        print(f"Fold {index + 1}/5: preprocessing...", flush=True)
        tx_train, state = fit_preprocessor(x[train], column_names)
        tx_validation = transform_features(x[validation], column_names, state)
        print(
            f"  Training: {tx_train.shape}; validation: {tx_validation.shape}",
            flush=True,
        )
        start = perf_counter()
        weights, loss = logistic_regression(
            (y[train] + 1) / 2, tx_train, np.zeros(tx_train.shape[1]), 5000, 0.19
        )
        scores = _sigmoid(tx_validation @ weights)
        probabilities[validation] = scores
        fold_numbers[validation] = index + 1
        threshold, _ = select_threshold(y[validation], scores)
        metrics = classification_metrics(
            y[validation], np.where(scores >= threshold, 1, -1)
        )
        results.append(
            {
                "fold": index + 1,
                "threshold": threshold,
                **metrics,
                "training_rows": len(train),
                "validation_rows": len(validation),
                "feature_count": tx_train.shape[1],
                "training_seconds": perf_counter() - start,
                "train_loss": float(loss),
            }
        )
        print(
            f"  threshold={threshold:.10f}; accuracy={metrics['accuracy']:.6f}; "
            f"F1={metrics['f1']:.6f}",
            flush=True,
        )
        del tx_train, tx_validation

    threshold, _ = select_threshold(y, probabilities)
    metrics = classification_metrics(y, np.where(probabilities >= threshold, 1, -1))
    results.append(
        {"fold": "OOF", "threshold": threshold, **metrics, "validation_rows": len(y)}
    )
    output = root / "results" / "cross_validation.csv"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    np.savez_compressed(
        output.with_name("cross_validation_predictions.npz"),
        oof_probabilities=probabilities,
        y=y,
        ids=ids,
        fold=fold_numbers,
        threshold=threshold,
        **metrics,
    )
    print(
        f"OOF threshold: {threshold!r}; accuracy={metrics['accuracy']:.6f}; "
        f"F1={metrics['f1']:.6f}",
        flush=True,
    )
    print(f"Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
