import csv
from pathlib import Path

import numpy as np

from utils.helpers import create_csv_submission, load_csv_data, split_indices
from utils.implementations import _sigmoid, logistic_regression
from utils.preprocessing import fit_preprocessor, transform_features

THRESHOLD = 0.1944888408230755


def main():
    root = Path(__file__).resolve().parent
    data_path = root / "dataset"
    with (data_path / "x_train.csv").open(newline="") as file:
        column_names = next(csv.reader(file))[1:]
    print("Loading data...", flush=True)
    x_train, x_test, y, _, test_ids = load_csv_data(data_path)
    train, _ = split_indices(y)

    print("Preprocessing...", flush=True)
    tx_train, state = fit_preprocessor(x_train[train], column_names)
    tx_test = transform_features(x_test, column_names, state)
    del x_train, x_test

    print(f"Training logistic regression: {tx_train.shape}...", flush=True)
    weights, _ = logistic_regression(
        (y[train] + 1) / 2, tx_train, np.zeros(tx_train.shape[1]), 5000, 0.19
    )
    predictions = np.where(_sigmoid(tx_test @ weights) >= THRESHOLD, 1, -1)

    output = root / "results" / "submission.csv"
    output.parent.mkdir(exist_ok=True)
    create_csv_submission(test_ids, predictions, output)
    print(f"Saved {len(predictions)} predictions to {output}", flush=True)


if __name__ == "__main__":
    main()
