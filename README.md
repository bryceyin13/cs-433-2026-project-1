# Project 1

## Environment Setup and Installation

```bash
conda create -n cs-433-p1 python=3.12 pip -y
conda activate cs-433-p1
python -m pip install -r requirement.txt
```

Place `x_train.csv`, `y_train.csv`, and `x_test.csv` in `dataset/`, then run:

```bash
python run.py
```

This trains logistic regression on an 80% split and saves test predictions to
`results/submission.csv` using the fixed OOF-selected threshold.

Select the threshold separately with `python -m scripts.threshold`.
Run five-fold cross-validation with `python -m scripts.cross_validation`.

Compile the report from its directory:

```bash
cd report
latexmk -pdf main.tex
```

## Project Structure

```text
.
├── run.py
├── scripts/
│   ├── cross_validation.py
│   └── threshold.py
├── utils/
│   ├── helpers.py
│   ├── implementations.py
│   └── preprocessing.py
├── dataset/                 # Input data (ignored by Git)
├── results/                 # Generated files (ignored by Git)
├── .gitignore
├── requirement.txt
└── README.md
```
