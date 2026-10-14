"""Classification metrics and experiment logging."""

import csv
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)

from src import config


def classification_metrics(y_true, y_pred, proba=None, classes=config.CLASSES) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    out = {
        "n": int(len(y_true)),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0),
    }
    for c, pi, ri, fi in zip(classes, p, r, f):
        name = config.CLASS_NAMES[c]
        out[f"precision_{name}"], out[f"recall_{name}"], out[f"f1_{name}"] = pi, ri, fi
    if proba is not None:
        out["log_loss"] = log_loss(y_true, proba, labels=classes)
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
            for k, v in out.items()}


def per_year_macro_f1(y_true: pd.Series, y_pred) -> pd.Series:
    df = pd.DataFrame({"y": y_true.values, "p": np.asarray(y_pred)}, index=y_true.index)
    return df.groupby(df.index.year).apply(
        lambda g: f1_score(g["y"], g["p"], labels=config.CLASSES, average="macro", zero_division=0))


def plot_confusion(y_true, y_pred, title: str, ax=None):
    labels = [config.CLASS_NAMES[c] for c in config.CLASSES]
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=config.CLASSES, display_labels=labels, ax=ax,
        colorbar=False, cmap="Blues")
    disp.ax_.set_title(title)
    return disp


LOG_FIELDS = ["exp_id", "timestamp", "split", "label_scheme", "feature_set", "imbalance",
              "model", "params", "cv_macro_f1_mean", "cv_macro_f1_std", "val_macro_f1",
              "val_accuracy", "val_bt_sharpe", "conclusion"]


def log_experiment(row: dict, path=None) -> None:
    """Append one row to outputs/experiment_log.csv (test results are NOT logged here)."""
    path = path or config.OUTPUT_DIR / "experiment_log.csv"
    new = not path.exists()
    row = {**{k: "" for k in LOG_FIELDS}, **row,
           "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")}
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row[k] for k in LOG_FIELDS})
