"""Evaluate correlation between feasibility and veracity prediction accuracy."""

import argparse
import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mutual_info_score,
    precision_score,
    recall_score,
)

keys = ["Correct", "Feasibility with Search"]


def quantify_predictive_power(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, Any]:
    """
    Compute a suite of binary-classification metrics between two 0/1 series.

    :param y_true: ground-truth booleans (or 0/1 ints)
    :param y_pred: predicted booleans (or 0/1 ints)
    :returns: dict with
        - accuracy, precision, recall, f1, mutual_info, phi_coefficient,
        - tn, fp, fn, tp counts
    """
    y_true_i = y_true.astype(int)
    y_pred_i = y_pred.astype(int)

    # basic scores
    acc = accuracy_score(y_true_i, y_pred_i)
    prec = precision_score(y_true_i, y_pred_i, zero_division=0)
    rec = recall_score(y_true_i, y_pred_i, zero_division=0)
    f1 = f1_score(y_true_i, y_pred_i, zero_division=0)
    mi = mutual_info_score(y_true_i, y_pred_i)

    # force 2×2 CM even if one class is missing
    cm = confusion_matrix(y_true_i, y_pred_i, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # φ-coefficient (Pearson ρ for binary)
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    phi = (tp * tn - fp * fn) / np.sqrt(denom) if denom > 0 else 0.0

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mutual_info": mi,
        "phi_coefficient": phi,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": len(y_true),
        f"mean of {keys[0]}": y_true_i.mean(),
        f"mean of {keys[1]}": y_pred_i.mean(),
    }


def summarize_by_source(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group df by 'data_source', compute predictive metrics of feature_2 → feature_1
    for each source, and add an 'ALL' row for the aggregate.

    :param df: DataFrame with columns ['data_source','feature_1','feature_2']
    :returns: DataFrame indexed by data_source, columns are the metrics.
    """
    records: list[Dict[str, Any]] = []

    # per‐source
    for source, group in df.groupby("data_source"):
        metrics = quantify_predictive_power(group[keys[0]], group[keys[1]])
        metrics["data_source"] = source
        records.append(metrics)

    # aggregate
    total = quantify_predictive_power(df[keys[0]], df[keys[1]])
    print(json.dumps(total, indent=2, default=str))
    total["data_source"] = "ALL"
    records.append(total)

    result = pd.DataFrame.from_records(records).set_index("data_source")
    return result


parser = argparse.ArgumentParser()
parser.add_argument("jsonl_path")

if __name__ == "__main__":
    args = parser.parse_args()
    rows = []
    with open(args.jsonl_path) as jsonl_file:
        for _line in jsonl_file:
            _row = json.loads(_line)

            if not all(_row.get(key) for key in keys):
                continue

            _labels = {k: _row[k][0] == "True" for k in keys}
            _attribution = {"data_source": _row["metadata"]["data_source"]["dataset"]}
            rows.append({**_labels, **_attribution})

    df = pd.DataFrame(rows)

    # 2. Quantify predictive power
    summary = summarize_by_source(df)

    # 3. Display
    print(summary.to_latex())
    print(summary.to_markdown())
    print(summary.to_csv())
