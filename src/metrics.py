import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, hamming_loss,
)

def compute_metrics(Y_true, Y_pred, Y_proba, label_names):
    metrics = {
        "accuracy":        accuracy_score(Y_true, Y_pred),
        "hamming_loss":    hamming_loss(Y_true, Y_pred),
        "f1_macro":        f1_score(Y_true, Y_pred, average="macro",    zero_division=0),
        "f1_micro":        f1_score(Y_true, Y_pred, average="micro",    zero_division=0),
        "f1_weighted":     f1_score(Y_true, Y_pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(Y_true, Y_pred, average="macro", zero_division=0),
        "recall_macro":    recall_score(Y_true, Y_pred, average="macro",    zero_division=0),
    }

    valid_cols = [j for j in range(Y_true.shape[1]) if Y_true[:, j].sum() > 0]
    if valid_cols:
        metrics["roc_auc_macro"] = roc_auc_score(
            Y_true[:, valid_cols], Y_proba[:, valid_cols], average="macro"
        )
    else:
        metrics["roc_auc_macro"] = float("nan")

    f1_per = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    metrics["f1_per_label"] = dict(zip(label_names, f1_per))
    return metrics