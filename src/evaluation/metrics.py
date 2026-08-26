from __future__ import annotations

from typing import Any, Optional

import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, roc_auc_score


def classification_metrics(target: np.ndarray, logits: np.ndarray, class_names: list[str] | None = None) -> dict[str, Any]:
    prediction = logits.argmax(axis=1)
    probability = np.exp(logits - logits.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    result = {
        "accuracy": float(accuracy_score(target, prediction)),
        "macro_f1": float(f1_score(target, prediction, average="macro", zero_division=0)),
    }
    try:
        if logits.shape[1] == 2:
            result["auc"] = float(roc_auc_score(target, probability[:, 1]))
        else:
            result["macro_auc"] = float(roc_auc_score(target, probability, multi_class="ovr", average="macro"))
    except ValueError:
        result["auc" if logits.shape[1] == 2 else "macro_auc"] = float("nan")
    names = class_names or [str(index) for index in range(logits.shape[1])]
    per_class = {}
    class_accuracies, class_aucs = [], []
    for index, name in enumerate(names):
        positive = target == index
        class_accuracy = float((prediction[positive] == index).mean()) if positive.any() else float("nan")
        try:
            class_auc = float(roc_auc_score(positive.astype(int), probability[:, index]))
        except ValueError:
            class_auc = float("nan")
        per_class[name] = {"accuracy": class_accuracy, "auc": class_auc, "n": int(positive.sum())}
        class_accuracies.append(class_accuracy)
        class_aucs.append(class_auc)
    result["per_class"] = per_class
    result["macro_accuracy"] = float(np.nanmean(class_accuracies))
    result["macro_auc"] = float(np.nanmean(class_aucs))
    if logits.shape[1] == 2:
        positive = target == 1
        negative = ~positive
        result["sensitivity"] = float(((prediction == 1) & positive).sum() / max(positive.sum(), 1))
        result["specificity"] = float(((prediction == 0) & negative).sum() / max(negative.sum(), 1))
    return result


def regression_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    names: list[str],
    train_sd: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    normalized_mae, normalized_rmse = [], []
    if train_sd is None:
        train_sd = np.nanstd(target, axis=0, ddof=1)
    for index, name in enumerate(names):
        mask = np.isfinite(target[:, index]) & np.isfinite(prediction[:, index])
        y, yhat = target[mask, index], prediction[mask, index]
        mae = float(mean_absolute_error(y, yhat))
        rmse = float(mean_squared_error(y, yhat) ** 0.5)
        corr = float(pearsonr(y, yhat).statistic) if len(y) > 1 and np.std(y) > 0 and np.std(yhat) > 0 else float("nan")
        result[name] = {"mae": mae, "rmse": rmse, "pearson": corr, "n": int(len(y))}
        scale = float(train_sd[index])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"Training-set SD for {name} must be positive, got {scale}")
        normalized_mae.append(mae / scale)
        normalized_rmse.append(rmse / scale)
    result["mean_mae"] = float(np.mean([result[name]["mae"] for name in names]))
    result["mean_rmse"] = float(np.mean([result[name]["rmse"] for name in names]))
    result["normalized_mae"] = float(np.mean(normalized_mae))
    result["normalized_rmse"] = float(np.mean(normalized_rmse))
    result["training_target_sd"] = {name: float(value) for name, value in zip(names, train_sd)}
    return result
