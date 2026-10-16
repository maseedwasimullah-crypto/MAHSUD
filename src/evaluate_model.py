"""Evaluation metrics and model-vs-baseline comparison helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, and a directional sanity ratio."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mean_true = float(np.mean(y_true)) if len(y_true) else float("nan")
    baseline_like = float(np.mean(np.abs(np.diff(y_true)))) if len(y_true) > 1 else float("nan")
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mean_true": round(mean_true, 4),
        "mae_over_mean_true_pct": round(mae / mean_true * 100, 2) if mean_true else None,
        "mae_over_naive_diff_baseline": round(mae / baseline_like, 4)
        if baseline_like and baseline_like > 0
        else None,
    }


def compare_to_baselines(
    model_metrics: dict, naive_metrics: dict, seasonal_metrics: dict
) -> dict:
    """Summarize whether the model beats simple baselines on the test window."""
    return {
        "model_mae": model_metrics["mae"],
        "naive_mae": naive_metrics["mae"],
        "seasonal_naive_mae": seasonal_metrics["mae"],
        "beats_naive": model_metrics["mae"] < naive_metrics["mae"],
        "beats_seasonal_naive": model_metrics["mae"] < seasonal_metrics["mae"],
        "mae_improvement_vs_naive_pct": round(
            (naive_metrics["mae"] - model_metrics["mae"]) / max(naive_metrics["mae"], 1e-9) * 100,
            2,
        ),
    }
