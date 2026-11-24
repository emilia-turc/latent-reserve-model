"""Goodness-of-fit metrics (manuscript, "Model evaluation and comparison")."""

from __future__ import annotations

import numpy as np
import pandas as pd


def r2(E_data, E_sim) -> float:
    """Coefficient of determination; can be negative."""
    E_data, E_sim = np.asarray(E_data), np.asarray(E_sim)
    ss_tot = np.sum((E_data - E_data.mean()) ** 2)
    return float(1 - np.sum((E_data - E_sim) ** 2) / ss_tot)


def rmse(E_data, E_sim) -> float:
    return float(np.sqrt(np.mean((np.asarray(E_data) - np.asarray(E_sim)) ** 2)))


def bias(E_data, E_sim) -> float:
    """Mean residual, simulation - data."""
    return float(np.mean(np.asarray(E_sim) - np.asarray(E_data)))


def bic(E_data, E_sim, k: int) -> float:
    """Per-sample BIC: ln(RSS/n) + k ln(n)/n, k counting the delay."""
    resid = np.asarray(E_data) - np.asarray(E_sim)
    n = len(resid)
    rss = max(np.sum(resid**2), np.finfo(float).eps)
    return float(np.log(rss / n) + k * np.log(n) / n)


def aic(E_data, E_sim, k: int) -> float:
    resid = np.asarray(E_data) - np.asarray(E_sim)
    n = len(resid)
    rss = max(np.sum(resid**2), np.finfo(float).eps)
    return float(np.log(rss / n) + 2 * k / n)


def patient_metrics(E_data, E_sim, k: int) -> dict:
    return {
        "r2": r2(E_data, E_sim),
        "rmse": rmse(E_data, E_sim),
        "bias": bias(E_data, E_sim),
        "aic": aic(E_data, E_sim, k),
        "bic": bic(E_data, E_sim, k),
        "n": len(E_data),
    }


def pooled_metrics(sims: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Pooled R2 and bias over patients: ``sims[patient] = (E_data, E_sim)``."""
    x = np.concatenate([d for d, _ in sims.values()])
    y = np.concatenate([s for _, s in sims.values()])
    return {
        "pooled_r2": r2(x, y),
        "pooled_bias": bias(x, y),
        "pooled_bias_percent": 100 * bias(x, y) / (x.max() - x.min()),
    }


def summarize(per_patient: pd.DataFrame) -> dict:
    """Median and interquartile range of the per-patient R2."""
    q1, med, q3 = per_patient["r2"].quantile([0.25, 0.5, 0.75])
    return {"r2_median": med, "r2_q1": q1, "r2_q3": q3, "n_patients": len(per_patient)}
