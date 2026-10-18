from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def build_design_matrix(bio_values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize the bioclim columns with fixed (early-period) statistics and append squares."""
    z = (np.asarray(bio_values, dtype=np.float64) - mean) / std
    return np.concatenate([z, z**2], axis=1)


def fit_sdm(design_matrix: np.ndarray, presence: np.ndarray, kind: str = "logistic"):
    if kind == "logistic":
        model = LogisticRegression(max_iter=1000)
    elif kind == "rf":
        model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=0, n_jobs=-1)
    else:
        raise ValueError(f"unknown SDM kind: {kind}")
    model.fit(design_matrix, presence)
    return model


def project_sdm(model, design_matrix: np.ndarray) -> np.ndarray:
    """Predicted suitability on a set of cells, normalized to sum to 1."""
    suitability = model.predict_proba(design_matrix)[:, 1]
    total = float(suitability.sum())
    if total <= 0:
        raise ValueError("predicted suitability sums to zero")
    return (suitability / total).astype(np.float32)


def holdout_auc(design_matrix: np.ndarray, presence: np.ndarray, kind: str = "logistic") -> float:
    """AUC on a stratified 20% hold-out of the training routes."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        design_matrix, presence, test_size=0.2, stratify=presence, random_state=0,
    )
    model = fit_sdm(X_tr, y_tr, kind)
    return float(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]))
