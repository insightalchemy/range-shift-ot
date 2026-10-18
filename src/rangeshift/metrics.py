from __future__ import annotations

import numpy as np
import ot


def _normalized(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=np.float64)
    return mu / mu.sum()


def wasserstein_distance(mu_pred: np.ndarray, mu_obs: np.ndarray, coords: np.ndarray) -> float:
    """Exact Wasserstein-1 with Euclidean ground cost on coords (same units as coords)."""
    a, b = _normalized(mu_pred), _normalized(mu_obs)
    M = ot.dist(np.asarray(coords, dtype=np.float64), metric="euclidean")
    return float(ot.emd2(a, b, M))


def schoeners_d(mu_pred: np.ndarray, mu_obs: np.ndarray) -> float:
    a, b = _normalized(mu_pred), _normalized(mu_obs)
    return float(1.0 - 0.5 * np.abs(a - b).sum())


def centroid_shift_error(mu_pred: np.ndarray, mu_obs: np.ndarray, coords: np.ndarray) -> float:
    a, b = _normalized(mu_pred), _normalized(mu_obs)
    coords = np.asarray(coords, dtype=np.float64)
    return float(np.linalg.norm((a[:, None] * coords).sum(axis=0) - (b[:, None] * coords).sum(axis=0)))


def northern_half_mass(mu: np.ndarray, n_x: int, n_y: int, support_idx: np.ndarray) -> float:
    """Mass in grid rows iy >= n_y // 2."""
    full = np.zeros(n_x * n_y, dtype=np.float64)
    full[support_idx] = np.asarray(mu, dtype=np.float64)
    iy = np.arange(n_x * n_y) // n_x
    return float(full[iy >= n_y // 2].sum())


def northern_half_mass_error(
    mu_pred: np.ndarray, mu_obs: np.ndarray, n_x: int, n_y: int, support_idx: np.ndarray,
) -> float:
    a, b = _normalized(mu_pred), _normalized(mu_obs)
    return abs(northern_half_mass(a, n_x, n_y, support_idx) - northern_half_mass(b, n_x, n_y, support_idx))
