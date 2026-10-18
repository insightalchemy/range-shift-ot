from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import cKDTree

from rangeshift.cost import pairwise_euclidean
from rangeshift.forward_ot import unbalanced_sinkhorn


def persistence(mu_t0: np.ndarray) -> np.ndarray:
    return np.asarray(mu_t0, dtype=np.float32).copy()


def sdm_only(mu_t1_sdm: np.ndarray) -> np.ndarray:
    return np.asarray(mu_t1_sdm, dtype=np.float32).copy()


def climate_velocity(
    mu_t0: np.ndarray, climate_t0: np.ndarray, climate_t1: np.ndarray,
) -> np.ndarray:
    """Move each cell's mass to the cell whose period-2 climate is nearest its period-1 climate."""
    tree = cKDTree(np.asarray(climate_t1, dtype=np.float64))
    _, nearest = tree.query(np.asarray(climate_t0, dtype=np.float64), k=1)
    mu_pred = np.zeros(np.asarray(mu_t0).shape[0], dtype=np.float64)
    np.add.at(mu_pred, nearest, np.asarray(mu_t0, dtype=np.float64))
    total = mu_pred.sum()
    if total <= 0:
        raise ValueError("climate_velocity: predicted mass sums to zero")
    return (mu_pred / total).astype(np.float32)


def uniform_cost_ot(
    mu_t0: np.ndarray,
    mu_t1_sdm: np.ndarray,
    coords: np.ndarray,
    tau: float = 1.0,
    eps: float | None = None,
    cell_size_m: float = 50_000.0,
    n_iters: int = 200,
    tol: float = 1e-4,
) -> np.ndarray:
    """Unbalanced OT with a geographic-only cost (beta = [1, 0, 0]); eps defaults to 0.05 * mean cost."""
    coords_cells = np.asarray(coords, dtype=np.float32) / cell_size_m
    C = pairwise_euclidean(coords_cells)
    if eps is None:
        eps = float(0.05 * C.mean().item())

    mu_t0_t = torch.as_tensor(np.asarray(mu_t0, dtype=np.float32))
    mu_t1_sdm_t = torch.as_tensor(np.asarray(mu_t1_sdm, dtype=np.float32))
    with torch.no_grad():
        coupling = unbalanced_sinkhorn(mu_t0_t, mu_t1_sdm_t, C, eps=eps, tau=tau, n_iters=n_iters, tol=tol)
        mu_pred = coupling.sum(dim=0).numpy().astype(np.float64)
    total = mu_pred.sum()
    if total <= 0:
        raise ValueError("uniform_cost_ot: predicted mass sums to zero")
    return (mu_pred / total).astype(np.float32)
