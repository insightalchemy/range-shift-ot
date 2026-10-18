import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from rangeshift.cost import build_cost_matrix, pairwise_abs_difference, pairwise_euclidean
from rangeshift.forward_ot import unbalanced_sinkhorn
from rangeshift.inverse_ot import SpeciesData, learn_beta, stack_features

SIDE = 10
BETA_TRUE = torch.tensor([1.0, 2.0, 0.5])
EPS, TAU = 0.05, 1.0


def smooth(rng: np.random.Generator, sigma: float) -> np.ndarray:
    return gaussian_filter(rng.standard_normal((SIDE, SIDE)), sigma=sigma).astype(np.float32).ravel()


def smooth_distribution(rng: np.random.Generator) -> torch.Tensor:
    field = np.exp(2.0 * gaussian_filter(rng.standard_normal((SIDE, SIDE)), sigma=1.5)).ravel()
    return torch.as_tensor(field / field.sum(), dtype=torch.float32)


def test_recovers_known_cost_weights():
    rng = np.random.default_rng(0)
    xs, ys = np.meshgrid(np.linspace(0, 1, SIDE), np.linspace(0, 1, SIDE), indexing="ij")
    coords = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)
    features = {
        "d_geo": pairwise_euclidean(coords),
        "d_climate": pairwise_abs_difference(smooth(rng, 2.0)),
        "d_elev": pairwise_abs_difference(smooth(rng, 1.5)),
    }
    stack, _ = stack_features(features)
    C_true = build_cost_matrix(BETA_TRUE, stack)

    species = []
    for _ in range(6):
        mu_t0, mu_t1_sdm = smooth_distribution(rng), smooth_distribution(rng)
        with torch.no_grad():
            coupling = unbalanced_sinkhorn(mu_t0, mu_t1_sdm, C_true, eps=EPS, tau=TAU, n_iters=300, tol=1e-7)
        mu_obs = coupling.sum(dim=0)
        species.append(SpeciesData(mu_t0, mu_t1_sdm, mu_obs / mu_obs.sum()))

    history: list[dict] = []
    beta = learn_beta(
        species, features, n_beta=3, eps=EPS, tau=TAU, lr=0.05, n_epochs=60,
        inner_iters=40, outer_iters=30, history=history,
    )

    assert history[-1]["loss"] < history[0]["loss"]
    rel_err = (beta - BETA_TRUE).abs() / BETA_TRUE
    assert rel_err[1:].max() < 0.2, f"recovered {beta.tolist()} vs true {BETA_TRUE.tolist()}"
