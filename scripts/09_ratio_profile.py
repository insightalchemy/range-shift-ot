"""Profile the validation loss over fixed climate/elevation ratios r = beta1 / beta2, fitting only the overall scale."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.cost import build_cost_matrix
from rangeshift.forward_ot import unbalanced_sinkhorn
from rangeshift.inverse_ot import SpeciesData, stack_features
from rangeshift.metrics import wasserstein_distance
from rangeshift.pipeline import (
    CURATED_TRAIN,
    EPS,
    LAMBDA_CURATED,
    OUTER_TAU,
    TAU,
    add_path_args,
    load_features,
    load_species_arrays,
    load_species_set,
    load_support,
    to_species_data,
    write_json,
)

RATIOS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.76, 2.0, 2.5, 3.0, 4.0]
VALIDATION = ["06140", "05110"]
N_EPOCHS = 200
LR = 0.05


def fit_scale(species: list[SpeciesData], stack: torch.Tensor, r: float, lam: float, epochs: int) -> float:
    """Optimize m in beta = [1, m * r, m] under the debiased Sinkhorn divergence loss."""
    C_geo = stack[0]
    raw_m = torch.tensor(float(np.log(np.e - 1.0)), requires_grad=True)
    optimizer = torch.optim.Adam([raw_m], lr=LR)

    with torch.no_grad():
        cost_bb = [
            (unbalanced_sinkhorn(sd.mu_t1_obs, sd.mu_t1_obs, C_geo, eps=EPS, tau=OUTER_TAU, n_iters=100, tol=1e-4) * C_geo).sum().item()
            for sd in species
        ]

    for _ in range(epochs):
        optimizer.zero_grad()
        for sd, cbb in zip(species, cost_bb):
            m = torch.nn.functional.softplus(raw_m)
            C_pred = build_cost_matrix(torch.stack([torch.tensor(1.0), m * r, m]), stack)

            coupling = unbalanced_sinkhorn(sd.mu_t0, sd.mu_t1_sdm, C_pred, eps=EPS, tau=TAU, n_iters=100, tol=1e-4)
            mu_pred = coupling.sum(dim=0)
            mu_pred = mu_pred / (mu_pred.sum() + 1e-12)

            cost_ab = (unbalanced_sinkhorn(mu_pred, sd.mu_t1_obs, C_geo, eps=EPS, tau=OUTER_TAU, n_iters=100, tol=1e-4) * C_geo).sum()
            cost_aa = (unbalanced_sinkhorn(mu_pred, mu_pred, C_geo, eps=EPS, tau=OUTER_TAU, n_iters=100, tol=1e-4) * C_geo).sum()
            loss = cost_ab - 0.5 * (cost_aa + cbb) + lam * (m ** 2) / len(species)
            loss.backward()
        optimizer.step()
    with torch.no_grad():
        return float(torch.nn.functional.softplus(raw_m))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS)
    args = parser.parse_args()
    ratios = args.ratios

    species = load_species_set(args.results, "curated")
    support = load_support(args.processed)
    stack, _ = stack_features(load_features(args.processed))

    def load(aous: list[str]) -> list[SpeciesData]:
        return [to_species_data(load_species_arrays(args.processed, a, species.records[a]["slug"])) for a in aous]

    fit_set = load([a for a in CURATED_TRAIN if a not in VALIDATION])
    val_set = load([a for a in CURATED_TRAIN if a in VALIDATION])

    profile = []
    for r in ratios:
        t0 = time.perf_counter()
        m = fit_scale(fit_set, stack, r, LAMBDA_CURATED, args.epochs)
        beta = torch.tensor([1.0, m * r, m])
        C = build_cost_matrix(beta, stack)
        losses = []
        for sd in val_set:
            with torch.no_grad():
                coupling = unbalanced_sinkhorn(sd.mu_t0, sd.mu_t1_sdm, C, eps=EPS, tau=TAU, n_iters=200, tol=1e-4)
                mu_pred = coupling.sum(dim=0)
                mu_pred = (mu_pred / mu_pred.sum()).numpy()
            losses.append(wasserstein_distance(mu_pred, sd.mu_t1_obs.numpy(), support.coords_m))
        profile.append({"r": r, "m": m, "beta": beta.numpy().tolist(), "loss": float(np.mean(losses))})
        print(f"r {r:5.2f}  m {m:.4f}  validation W {profile[-1]['loss']:.1f} m  ({time.perf_counter() - t0:.0f}s)")

    loss = np.array([p["loss"] for p in profile])
    r = np.array(ratios)
    best = int(np.argmin(loss))
    write_json(args.results / "ratio_profile.json", {
        "l2_lambda": LAMBDA_CURATED, "n_epochs": args.epochs,
        "r": ratios, "loss": loss.tolist(), "m": [p["m"] for p in profile],
        "minimizing_r": float(r[best]),
        "within_1pct": [float(r[loss <= loss[best] * 1.01].min()), float(r[loss <= loss[best] * 1.01].max())],
        "within_5pct": [float(r[loss <= loss[best] * 1.05].min()), float(r[loss <= loss[best] * 1.05].max())],
    })


if __name__ == "__main__":
    main()
