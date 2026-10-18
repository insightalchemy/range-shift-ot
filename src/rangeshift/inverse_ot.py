from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from rangeshift.cost import beta_from_raw, build_cost_matrix
from rangeshift.forward_ot import unbalanced_sinkhorn

ANCHOR = "d_geo"


@dataclass
class SpeciesData:
    mu_t0: torch.Tensor
    mu_t1_sdm: torch.Tensor
    mu_t1_obs: torch.Tensor


def stack_features(features: dict[str, torch.Tensor]) -> tuple[torch.Tensor, list[str]]:
    """Stack feature matrices with d_geo first and the rest in alphabetical order."""
    if ANCHOR not in features:
        raise ValueError(f"features must include '{ANCHOR}'")
    keys = [ANCHOR] + sorted(k for k in features if k != ANCHOR)
    return torch.stack([features[k].to(torch.float32) for k in keys]), keys


def learn_beta(
    species_data: list[SpeciesData],
    features: dict[str, torch.Tensor],
    n_beta: int,
    eps: float,
    tau: float,
    lr: float = 1e-2,
    n_epochs: int = 200,
    outer_tau: float = 50.0,
    inner_iters: int = 100,
    inner_tol: float = 1e-4,
    outer_iters: int = 100,
    outer_tol: float = 1e-4,
    raw_init: torch.Tensor | None = None,
    history: list[dict] | None = None,
    verbose: bool = False,
    checkpoint_every: int = 0,
    checkpoint_path: str | Path | None = None,
    l2_lambda: float = 0.0,
) -> torch.Tensor:
    """Fit cost weights beta by Adam through unbalanced Sinkhorn.

    The loss is a debiased Sinkhorn divergence between the predicted and observed
    period-2 distributions under the fixed geographic cost, summed over species,
    plus an L2 penalty on the free weights. beta[0] = 1 is the geographic anchor.
    """
    feature_stack, keys = stack_features(features)
    if feature_stack.shape[0] != n_beta:
        raise ValueError(f"n_beta={n_beta} does not match {feature_stack.shape[0]} features ({keys})")
    C_geo = feature_stack[0]

    n_free = n_beta - 1
    if raw_init is None:
        inv_softplus_one = float(torch.log(torch.exp(torch.tensor(1.0)) - 1.0))
        raw = torch.full((n_free,), inv_softplus_one, dtype=torch.float32, requires_grad=True)
    else:
        raw = raw_init.clone().detach().to(torch.float32).requires_grad_(True)

    optimizer = torch.optim.Adam([raw], lr=lr)

    # The self-term for the observed distribution does not depend on beta.
    with torch.no_grad():
        cost_bb = [
            (
                unbalanced_sinkhorn(
                    sd.mu_t1_obs, sd.mu_t1_obs, C_geo, eps=eps, tau=outer_tau,
                    n_iters=outer_iters, tol=outer_tol,
                ) * C_geo
            ).sum().item()
            for sd in species_data
        ]

    for epoch in range(n_epochs):
        optimizer.zero_grad()
        epoch_loss = 0.0

        # One graph per species keeps peak memory to a single species.
        for sd, cbb in zip(species_data, cost_bb):
            beta = beta_from_raw(raw)
            C_pred = build_cost_matrix(beta, feature_stack)

            coupling_inner = unbalanced_sinkhorn(
                sd.mu_t0, sd.mu_t1_sdm, C_pred, eps=eps, tau=tau,
                n_iters=inner_iters, tol=inner_tol,
            )
            mu_t1_pred = coupling_inner.sum(dim=0)
            # Exact zeros (from zeros in mu_t1_sdm) give NaN gradients in the self-comparison term.
            mu_t1_pred = mu_t1_pred + 1e-12
            mu_t1_pred = mu_t1_pred / mu_t1_pred.sum()

            coupling_ab = unbalanced_sinkhorn(
                mu_t1_pred, sd.mu_t1_obs, C_geo, eps=eps, tau=outer_tau,
                n_iters=outer_iters, tol=outer_tol,
            )
            cost_ab = (coupling_ab * C_geo).sum()
            coupling_aa = unbalanced_sinkhorn(
                mu_t1_pred, mu_t1_pred, C_geo, eps=eps, tau=outer_tau,
                n_iters=outer_iters, tol=outer_tol,
            )
            cost_aa = (coupling_aa * C_geo).sum()

            species_loss = cost_ab - 0.5 * (cost_aa + cbb)
            species_loss.backward()
            epoch_loss += species_loss.item()

        if l2_lambda > 0.0:
            beta_reg = beta_from_raw(raw)
            reg_loss = l2_lambda * (beta_reg[1:] ** 2).sum()
            reg_loss.backward()
            epoch_loss += reg_loss.item()

        optimizer.step()

        if history is not None:
            with torch.no_grad():
                history.append({
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "beta": beta_from_raw(raw).detach().numpy().copy(),
                })
        if verbose and epoch % 10 == 0:
            with torch.no_grad():
                print(f"epoch {epoch:4d}  loss {epoch_loss:.6f}  beta {beta_from_raw(raw).numpy()}")

        if checkpoint_every and checkpoint_path and (epoch + 1) % checkpoint_every == 0:
            with torch.no_grad():
                beta_val = beta_from_raw(raw).detach().numpy().tolist()
            with open(checkpoint_path, "w") as f:
                json.dump({"epoch": epoch, "loss": epoch_loss, "beta": beta_val}, f)

    return beta_from_raw(raw).detach()
