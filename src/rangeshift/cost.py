from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pairwise_euclidean(points: np.ndarray) -> torch.Tensor:
    """Pairwise Euclidean distances between the rows of an (n, d) array, float32."""
    pts = torch.as_tensor(np.asarray(points), dtype=torch.float32)
    diff = pts.unsqueeze(1) - pts.unsqueeze(0)
    return torch.sqrt(torch.clamp((diff * diff).sum(dim=-1), min=0.0)).to(torch.float32)


def pairwise_abs_difference(feature: np.ndarray) -> torch.Tensor:
    """Pairwise |f_i - f_j| of a per-cell scalar feature, float32."""
    f = torch.as_tensor(np.asarray(feature), dtype=torch.float32).reshape(-1)
    return (f.unsqueeze(1) - f.unsqueeze(0)).abs().to(torch.float32)


def beta_from_raw(raw_beta: torch.Tensor) -> torch.Tensor:
    """Prepend the fixed geographic anchor beta[0] = 1 and map the free weights through softplus."""
    raw_beta = raw_beta.to(torch.float32)
    anchor = torch.ones(1, dtype=torch.float32, device=raw_beta.device)
    return torch.cat([anchor, F.softplus(raw_beta)])


def build_cost_matrix(beta: torch.Tensor, feature_stack: torch.Tensor) -> torch.Tensor:
    """C = sum_k beta[k] * feature_stack[k]."""
    beta = beta.to(torch.float32)
    feature_stack = feature_stack.to(torch.float32)
    if beta.shape[0] != feature_stack.shape[0]:
        raise ValueError(f"beta has {beta.shape[0]} entries, feature_stack has {feature_stack.shape[0]}")
    return torch.einsum("k,kij->ij", beta, feature_stack).to(torch.float32)
