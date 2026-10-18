from __future__ import annotations

import torch


def unbalanced_sinkhorn(
    mu: torch.Tensor,
    nu: torch.Tensor,
    C: torch.Tensor,
    eps: float = 0.05,
    tau: float = 1.0,
    n_iters: int = 200,
    tol: float = 1e-6,
) -> torch.Tensor:
    """Log-domain entropic Sinkhorn with KL-relaxed marginals (damping tau / (tau + eps)).

    Returns the (n, m) coupling, differentiable in C, mu and nu. Large tau recovers
    balanced OT; small tau lets mass be created or destroyed.
    """
    mu = mu.to(torch.float32)
    nu = nu.to(torch.float32)
    C = C.to(torch.float32)

    log_mu = torch.log(mu)
    log_nu = torch.log(nu)
    fi = tau / (tau + eps)

    n, m = C.shape
    f = torch.zeros(n, dtype=torch.float32, device=C.device)
    g = torch.zeros(m, dtype=torch.float32, device=C.device)

    # f is pinned at -inf where mu is 0, so -inf - (-inf) = NaN would block early stopping.
    finite_mask = torch.isfinite(log_mu)

    for _ in range(n_iters):
        f_prev = f
        lse_f = torch.logsumexp((-C + g.unsqueeze(0)) / eps, dim=1)
        f = fi * (eps * log_mu - eps * lse_f)
        lse_g = torch.logsumexp((-C + f.unsqueeze(1)) / eps, dim=0)
        g = fi * (eps * log_nu - eps * lse_g)

        with torch.no_grad():
            if finite_mask.any():
                delta = torch.max(torch.abs(f[finite_mask] - f_prev[finite_mask])).item()
            else:
                delta = 0.0
        if delta < tol:
            break

    coupling = torch.exp((f.unsqueeze(1) + g.unsqueeze(0) - C) / eps)
    return coupling.to(torch.float32)
