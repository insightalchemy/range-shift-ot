import numpy as np
import ot
import torch

from rangeshift.forward_ot import unbalanced_sinkhorn


def random_problem(n: int = 40, seed: int = 0):
    rng = np.random.default_rng(seed)
    a = rng.random(n)
    a /= a.sum()
    b = rng.random(n)
    b /= b.sum()
    M = ot.dist(rng.random((n, 2)), rng.random((n, 2)))
    M /= M.max()
    return a, b, M


def test_balanced_case_matches_pot():
    a, b, M = random_problem(seed=1)
    eps = 0.05
    G_pot = ot.sinkhorn(a, b, M, reg=eps, numItermax=5000, stopThr=1e-12)
    G = unbalanced_sinkhorn(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32),
        torch.tensor(M, dtype=torch.float32), eps=eps, tau=1e6, n_iters=5000, tol=1e-12,
    )
    assert np.abs(G.detach().numpy() - G_pot).max() < 1e-4


def test_unbalanced_case_matches_pot():
    a, b, M = random_problem(seed=2)
    eps, tau = 0.05, 0.8
    G_pot = ot.unbalanced.sinkhorn_knopp_unbalanced(
        a, b, M, reg=eps, reg_m=tau, reg_type="entropy", numItermax=5000, stopThr=1e-12,
    )
    G = unbalanced_sinkhorn(
        torch.tensor(a, dtype=torch.float32), torch.tensor(b, dtype=torch.float32),
        torch.tensor(M, dtype=torch.float32), eps=eps, tau=tau, n_iters=5000, tol=1e-12,
    )
    assert np.abs(G.detach().numpy() - G_pot).max() < 1e-4


def test_gradients_flow_through_cost():
    n = 30
    torch.manual_seed(0)
    mu = torch.softmax(torch.randn(n), dim=0)
    nu = torch.softmax(torch.randn(n), dim=0)
    C = torch.rand(n, n, requires_grad=True)
    unbalanced_sinkhorn(mu, nu, C, eps=0.05, tau=1.0, n_iters=100).sum().backward()
    assert C.grad is not None
    assert torch.isfinite(C.grad).all()
    assert C.grad.norm().item() > 0.0


def test_sparse_source_is_finite():
    n = 50
    torch.manual_seed(0)
    mu = torch.zeros(n)
    mu[:3] = torch.tensor([0.7, 0.2, 0.1])
    nu = torch.softmax(torch.randn(n), dim=0)
    coupling = unbalanced_sinkhorn(mu, nu, torch.rand(n, n), eps=0.01, tau=1.0, n_iters=200)
    assert not torch.isnan(coupling).any()
    assert not torch.isinf(coupling).any()
