"""Fit the cost weights beta on the training species."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.inverse_ot import learn_beta
from rangeshift.pipeline import (
    EPS,
    LAMBDA_CURATED,
    LAMBDA_UNBIASED,
    LR,
    OUTER_TAU,
    TAU,
    add_path_args,
    load_features,
    load_species_arrays,
    load_species_set,
    to_species_data,
    write_json,
)

DEFAULTS = {
    "curated": {"epochs": 400, "lam": LAMBDA_CURATED},
    "unbiased": {"epochs": 200, "lam": LAMBDA_UNBIASED},
}
DRIFT_WINDOW = 50


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--config", choices=list(DEFAULTS), default="unbiased")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lam", type=float, help="L2 weight on the free betas")
    args = parser.parse_args()
    epochs = args.epochs or DEFAULTS[args.config]["epochs"]
    lam = args.lam if args.lam is not None else DEFAULTS[args.config]["lam"]

    species = load_species_set(args.results, args.config)
    features = load_features(args.processed)
    train = [to_species_data(load_species_arrays(args.processed, a, species.records[a]["slug"])) for a in species.train]
    print(f"{args.config}: {len(train)} training species, lambda {lam}, {epochs} epochs")

    history: list[dict] = []
    t0 = time.perf_counter()
    beta = learn_beta(
        train, features, n_beta=3, eps=EPS, tau=TAU, lr=LR, n_epochs=epochs, outer_tau=OUTER_TAU,
        inner_iters=100, outer_iters=100, l2_lambda=lam, history=history, verbose=True,
        checkpoint_every=20, checkpoint_path=args.processed / f"fit_{args.config}_checkpoint.json",
    )
    wall = time.perf_counter() - t0

    b1 = [h["beta"][1] for h in history]
    drift_last = float((b1[-1] - b1[-1 - DRIFT_WINDOW]) / DRIFT_WINDOW) if len(b1) > DRIFT_WINDOW else None
    drift_prev = float((b1[-1 - DRIFT_WINDOW] - b1[-1 - 2 * DRIFT_WINDOW]) / DRIFT_WINDOW) if len(b1) > 2 * DRIFT_WINDOW else None

    args.results.mkdir(parents=True, exist_ok=True)
    np.savez(args.results / f"beta_{args.config}.npz", beta=beta.numpy(), eps=EPS, tau=TAU, outer_tau=OUTER_TAU, l2_lambda=lam)
    write_json(args.results / f"fit_{args.config}.json", {
        "config": args.config, "eps": EPS, "tau": TAU, "outer_tau": OUTER_TAU, "lr": LR,
        "l2_lambda": lam, "n_epochs": epochs, "train": species.train,
        "beta": beta.numpy().tolist(), "ratio": float(beta[1] / beta[2]),
        "drift_last_50": drift_last, "drift_prev_50": drift_prev,
        "wall_clock_s": wall,
        "loss": [h["loss"] for h in history],
        "beta_trajectory": [h["beta"].tolist() for h in history],
    })
    print(f"beta {beta.numpy()}  ratio {float(beta[1] / beta[2]):.4f}  wall {wall:.0f}s")


if __name__ == "__main__":
    main()
