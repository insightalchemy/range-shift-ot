"""Draw the figures from the saved fits and evaluations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.cost import build_cost_matrix
from rangeshift.inverse_ot import stack_features
from rangeshift.pipeline import (
    CURATED_TEST,
    RBWO,
    add_path_args,
    load_features,
    load_species_arrays,
    load_species_set,
    load_support,
    predict,
    read_json,
    to_species_data,
)
from rangeshift.visualize import (
    DPI,
    plot_baseline_comparison,
    plot_cost_surface,
    plot_loss_curve,
    plot_ratio_profile,
    plot_signed_bias,
    plot_species_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--config", choices=["curated", "unbiased"], default="unbiased")
    args = parser.parse_args()
    out = args.results / "figures"
    out.mkdir(parents=True, exist_ok=True)

    species = load_species_set(args.results, args.config)
    support = load_support(args.processed)
    features = load_features(args.processed)
    stack, _ = stack_features(features)
    fit = np.load(args.results / f"beta_{args.config}.npz")
    beta = torch.as_tensor(fit["beta"], dtype=torch.float32)
    evaluation = read_json(args.results / f"evaluation_{args.config}.json")
    names = {a: species.records[a]["name"] for a in CURATED_TEST}

    def save(fig, name: str) -> None:
        fig.savefig(out / name, bbox_inches="tight", dpi=DPI)
        print(f"wrote {out / name}")

    ecological = build_cost_matrix(beta[1:], stack[1:])
    save(plot_cost_surface(ecological.numpy().mean(axis=1), support.coords_m), f"cost_surface_{args.config}.png")

    arrays = load_species_arrays(args.processed, RBWO, species.records[RBWO]["slug"])
    mu_pred, _ = predict(to_species_data(arrays), build_cost_matrix(beta, stack), float(fit["eps"]), float(fit["tau"]))
    save(plot_species_panel(arrays["mu_t0"], arrays["mu_t1_sdm"], mu_pred, arrays["mu_t1_obs"], support.coords_m),
         f"species_panel_rbwo_{args.config}.png")

    save(plot_baseline_comparison(evaluation["rows"], CURATED_TEST, names), f"baseline_comparison_{args.config}.png")
    save(plot_signed_bias({names[a]: evaluation["signed_bias"][names[a]] for a in CURATED_TEST}), f"signed_bias_{args.config}.png")
    save(plot_loss_curve(read_json(args.results / f"fit_{args.config}.json")["loss"]), f"loss_curve_{args.config}.png")

    profile_path = args.results / "ratio_profile.json"
    if profile_path.exists():
        profile = read_json(profile_path)
        save(plot_ratio_profile(profile["r"], profile["loss"]), "ratio_profile.png")


if __name__ == "__main__":
    main()
