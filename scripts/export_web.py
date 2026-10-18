"""Write the JSON files read by the project website."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.cost import build_cost_matrix
from rangeshift.inverse_ot import stack_features
from rangeshift.metrics import wasserstein_distance
from rangeshift.pipeline import (
    CELL_KM,
    CURATED,
    add_path_args,
    load_climate,
    load_features,
    load_species_arrays,
    load_species_set,
    load_support,
    method_predictions,
    predict,
    read_json,
    to_species_data,
    write_json,
)

SIG_FIGS = 5


def rounded(values) -> list[float]:
    return [float(f"{v:.{SIG_FIGS}g}") for v in np.asarray(values, dtype=np.float64).ravel()]


def species_entries(args, config: str, group: str, aous: list[str], support, clim) -> list[dict]:
    species = load_species_set(args.results, config)
    fit = np.load(args.results / f"beta_{config}.npz")
    beta = torch.as_tensor(fit["beta"], dtype=torch.float32)
    eps, tau = float(fit["eps"]), float(fit["tau"])
    stack, _ = stack_features(load_features(args.processed))
    C = build_cost_matrix(beta, stack)

    entries = []
    for aou in aous:
        rec = species.records[aou]
        arrays = load_species_arrays(args.processed, aou, rec["slug"])
        mu_learned, _ = predict(to_species_data(arrays), C, eps, tau)
        preds = method_predictions(arrays, mu_learned, support.coords_m, *clim, tau=tau)
        w_km = {
            key: wasserstein_distance(preds[method], arrays["mu_t1_obs"], support.coords_m) / 1000.0
            for key, method in (
                ("learned", "learned_cost_ot"), ("uniform", "uniform_cost_ot"), ("climate_only", "sdm_only"),
                ("velocity", "climate_velocity"), ("persistence", "persistence"),
            )
        }
        entries.append({
            "code": aou, "name": rec["name"], "group": group,
            "mu_t0": rounded(arrays["mu_t0"]), "mu_sdm": rounded(arrays["mu_t1_sdm"]), "mu_obs": rounded(arrays["mu_t1_obs"]),
            "pred": {
                "learned": rounded(preds["learned_cost_ot"]), "uniform": rounded(preds["uniform_cost_ot"]),
                "climate_only": rounded(preds["sdm_only"]), "persistence": rounded(preds["persistence"]),
            },
            "wasserstein_km": {k: float(f"{v:.{SIG_FIGS}g}") for k, v in w_km.items()},
        })
        print(f"{group:8s} {rec['name']}")
    return entries


def check(out: Path, n_cells: int) -> None:
    grid = read_json(out / "grid.json")
    species = read_json(out / "species.json")
    cost = read_json(out / "cost.json")
    profile = read_json(out / "profile.json")
    assert len(grid["cells"]) == n_cells
    for s in species:
        for key in ("mu_t0", "mu_sdm", "mu_obs"):
            assert len(s[key]) == n_cells, (s["code"], key)
        for key, arr in s["pred"].items():
            assert len(arr) == n_cells, (s["code"], key)
        assert set(s["wasserstein_km"]) == {"learned", "uniform", "climate_only", "velocity", "persistence"}
    assert len(cost["xy_km"]) == len(cost["clim"]) == len(cost["elev"]) == len(cost["mean_eco"]) == n_cells
    assert all(len(row) == 4 for row in cost["clim"]) and len(cost["beta"]) == 3
    assert len(profile["r"]) == len(profile["loss"])
    n_heldout = sum(s["group"] == "heldout" for s in species)
    print(f"ok: {n_cells} cells, {len(species)} species ({n_heldout} heldout, {len(species) - n_heldout} curated), "
          f"{len(profile['r'])} profile points")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--out", type=Path, default=Path("web_data"))
    parser.add_argument("--cost-config", choices=["curated", "unbiased"], default="unbiased",
                        help="which fitted beta is written to cost.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    support = load_support(args.processed)
    n_cells = len(support.idx)
    clim = load_climate(args.processed)

    lonlat = Transformer.from_crs("EPSG:5070", "EPSG:4326", always_xy=True)
    lon, lat = lonlat.transform(support.coords_m[:, 0].astype(np.float64), support.coords_m[:, 1].astype(np.float64))
    write_json(args.out / "grid.json", {"cells": [[round(float(x), 4), round(float(y), 4)] for x, y in zip(lon, lat)]})

    unbiased = load_species_set(args.results, "unbiased")
    entries = species_entries(args, "unbiased", "heldout", unbiased.test, support, clim)
    entries += species_entries(args, "curated", "curated", list(CURATED), support, clim)
    write_json(args.out / "species.json", entries)

    fs = np.load(args.processed / "features" / "features.npz")
    beta = torch.as_tensor(np.load(args.results / f"beta_{args.cost_config}.npz")["beta"], dtype=torch.float32)
    eco = build_cost_matrix(beta[1:], torch.stack([torch.as_tensor(fs["d_climate"]), torch.as_tensor(fs["d_elev"])]))
    write_json(args.out / "cost.json", {
        "beta": [1.0, float(beta[1]), float(beta[2])],
        "cell_km": CELL_KM,
        "xy_km": [[round(float(x), 3), round(float(y), 3)] for x, y in support.coords_m / 1000.0],
        "clim": [rounded(row) for row in fs["bio_z"]],
        "elev": rounded(fs["elev_z"]),
        "mean_eco": rounded(eco.numpy().mean(axis=1)),
    })

    profile = read_json(args.results / "ratio_profile.json")
    write_json(args.out / "profile.json", {"r": profile["r"], "loss": profile["loss"]})
    check(args.out, n_cells)


if __name__ == "__main__":
    main()
