"""Sensitivity of the curated-species result to the SDM (random forest) and the grid resolution (100 km)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.bbs import (
    apply_coverage_mask,
    build_distribution,
    consistent_coverage_mask,
    grid_raw_values,
    load_routes,
    load_species_counts,
    load_surveyed_route_years,
    route_cells,
    route_window_values,
)
from rangeshift.cost import build_cost_matrix
from rangeshift.features import BIO_VARS, extract_cell_values, pairwise_features, window_bioclim, zscore
from rangeshift.grid import build_eastern_us_grid, support_indices
from rangeshift.inverse_ot import SpeciesData, learn_beta, stack_features
from rangeshift.pipeline import (
    CARW,
    CURATED,
    CURATED_TEST,
    CURATED_TRAIN,
    EARLY_YEARS,
    EPS,
    LAMBDA_CURATED,
    LATE_YEARS,
    LR,
    OUTER_TAU,
    RBWO,
    TAU,
    Support,
    add_path_args,
    cost_to_expand,
    load_climate,
    load_features,
    load_species_arrays,
    load_species_set,
    load_support,
    method_predictions,
    predict,
    primary_design,
    read_json,
    score_rows,
    to_species_data,
    train_species_sdms,
    win_counts,
    write_json,
)

N_EPOCHS = 200
LAMBDA_RESOLUTION = 0.03
EXPANDING_STABLE = ["07550", RBWO, CARW]
CONTRACTING = ["06220", "05010"]


def checks(rows: list[dict], expansion: dict) -> dict:
    """Core ablation, expander/contractor split and RBWO vs CARW cost to expand."""
    by = {(r["aou"], r["method"]): r["wasserstein_m"] for r in rows}

    def wins(members: list[str], baseline: str) -> int:
        return sum(by[(a, "learned_cost_ot")] < by[(a, baseline)] for a in members)

    exp_wins, con_wins = wins(EXPANDING_STABLE, "persistence"), wins(CONTRACTING, "persistence")
    return {
        "beats_uniform": wins(CURATED_TEST, "uniform_cost_ot"), "n": len(CURATED_TEST),
        "expanding_stable_wins_vs_persistence": f"{exp_wins}/{len(EXPANDING_STABLE)}",
        "contracting_wins_vs_persistence": f"{con_wins}/{len(CONTRACTING)}",
        "split_holds": bool(exp_wins >= 2 and con_wins == 0),
        "cost_to_expand": expansion,
        "rbwo_lower_than_carw": bool(expansion[RBWO]["cost_to_expand"] < expansion[CARW]["cost_to_expand"]),
    }


def fit_and_evaluate(
    data: dict[str, SpeciesData], features: dict, support: Support, names: dict[str, str],
    lam: float, cell_size_m: float, epochs: int, clim: tuple | None = None,
) -> dict:
    beta = learn_beta(
        [data[a] for a in CURATED_TRAIN], features, n_beta=3, eps=EPS, tau=TAU, lr=LR, n_epochs=epochs,
        outer_tau=OUTER_TAU, inner_iters=100, outer_iters=100, l2_lambda=lam, verbose=True,
    )
    stack, _ = stack_features(features)
    C = build_cost_matrix(beta, stack)
    rows, expansion = [], {}
    for aou in CURATED_TEST:
        sd = data[aou]
        arrays = {"mu_t0": sd.mu_t0.numpy(), "mu_t1_sdm": sd.mu_t1_sdm.numpy(), "mu_t1_obs": sd.mu_t1_obs.numpy()}
        mu_learned, coupling = predict(sd, C, EPS, TAU)
        preds = method_predictions(arrays, mu_learned, support.coords_m, *(clim or (None, None)), cell_size_m=cell_size_m)
        rows.extend(score_rows(names[aou], aou, preds, arrays["mu_t1_obs"], support))
        if aou in (RBWO, CARW):
            expansion[aou] = cost_to_expand(coupling, C.numpy(), arrays["mu_t0"], arrays["mu_t1_obs"])
    return {
        "l2_lambda": lam, "beta": beta.numpy().tolist(), "ratio": float(beta[1] / beta[2]),
        "win_counts": win_counts(rows, CURATED_TEST, baselines=[m for m in preds if m != "learned_cost_ot"]),
        "checks": checks(rows, expansion), "rows": rows,
    }


def rf_sdm(args, support: Support, species_set) -> dict:
    bbs_dir = args.data / "bbs"
    routes = load_routes(bbs_dir)
    surveyed = load_surveyed_route_years(bbs_dir)
    counts = load_species_counts(bbs_dir, set(CURATED))
    late = np.load(args.processed / "features" / "late_bioclim.npz")
    route_keys, design_route, design_late = primary_design(
        routes, support.grid, support.idx, args.data / "worldclim", {v: late[v] for v in BIO_VARS},
    )
    fitted = train_species_sdms(counts, routes, surveyed, EARLY_YEARS, route_keys, design_route, design_late, kind="rf")

    data = {}
    for aou, slug in CURATED.items():
        arrays = load_species_arrays(args.processed, aou, slug)
        mu_sdm = fitted[aou][0]
        arrays["mu_t1_sdm"] = mu_sdm / mu_sdm.sum()
        data[aou] = to_species_data(arrays)

    names = {a: species_set.records[a]["name"] for a in CURATED}
    result = fit_and_evaluate(data, load_features(args.processed), support, names, LAMBDA_CURATED, 50_000.0,
                              args.epochs, clim=load_climate(args.processed))
    result["auc"] = {a: fitted[a][1] for a in CURATED}
    # Baselines are those of the primary (logistic-SDM) evaluation.
    primary = read_json(args.results / "evaluation_curated.json")["rows"]
    by = {(r["aou"], r["method"]): r["wasserstein_m"] for r in primary}
    rf_w = {r["aou"]: r["wasserstein_m"] for r in result["rows"] if r["method"] == "learned_cost_ot"}
    result["vs_primary_baselines"] = {
        "beats_uniform": sum(rf_w[a] < by[(a, "uniform_cost_ot")] for a in CURATED_TEST),
        "beats_persistence": sum(rf_w[a] < by[(a, "persistence")] for a in CURATED_TEST),
    }
    return result


def resolution_100km(args, species_set) -> dict:
    cell_m = 100_000.0
    grid = build_eastern_us_grid(cell_m)
    bbs_dir, baseline = args.data / "bbs", args.data / "worldclim"
    routes = load_routes(bbs_dir)
    surveyed = load_surveyed_route_years(bbs_dir)
    mask = consistent_coverage_mask(routes, surveyed, grid, EARLY_YEARS, LATE_YEARS)
    idx = support_indices(mask)
    coords_m = grid.cell_centers()[idx]
    support = Support(grid, mask, idx, coords_m)
    print(f"100 km grid {grid.n_x}x{grid.n_y}, support {len(idx)} cells")

    counts = load_species_counts(bbs_dir, set(CURATED))
    cells = route_cells(routes, grid)

    def mu(aou: str, years: range) -> np.ndarray:
        values = route_window_values(routes, counts[aou], surveyed, years)
        return build_distribution(apply_coverage_mask(grid_raw_values(values, cells, grid.n_cells), mask))[idx]

    bio_z = np.stack([zscore(extract_cell_values(baseline / f"wc2.1_10m_{v}.tif", grid)[idx]) for v in BIO_VARS], axis=1)
    elev_z = zscore(extract_cell_values(baseline / "wc2.1_10m_elev.tif", grid)[idx])
    features = pairwise_features(coords_m, bio_z, elev_z, cell_m / 1000.0)

    late = window_bioclim(grid, LATE_YEARS, args.data / "worldclim_monthly")
    route_keys, design_route, design_late = primary_design(routes, grid, idx, baseline, late)
    fitted = train_species_sdms(counts, routes, surveyed, EARLY_YEARS, route_keys, design_route, design_late)

    data = {}
    for aou in CURATED:
        mu_sdm = fitted[aou][0]
        data[aou] = SpeciesData(
            mu_t0=torch.as_tensor(mu(aou, EARLY_YEARS), dtype=torch.float32),
            mu_t1_sdm=torch.as_tensor(mu_sdm / mu_sdm.sum(), dtype=torch.float32),
            mu_t1_obs=torch.as_tensor(mu(aou, LATE_YEARS), dtype=torch.float32),
        )
    names = {a: species_set.records[a]["name"] for a in CURATED}
    result = fit_and_evaluate(data, features, support, names, LAMBDA_RESOLUTION, cell_m, args.epochs)
    result["n_cells"] = len(idx)
    result["auc"] = {a: fitted[a][1] for a in CURATED}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--axis", choices=["sdm", "resolution", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    args = parser.parse_args()
    species_set = load_species_set(args.results, "curated")

    if args.axis in ("sdm", "all"):
        result = rf_sdm(args, load_support(args.processed), species_set)
        write_json(args.results / "sensitivity_sdm.json", result)
        print("random forest SDM:", result["checks"])
    if args.axis in ("resolution", "all"):
        result = resolution_100km(args, species_set)
        write_json(args.results / "sensitivity_resolution.json", result)
        print("100 km grid:", result["checks"])


if __name__ == "__main__":
    main()
