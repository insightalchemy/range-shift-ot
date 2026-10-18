"""Fit beta on the 1995-2000 -> 2006-2011 transition and apply it unchanged to 2006-2011 -> 2017-2022."""
from __future__ import annotations

import argparse
import math
import sys
import time
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
from rangeshift.features import BIO_VARS, extract_cell_values, pairwise_features, window_bioclim
from rangeshift.grid import support_indices
from rangeshift.inverse_ot import SpeciesData, learn_beta, stack_features
from rangeshift.metrics import centroid_shift_error
from rangeshift.pipeline import (
    CELL_KM,
    EPS,
    LAMBDA_TEMPORAL,
    LR,
    OUTER_TAU,
    TAU,
    WINDOWS,
    Support,
    add_path_args,
    eastern_routes,
    load_support,
    load_species_set,
    method_predictions,
    northern_masses,
    percent_reduction_vs_uniform,
    predict,
    score_rows,
    shift_split,
    train_species_sdms,
    win_counts,
    write_json,
)
from rangeshift.sdm import build_design_matrix

N_EPOCHS = 200
CONFORMAL_ALPHA = 0.10


def build_transition(grid, routes, surveyed, counts, early: range, late: range, bio_late, baseline: Path) -> dict:
    mask = consistent_coverage_mask(routes, surveyed, grid, early, late)
    idx = support_indices(mask)
    cells = route_cells(routes, grid)

    def mu(aou: str, years: range) -> np.ndarray:
        values = route_window_values(routes, counts[aou], surveyed, years)
        return build_distribution(apply_coverage_mask(grid_raw_values(values, cells, grid.n_cells), mask))[idx]

    return {
        "mask": mask, "idx": idx,
        "mu": {aou: (mu(aou, early), mu(aou, late)) for aou in counts},
        "bio_late": {v: bio_late[v][idx] for v in BIO_VARS},
        "elev": extract_cell_values(baseline / "wc2.1_10m_elev.tif", grid)[idx],
        "coords_m": grid.cell_centers()[idx],
    }


def shared_stats(grid, ab: dict, bc: dict, baseline: Path) -> dict:
    """z-score statistics pooled over both transitions so one beta means the same on each."""
    union = sorted(set(ab["idx"].tolist()) | set(bc["idx"].tolist()))
    elev = extract_cell_values(baseline / "wc2.1_10m_elev.tif", grid)[union]
    stats = {"elev": {"mean": float(elev.mean()), "std": float(elev.std())}}
    for v in BIO_VARS:
        pooled = np.concatenate([ab["bio_late"][v], bc["bio_late"][v]])
        stats[v] = {"mean": float(pooled.mean()), "std": float(pooled.std())}
    return stats


def standardize(bio: dict[str, np.ndarray], stats: dict) -> np.ndarray:
    return np.stack([(bio[v] - stats[v]["mean"]) / stats[v]["std"] for v in BIO_VARS], axis=1)


def transition_features(tr: dict, stats: dict) -> dict[str, torch.Tensor]:
    elev_z = (tr["elev"] - stats["elev"]["mean"]) / stats["elev"]["std"]
    return pairwise_features(tr["coords_m"], standardize(tr["bio_late"], stats), elev_z, CELL_KM)


def transition_sdms(grid, routes, surveyed, counts, early: range, tr: dict, bio_early_full: dict) -> dict:
    route_keys = eastern_routes(routes, grid)
    lons = np.array([routes[k][0] for k in route_keys])
    lats = np.array([routes[k][1] for k in route_keys])
    ix, iy = grid.cell_indices(lons, lats)
    flat = grid.flat_index(ix, iy)
    route_bio = np.zeros((len(route_keys), len(BIO_VARS)))
    for j, v in enumerate(BIO_VARS):
        route_bio[:, j] = bio_early_full[v][flat]
    mean, std = np.nanmean(route_bio, axis=0), np.nanstd(route_bio, axis=0)

    late_bio = np.stack([tr["bio_late"][v] for v in BIO_VARS], axis=1)
    return train_species_sdms(
        counts, routes, surveyed, early, route_keys,
        build_design_matrix(route_bio, mean, std), build_design_matrix(late_bio, mean, std),
    )


def species_data(tr: dict, sdms: dict) -> dict[str, SpeciesData]:
    out = {}
    for aou, (mu_t0, mu_t1_obs) in tr["mu"].items():
        if aou in sdms:
            out[aou] = SpeciesData(
                mu_t0=torch.as_tensor(mu_t0, dtype=torch.float32),
                mu_t1_sdm=torch.as_tensor(sdms[aou][0], dtype=torch.float32),
                mu_t1_obs=torch.as_tensor(mu_t1_obs, dtype=torch.float32),
            )
    return out


def leave_one_out_conformal(pred: dict[str, float], actual: dict[str, float], alpha: float) -> dict:
    """Split-conformal intervals for each species, calibrated on the residuals of all the others."""
    aous = sorted(pred)
    residual = {a: abs(actual[a] - pred[a]) for a in aous}
    rows = []
    for held_out in aous:
        cal = sorted(residual[a] for a in aous if a != held_out)
        k = math.ceil((1 - alpha) * (len(cal) + 1))
        radius = float("inf") if k > len(cal) else cal[k - 1]
        lo, hi = pred[held_out] - radius, pred[held_out] + radius
        rows.append({"aou": held_out, "actual": actual[held_out], "pred": pred[held_out],
                     "lower": lo, "upper": hi, "covered": bool(lo <= actual[held_out] <= hi)})
    widths = [r["upper"] - r["lower"] for r in rows if math.isfinite(r["upper"] - r["lower"])]
    return {
        "alpha": alpha, "n": len(rows), "n_covered": sum(r["covered"] for r in rows),
        "empirical_coverage": sum(r["covered"] for r in rows) / len(rows),
        "mean_width_m": float(np.mean(widths)) if widths else None, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--config", choices=["curated", "unbiased"], default="unbiased")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--lam", type=float, default=LAMBDA_TEMPORAL)
    args = parser.parse_args()

    species = load_species_set(args.results, args.config)
    support = load_support(args.processed)
    grid = support.grid
    bbs_dir, baseline = args.data / "bbs", args.data / "worldclim"
    monthly = args.data / "worldclim_monthly"

    routes = load_routes(bbs_dir)
    surveyed = load_surveyed_route_years(bbs_dir)
    counts = load_species_counts(bbs_dir, set(species.records))

    bio_A = window_bioclim(grid, WINDOWS["A"], monthly)
    bio_B = window_bioclim(grid, WINDOWS["B"], monthly)
    late = np.load(args.processed / "features" / "late_bioclim.npz")
    bio_C = {v: late[v] for v in BIO_VARS}
    bio1 = {k: float(np.nanmean(b["bio_1"])) for k, b in (("A", bio_A), ("B", bio_B), ("C", bio_C))}

    ab = build_transition(grid, routes, surveyed, counts, WINDOWS["A"], WINDOWS["B"], bio_B, baseline)
    bc = build_transition(grid, routes, surveyed, counts, WINDOWS["B"], WINDOWS["C"], bio_C, baseline)
    stats = shared_stats(grid, ab, bc, baseline)
    feat_ab, feat_bc = transition_features(ab, stats), transition_features(bc, stats)
    print(f"support A-B {len(ab['idx'])} cells, B-C {len(bc['idx'])} cells")

    sdms_ab = transition_sdms(grid, routes, surveyed, counts, WINDOWS["A"], ab, bio_A)
    sdms_bc = transition_sdms(grid, routes, surveyed, counts, WINDOWS["B"], bc, bio_B)
    data_ab, data_bc = species_data(ab, sdms_ab), species_data(bc, sdms_bc)

    train = [a for a in species.train if a in data_ab]
    test = [a for a in species.test if a in data_bc]
    t0 = time.perf_counter()
    beta = learn_beta(
        [data_ab[a] for a in train], feat_ab, n_beta=3, eps=EPS, tau=TAU, lr=LR, n_epochs=args.epochs,
        outer_tau=OUTER_TAU, inner_iters=100, outer_iters=100, l2_lambda=args.lam, verbose=True,
    )
    wall = time.perf_counter() - t0
    args.results.mkdir(parents=True, exist_ok=True)
    np.savez(args.results / f"beta_temporal_{args.config}.npz", beta=beta.numpy(), eps=EPS, tau=TAU, l2_lambda=args.lam)

    stack, _ = stack_features(feat_bc)
    C = build_cost_matrix(beta, stack)
    bc_support = Support(grid, bc["mask"], bc["idx"], bc["coords_m"])
    clim_t0, clim_t1 = standardize({v: bio_B[v][bc["idx"]] for v in BIO_VARS}, stats), standardize(bc["bio_late"], stats)

    rows, shifts, pred_shift, true_shift = [], {}, {}, {}
    for aou in test:
        sd = data_bc[aou]
        arrays = {"mu_t0": sd.mu_t0.numpy(), "mu_t1_sdm": sd.mu_t1_sdm.numpy(), "mu_t1_obs": sd.mu_t1_obs.numpy()}
        mu_learned, _ = predict(sd, C, EPS, TAU)
        preds = method_predictions(arrays, mu_learned, bc["coords_m"], clim_t0, clim_t1)
        rows.extend(score_rows(species.records[aou]["name"], aou, preds, arrays["mu_t1_obs"], bc_support))
        shifts[aou] = northern_masses(arrays, mu_learned, bc_support)["shift"]
        pred_shift[aou] = centroid_shift_error(mu_learned, arrays["mu_t0"], bc["coords_m"])
        true_shift[aou] = centroid_shift_error(arrays["mu_t0"], arrays["mu_t1_obs"], bc["coords_m"])

    wins = win_counts(rows, test)
    write_json(args.results / f"temporal_transfer_{args.config}.json", {
        "config": args.config,
        "windows": {k: f"{w.start}-{w.stop - 1}" for k, w in WINDOWS.items()},
        "bio1_mean": bio1,
        "shared_stats": stats,
        "support_cells": {"A_to_B": len(ab["idx"]), "B_to_C": len(bc["idx"])},
        "l2_lambda": args.lam, "n_epochs": args.epochs,
        "n_train": len(train), "n_test": len(test),
        "beta": beta.numpy().tolist(), "ratio": float(beta[1] / beta[2]), "wall_clock_s": wall,
        "sdm_auc_mean": {
            "A_to_B": float(np.mean([v[1] for v in sdms_ab.values()])),
            "B_to_C": float(np.mean([v[1] for v in sdms_bc.values()])),
        },
        "win_counts": wins,
        "percent_reduction_vs_uniform": percent_reduction_vs_uniform(rows, test),
        "shift_split": shift_split(rows, test, shifts),
        "conformal_centroid_shift": leave_one_out_conformal(pred_shift, true_shift, CONFORMAL_ALPHA),
        "rows": rows,
    })
    for baseline_name, w in wins.items():
        print(f"vs {baseline_name:18s} {w['wins']}/{w['n']}  p={w['sign_test_p']:.3g}")


if __name__ == "__main__":
    main()
