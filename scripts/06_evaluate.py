"""Evaluate the learned cost on held-out species against four baselines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import rioxarray
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.cost import build_cost_matrix
from rangeshift.grid import coarsen, reproject_to_grid
from rangeshift.inverse_ot import stack_features
from rangeshift.pipeline import (
    BASELINES,
    CARW,
    RBWO,
    add_path_args,
    cost_to_expand,
    load_climate,
    load_features,
    load_species_arrays,
    load_species_set,
    load_support,
    method_predictions,
    northern_masses,
    percent_reduction_vs_uniform,
    predict,
    score_rows,
    shift_split,
    to_species_data,
    win_counts,
    write_json,
)

LOWER_IS_BETTER = {
    "wasserstein_m": True, "centroid_shift_m": True, "northern_half_mass_error": True, "schoeners_d": False,
}
EXPANDING_STABLE = ["07550", RBWO, CARW]
CONTRACTING = ["06220", "05010"]
EBIRD_CODES = {"woothr": "07550", "logshr": "06220", "rebwoo": "04090", "carwre": "07180", "easmea": "05010"}


def metric_wins(rows: list[dict], aous: list[str]) -> dict:
    by = {(r["aou"], r["method"]): r for r in rows}
    out = {}
    for metric, lower in LOWER_IS_BETTER.items():
        out[metric] = {}
        for baseline in BASELINES:
            wins = 0
            for a in aous:
                x, y = by[(a, "learned_cost_ot")][metric], by[(a, baseline)][metric]
                wins += (x < y) if lower else (x > y)
            out[metric][baseline] = {"wins": int(wins), "n": len(aous)}
    return out


def group_wins(rows: list[dict], groups: dict[str, list[str]]) -> dict:
    by = {(r["aou"], r["method"]): r for r in rows}
    out = {}
    for name, members in groups.items():
        out[name] = {}
        for metric, lower in LOWER_IS_BETTER.items():
            wins = 0
            for a in members:
                x, y = by[(a, "learned_cost_ot")][metric], by[(a, "persistence")][metric]
                wins += (x < y) if lower else (x > y)
            out[name][metric] = {"wins_vs_persistence": int(wins), "n": len(members)}
    return out


def ebird_band(path: Path):
    da = rioxarray.open_rasterio(path, masked=True)
    names = da.attrs.get("long_name")
    if da.sizes["band"] == 1:
        return da.squeeze("band", drop=True)
    return da.isel(band=list(names).index("breeding")).drop_vars("band", errors="ignore")


def ebird_crossval(args, support, records: dict[str, dict]) -> dict:
    rows = []
    for code, aou in EBIRD_CODES.items():
        path = args.data / "ebird" / f"{code}_abundance_seasonal_max_27km_2023.tif"
        if not path.exists():
            raise SystemExit(f"missing {path}; run 01_download.py --ebird")
        band = reproject_to_grid(ebird_band(path), support.grid.crs)
        ebird = np.nan_to_num(coarsen(band, support.grid)[support.idx], nan=0.0)
        obs = np.load(args.processed / "distributions" / f"{aou}_{records[aou]['slug']}.npz")["mu_t1_obs"]
        rho, p = spearmanr(ebird, obs)
        rows.append({"aou": aou, "species": records[aou]["name"], "spearman_rho": float(rho), "p": float(p)})
    return {"per_species": rows, "mean_spearman_rho": float(np.mean([r["spearman_rho"] for r in rows]))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--config", choices=["curated", "unbiased"], default="unbiased")
    parser.add_argument("--ebird-crossval", action="store_true", help="compare BBS with eBird Status (needs data/ebird)")
    args = parser.parse_args()

    species = load_species_set(args.results, args.config)
    support = load_support(args.processed)
    features = load_features(args.processed)
    clim_t0, clim_t1 = load_climate(args.processed)

    fit = np.load(args.results / f"beta_{args.config}.npz")
    beta = torch.as_tensor(fit["beta"], dtype=torch.float32)
    eps, tau = float(fit["eps"]), float(fit["tau"])
    stack, _ = stack_features(features)
    C = build_cost_matrix(beta, stack)
    C_np = C.numpy()

    rows, shifts, bias, nh_all, expansion = [], {}, {}, {}, {}
    for aou in species.test:
        rec = species.records[aou]
        arrays = load_species_arrays(args.processed, aou, rec["slug"])
        mu_learned, coupling = predict(to_species_data(arrays), C, eps, tau)
        preds = method_predictions(arrays, mu_learned, support.coords_m, clim_t0, clim_t1, tau=tau)
        species_rows = score_rows(rec["name"], aou, preds, arrays["mu_t1_obs"], support)
        rows.extend(species_rows)

        nh = northern_masses(arrays, mu_learned, support)
        nh_all[aou], shifts[aou] = nh, nh["shift"]
        bias[rec["name"]] = {
            "persistence": nh["t0"] - nh["obs"], "sdm_only": nh["sdm"] - nh["obs"],
            "learned_cost_ot": nh["learned"] - nh["obs"],
        }
        if aou in (RBWO, CARW):
            expansion[aou] = cost_to_expand(coupling, C_np, arrays["mu_t0"], arrays["mu_t1_obs"])
        print(f"{rec['name']:30s} learned W {species_rows[0]['wasserstein_m']:10.1f} m  persistence W {species_rows[1]['wasserstein_m']:10.1f} m")

    wins = win_counts(rows, species.test)
    result = {
        "config": args.config,
        "beta": beta.numpy().tolist(), "ratio": float(beta[1] / beta[2]),
        "n_train": len(species.train), "n_test": len(species.test),
        "win_counts": wins,
        "win_counts_by_metric": metric_wins(rows, species.test),
        "percent_reduction_vs_uniform": percent_reduction_vs_uniform(rows, species.test),
        "shift_split": shift_split(rows, species.test, shifts),
        "cost_to_expand": expansion,
        "rbwo_lower_than_carw": bool(expansion[RBWO]["cost_to_expand"] < expansion[CARW]["cost_to_expand"]) if len(expansion) == 2 else None,
        "northern_half_mass": nh_all,
        "signed_bias": bias,
        "rows": rows,
    }
    if args.config == "curated":
        result["group_wins_vs_persistence"] = group_wins(
            rows, {"expanding_stable": EXPANDING_STABLE, "contracting": CONTRACTING},
        )
    if args.ebird_crossval:
        result["ebird_crossval"] = ebird_crossval(args, support, species.records)
        write_json(args.results / "ebird_crossval.json", result["ebird_crossval"])
    write_json(args.results / f"evaluation_{args.config}.json", result)

    for baseline, w in wins.items():
        print(f"vs {baseline:18s} {w['wins']}/{w['n']}  p={w['sign_test_p']:.3g}")
    if expansion:
        print("cost to expand:", {a: round(v["cost_to_expand"], 4) for a, v in expansion.items()})


if __name__ == "__main__":
    main()
