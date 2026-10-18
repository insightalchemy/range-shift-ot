"""Fit a logistic climate-envelope SDM per species on early-window routes and project it onto period-2 climate."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.bbs import load_routes, load_species_counts, load_surveyed_route_years
from rangeshift.features import BIO_VARS
from rangeshift.pipeline import EARLY_YEARS, add_path_args, load_support, primary_design, read_json, train_species_sdms, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    args = parser.parse_args()
    support = load_support(args.processed)
    grid, idx = support.grid, support.idx
    bbs_dir, baseline = args.data / "bbs", args.data / "worldclim"

    species = read_json(args.results / "inclusion.json")["species"]
    routes = load_routes(bbs_dir)
    surveyed = load_surveyed_route_years(bbs_dir)
    counts = load_species_counts(bbs_dir, {r["aou"] for r in species})

    late = np.load(args.processed / "features" / "late_bioclim.npz")
    route_keys, design_route, design_late = primary_design(routes, grid, idx, baseline, {v: late[v] for v in BIO_VARS})
    fitted = train_species_sdms(counts, routes, surveyed, EARLY_YEARS, route_keys, design_route, design_late)

    out_dir = args.processed / "sdm"
    out_dir.mkdir(parents=True, exist_ok=True)
    aucs = {}
    for r in species:
        mu_sdm, auc = fitted[r["aou"]]
        np.savez(out_dir / f"{r['aou']}_{r['slug']}.npz", mu_t1_sdm=mu_sdm)
        aucs[r["aou"]] = auc

    values = np.array(list(aucs.values()))
    write_json(args.results / "sdm_auc.json", {
        "n_species": len(aucs),
        "mean": float(values.mean()), "median": float(np.median(values)),
        "min": float(values.min()), "max": float(values.max()),
        "below_0.7": [a for a, v in aucs.items() if v < 0.7],
        "auc": aucs,
    })
    print(f"{len(aucs)} SDMs, AUC mean {values.mean():.3f}, median {np.median(values):.3f}, range [{values.min():.3f}, {values.max():.3f}]")


if __name__ == "__main__":
    main()
