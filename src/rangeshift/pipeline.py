from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

from rangeshift.baselines import climate_velocity, persistence, sdm_only, uniform_cost_ot
from rangeshift.bbs import RouteKey, RouteYearKey, route_window_values
from rangeshift.features import BIO_VARS, extract_cell_values
from rangeshift.forward_ot import unbalanced_sinkhorn
from rangeshift.grid import AnalysisGrid, grid_from_meta, load_raster, point_sample_raster, support_indices
from rangeshift.inverse_ot import SpeciesData
from rangeshift.metrics import (
    centroid_shift_error,
    northern_half_mass,
    northern_half_mass_error,
    schoeners_d,
    wasserstein_distance,
)
from rangeshift.sdm import build_design_matrix, fit_sdm, holdout_auc, project_sdm

EPS = 0.05
TAU = 5.0
OUTER_TAU = 50.0
LR = 0.05
CELL_KM = 50.0
EARLY_YEARS = range(1997, 2003)
LATE_YEARS = range(2017, 2023)
WINDOWS = {"A": range(1995, 2001), "B": range(2006, 2012), "C": range(2017, 2023)}

# Fixed L2 weights per analysis.
LAMBDA_CURATED = 0.01
LAMBDA_UNBIASED = 0.0001
LAMBDA_TEMPORAL = 0.03

CURATED = {
    "04610": "eastern_wood_pewee", "05110": "common_grackle", "06220": "loggerhead_shrike",
    "05010": "eastern_meadowlark", "07550": "wood_thrush", "07660": "eastern_bluebird",
    "07050": "brown_thrasher", "05870": "eastern_towhee", "06140": "tree_swallow",
    "07180": "carolina_wren", "04090": "red_bellied_woodpecker", "04050": "pileated_woodpecker",
    "07030": "northern_mockingbird",
}
CURATED_TEST = ["07550", "06220", "04090", "07180", "05010"]
CURATED_TRAIN = sorted(set(CURATED) - set(CURATED_TEST))
RBWO, CARW = "04090", "07180"
SHIFT_THRESHOLD = 0.02
METHODS = ["learned_cost_ot", "persistence", "sdm_only", "uniform_cost_ot", "climate_velocity"]
BASELINES = METHODS[1:]


def slugify(name: str) -> str:
    return re.sub(r"[ \-/]", "_", name.lower().replace("'", ""))


def add_path_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--processed", type=Path, default=Path("processed"))
    parser.add_argument("--results", type=Path, default=Path("results"))


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with open(path) as f:
        return json.load(f)


@dataclass
class Support:
    grid: AnalysisGrid
    mask: np.ndarray
    idx: np.ndarray
    coords_m: np.ndarray


def load_support(processed: Path) -> Support:
    grid, mask = grid_from_meta(processed / "grid_meta.npz")
    idx = support_indices(mask)
    return Support(grid, mask, idx, grid.cell_centers()[idx])


def load_features(processed: Path) -> dict[str, torch.Tensor]:
    fs = np.load(processed / "features" / "features.npz")
    return {k: torch.as_tensor(fs[k], dtype=torch.float32) for k in ("d_geo", "d_climate", "d_elev")}


def load_climate(processed: Path) -> tuple[np.ndarray, np.ndarray]:
    fs = np.load(processed / "features" / "features.npz")
    return fs["clim_t0"], fs["clim_t1"]


def load_species_arrays(processed: Path, aou: str, slug: str) -> dict[str, np.ndarray]:
    dist = np.load(processed / "distributions" / f"{aou}_{slug}.npz")
    sdm = np.load(processed / "sdm" / f"{aou}_{slug}.npz")
    out = {}
    for key, arr in (("mu_t0", dist["mu_t0"]), ("mu_t1_obs", dist["mu_t1_obs"]), ("mu_t1_sdm", sdm["mu_t1_sdm"])):
        arr = arr.astype(np.float32)
        out[key] = arr / arr.sum()
    return out


def to_species_data(arrays: dict[str, np.ndarray]) -> SpeciesData:
    return SpeciesData(
        mu_t0=torch.as_tensor(arrays["mu_t0"], dtype=torch.float32),
        mu_t1_sdm=torch.as_tensor(arrays["mu_t1_sdm"], dtype=torch.float32),
        mu_t1_obs=torch.as_tensor(arrays["mu_t1_obs"], dtype=torch.float32),
    )


@dataclass
class SpeciesSet:
    name: str
    records: dict[str, dict]
    train: list[str]
    test: list[str]


def load_species_set(results: Path, name: str) -> SpeciesSet:
    inclusion = read_json(results / "inclusion.json")
    records = {r["aou"]: r for r in inclusion["species"]}
    if name == "curated":
        return SpeciesSet(name, {a: records[a] for a in CURATED}, list(CURATED_TRAIN), list(CURATED_TEST))
    if name == "unbiased":
        split = read_json(results / "split.json")
        return SpeciesSet(name, records, split["train"], split["test"])
    raise ValueError(f"unknown species set: {name}")


def make_split(species: list[dict], must_test: list[str], seed: int = 42, test_fraction: float = 1 / 3) -> dict:
    """Split stratified on data-sufficiency quartile; the species in must_test are always held out."""
    rng = np.random.RandomState(seed)
    mandatory = [a for a in must_test if any(r["aou"] == a for r in species)]
    remaining = [r for r in species if r["aou"] not in must_test]
    n_extra = max(0, round(len(species) * test_fraction) - len(mandatory))

    min_occ = np.array([r["min_occupied"] for r in remaining])
    quartile = np.digitize(min_occ, np.percentile(min_occ, [25, 50, 75]))
    test_extra, train = [], []
    for q in range(4):
        members = [r for r, qi in zip(remaining, quartile) if qi == q]
        rng.shuffle(members)
        k = round(len(members) * (n_extra / len(remaining)))
        test_extra.extend(r["aou"] for r in members[:k])
        train.extend(r["aou"] for r in members[k:])
    return {"seed": seed, "train": train, "test": mandatory + test_extra}


def predict(sd: SpeciesData, C: torch.Tensor, eps: float, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Forward-model forecast (normalized) and its coupling."""
    with torch.no_grad():
        coupling = unbalanced_sinkhorn(sd.mu_t0, sd.mu_t1_sdm, C, eps=eps, tau=tau, n_iters=300, tol=1e-4)
        mu_pred = coupling.sum(dim=0)
        mu_pred = (mu_pred / mu_pred.sum()).numpy()
    return mu_pred, coupling.numpy()


def method_predictions(
    arrays: dict[str, np.ndarray], mu_learned: np.ndarray, coords_m: np.ndarray,
    clim_t0: np.ndarray | None = None, clim_t1: np.ndarray | None = None,
    tau: float = TAU, cell_size_m: float = CELL_KM * 1000.0,
) -> dict[str, np.ndarray]:
    """Learned forecast and baselines; climate velocity is included only when climate arrays are given."""
    mu_t0, mu_t1_sdm = arrays["mu_t0"], arrays["mu_t1_sdm"]
    preds = {
        "learned_cost_ot": mu_learned,
        "persistence": persistence(mu_t0),
        "sdm_only": sdm_only(mu_t1_sdm),
        "uniform_cost_ot": uniform_cost_ot(
            mu_t0, mu_t1_sdm, coords_m, tau=tau, cell_size_m=cell_size_m, n_iters=300, tol=1e-4,
        ),
    }
    if clim_t0 is not None:
        preds["climate_velocity"] = climate_velocity(mu_t0, clim_t0, clim_t1)
    return preds


def score_rows(
    name: str, aou: str, preds: dict[str, np.ndarray], mu_t1_obs: np.ndarray,
    support: Support,
) -> list[dict]:
    grid = support.grid
    return [
        {
            "species": name, "aou": aou, "method": method,
            "wasserstein_m": wasserstein_distance(pred, mu_t1_obs, support.coords_m),
            "centroid_shift_m": centroid_shift_error(pred, mu_t1_obs, support.coords_m),
            "northern_half_mass_error": northern_half_mass_error(pred, mu_t1_obs, grid.n_x, grid.n_y, support.idx),
            "schoeners_d": schoeners_d(pred, mu_t1_obs),
        }
        for method, pred in preds.items()
    ]


def win_counts(
    rows: list[dict], aous: list[str], learned: str = "learned_cost_ot", baselines: list[str] = BASELINES,
) -> dict:
    """Wins of the learned model over each baseline on Wasserstein, with one-sided sign tests."""
    by = {(r["aou"], r["method"]): r for r in rows}
    out = {}
    for baseline in baselines:
        wins = sum(by[(a, learned)]["wasserstein_m"] < by[(a, baseline)]["wasserstein_m"] for a in aous)
        out[baseline] = {
            "wins": int(wins), "n": len(aous),
            "sign_test_p": float(binomtest(int(wins), len(aous), p=0.5, alternative="greater").pvalue),
        }
    return out


def percent_reduction_vs_uniform(rows: list[dict], aous: list[str], learned: str = "learned_cost_ot") -> dict:
    by = {(r["aou"], r["method"]): r for r in rows}
    red = np.array([
        (by[(a, "uniform_cost_ot")]["wasserstein_m"] - by[(a, learned)]["wasserstein_m"])
        / by[(a, "uniform_cost_ot")]["wasserstein_m"] * 100
        for a in aous
    ])
    return {
        "mean": float(red.mean()), "median": float(np.median(red)),
        "q25": float(np.percentile(red, 25)), "q75": float(np.percentile(red, 75)),
        "min": float(red.min()), "max": float(red.max()),
        "n_positive": int((red > 0).sum()), "n": len(red),
        "per_species": {a: float(v) for a, v in zip(aous, red)},
    }


def shift_split(
    rows: list[dict], aous: list[str], shifts: dict[str, float], learned: str = "learned_cost_ot",
) -> dict:
    """Learned-vs-persistence wins within expanding, contracting and stable species."""
    by = {(r["aou"], r["method"]): r for r in rows}

    def group(members: list[str]) -> dict:
        wins = sum(by[(a, learned)]["wasserstein_m"] < by[(a, "persistence")]["wasserstein_m"] for a in members)
        return {"n": len(members), "species": members, "wins_vs_persistence": int(wins)}

    return {
        "threshold": SHIFT_THRESHOLD,
        "expanding": group([a for a in aous if shifts[a] > SHIFT_THRESHOLD]),
        "contracting": group([a for a in aous if shifts[a] < -SHIFT_THRESHOLD]),
        "stable": group([a for a in aous if abs(shifts[a]) <= SHIFT_THRESHOLD]),
    }


def cost_to_expand(coupling: np.ndarray, C: np.ndarray, mu_t0: np.ndarray, mu_t1_obs: np.ndarray) -> dict:
    """Mean cost of mass arriving in cells where the observed distribution grew."""
    grew = mu_t1_obs > mu_t0
    mass = coupling[:, grew].sum()
    cost = (coupling[:, grew] * C[:, grew]).sum()
    return {
        "n_expansion_cells": int(grew.sum()),
        "mass_into_expansion_cells": float(mass),
        "cost_to_expand": float(cost / mass) if mass > 0 else float("nan"),
    }


def northern_masses(arrays: dict[str, np.ndarray], mu_learned: np.ndarray, support: Support) -> dict:
    grid = support.grid
    nh = {
        "t0": northern_half_mass(arrays["mu_t0"], grid.n_x, grid.n_y, support.idx),
        "obs": northern_half_mass(arrays["mu_t1_obs"], grid.n_x, grid.n_y, support.idx),
        "sdm": northern_half_mass(arrays["mu_t1_sdm"], grid.n_x, grid.n_y, support.idx),
        "learned": northern_half_mass(mu_learned, grid.n_x, grid.n_y, support.idx),
    }
    nh["shift"] = nh["obs"] - nh["t0"]
    return nh


def primary_design(
    routes: dict[RouteKey, tuple[float, float]], grid: AnalysisGrid, idx: np.ndarray,
    baseline_dir: Path, late_bio: dict[str, np.ndarray],
) -> tuple[list[RouteKey], np.ndarray, np.ndarray]:
    """SDM design matrices: baseline climate sampled at routes, projected onto the period-2 climate of the support."""
    route_keys = eastern_routes(routes, grid)
    lons = np.array([routes[k][0] for k in route_keys])
    lats = np.array([routes[k][1] for k in route_keys])
    route_bio = np.stack(
        [point_sample_raster(load_raster(baseline_dir / f"wc2.1_10m_{v}.tif"), lons, lats) for v in BIO_VARS], axis=1,
    )
    early_cells = np.stack(
        [extract_cell_values(baseline_dir / f"wc2.1_10m_{v}.tif", grid)[idx] for v in BIO_VARS], axis=1,
    )
    mean, std = early_cells.mean(axis=0), early_cells.std(axis=0)
    late_cells = np.stack([late_bio[v][idx] for v in BIO_VARS], axis=1)
    return route_keys, build_design_matrix(route_bio, mean, std), build_design_matrix(late_cells, mean, std)


def eastern_routes(routes: dict[RouteKey, tuple[float, float]], grid: AnalysisGrid) -> list[RouteKey]:
    keys = list(routes)
    ix, _ = grid.cell_indices(np.array([routes[k][0] for k in keys]), np.array([routes[k][1] for k in keys]))
    return [k for k, inside in zip(keys, ix >= 0) if inside]


def train_species_sdms(
    counts: dict[str, dict[RouteYearKey, int]],
    routes: dict[RouteKey, tuple[float, float]],
    surveyed: set[RouteYearKey],
    years_early: range,
    route_keys: list[RouteKey],
    design_route: np.ndarray,
    design_target: np.ndarray,
    kind: str = "logistic",
) -> dict[str, tuple[np.ndarray, float]]:
    """Fit one SDM per species on early-window route presence and project onto design_target.

    Returns {aou: (normalized suitability on the target cells, hold-out AUC)}.
    """
    pos = {k: i for i, k in enumerate(route_keys)}
    out = {}
    for aou, species_counts in counts.items():
        early = route_window_values(routes, species_counts, surveyed, years_early)
        keys = [k for k in route_keys if k in early]
        y = np.array([1 if early[k] > 0 else 0 for k in keys], dtype=int)
        if len(np.unique(y)) < 2 or len(y) < 20:
            continue
        X = design_route[[pos[k] for k in keys]]
        auc = holdout_auc(X, y, kind)
        out[aou] = (project_sdm(fit_sdm(X, y, kind), design_target), auc)
    return out
