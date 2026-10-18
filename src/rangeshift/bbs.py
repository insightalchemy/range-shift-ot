from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

import numpy as np

from rangeshift.grid import AnalysisGrid

RouteKey = tuple[str, str, str]
RouteYearKey = tuple[str, str, str, int]


def load_routes(bbs_dir: Path) -> dict[RouteKey, tuple[float, float]]:
    """US route locations as (lon, lat), keyed by (country, state, route)."""
    routes: dict[RouteKey, tuple[float, float]] = {}
    with open(bbs_dir / "Routes.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            if row["CountryNum"].strip() != "840":
                continue
            key = (row["CountryNum"].strip(), row["StateNum"].strip(), row["Route"].strip())
            routes[key] = (float(row["Longitude"]), float(row["Latitude"]))
    return routes


def load_surveyed_route_years(bbs_dir: Path) -> set[RouteYearKey]:
    """Route-years with RunType == 1 in Weather.csv; a RunType 1 row wins over duplicates."""
    best_runtype: dict[RouteYearKey, str] = {}
    with open(bbs_dir / "Weather.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            key = (
                row["CountryNum"].strip(), row["StateNum"].strip(),
                row["Route"].strip(), int(row["Year"]),
            )
            rt = row["RunType"].strip()
            if best_runtype.get(key) != "1":
                best_runtype[key] = rt
    return {key for key, rt in best_runtype.items() if rt == "1"}


def load_species_counts(
    bbs_dir: Path, aou_codes: set[str] | None = None,
) -> dict[str, dict[RouteYearKey, int]]:
    """SpeciesTotal per route-year for the requested AOU codes, or all codes if None."""
    counts: dict[str, dict[RouteYearKey, int]] = {a: {} for a in aou_codes} if aou_codes else {}
    with zipfile.ZipFile(bbs_dir / "States.zip") as z:
        for name in z.namelist():
            if not name.endswith(".csv"):
                continue
            with z.open(name) as raw:
                for row in csv.DictReader(io.TextIOWrapper(raw, encoding="latin-1")):
                    aou = row["AOU"].strip()
                    if aou_codes is None:
                        bucket = counts.setdefault(aou, {})
                    elif aou in counts:
                        bucket = counts[aou]
                    else:
                        continue
                    key = (
                        row["CountryNum"].strip(), row["StateNum"].strip(),
                        row["Route"].strip(), int(row["Year"]),
                    )
                    bucket[key] = int(row["SpeciesTotal"])
    return counts


def load_species_names(bbs_dir: Path) -> dict[str, str]:
    names = {}
    with open(bbs_dir / "SpeciesList.csv", encoding="latin-1") as f:
        for row in csv.DictReader(f):
            names[row["AOU"].strip().zfill(5)] = row["English_Common_Name"].strip()
    return names


def route_window_values(
    routes: dict[RouteKey, tuple[float, float]],
    counts: dict[RouteYearKey, int],
    surveyed: set[RouteYearKey],
    years: range,
) -> dict[RouteKey, float]:
    """Mean count per route over surveyed years in the window; unsurveyed years are skipped, not zero."""
    result: dict[RouteKey, float] = {}
    for route_key in routes:
        values = [
            counts.get((*route_key, year), 0)
            for year in years
            if (*route_key, year) in surveyed
        ]
        if values:
            result[route_key] = float(np.mean(values))
    return result


def route_cells(routes: dict[RouteKey, tuple[float, float]], grid: AnalysisGrid) -> dict[RouteKey, int]:
    """Flat grid-cell index of every route (-1 outside the grid)."""
    keys = list(routes)
    lons = np.array([routes[k][0] for k in keys])
    lats = np.array([routes[k][1] for k in keys])
    ix, iy = grid.cell_indices(lons, lats)
    return {k: int(f) for k, f in zip(keys, grid.flat_index(ix, iy))}


def consistent_coverage_mask(
    routes: dict[RouteKey, tuple[float, float]],
    surveyed: set[RouteYearKey],
    grid: AnalysisGrid,
    early_years: range,
    late_years: range,
) -> np.ndarray:
    """Cells with at least one surveyed route in both windows."""
    cells = route_cells(routes, grid)
    covered = []
    for years in (early_years, late_years):
        values = route_window_values(routes, {}, surveyed, years)
        covered.append({cells[k] for k in values if cells[k] >= 0})
    mask = np.zeros(grid.n_cells, dtype=bool)
    for flat in covered[0] & covered[1]:
        mask[flat] = True
    return mask


def apply_coverage_mask(raw: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, raw, 0.0).astype(np.float32)


def grid_raw_values(
    route_values: dict[RouteKey, float], cells: dict[RouteKey, int], n_cells: int,
) -> np.ndarray:
    """Mean of route values within each cell."""
    by_cell: dict[int, list[float]] = {}
    for route_key, value in route_values.items():
        flat = cells.get(route_key, -1)
        if flat >= 0:
            by_cell.setdefault(flat, []).append(value)
    raw = np.zeros(n_cells, dtype=np.float32)
    for flat, values in by_cell.items():
        raw[flat] = np.mean(values)
    return raw


def build_distribution(raw: np.ndarray) -> np.ndarray:
    total = float(raw.sum())
    if total <= 0:
        raise ValueError("raw cell values sum to zero")
    return (raw / total).astype(np.float32)
