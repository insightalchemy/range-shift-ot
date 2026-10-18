from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from rangeshift.cost import pairwise_abs_difference, pairwise_euclidean
from rangeshift.grid import AnalysisGrid, coarsen, load_raster, reproject_to_grid

BIO_VARS = ["bio_1", "bio_4", "bio_12", "bio_15"]

# Ocean pixels are all-NaN, which makes the monthly standard deviations warn.
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0")


def extract_cell_values(path: Path, grid: AnalysisGrid) -> np.ndarray:
    """Mean of a raster over each analysis cell (NaN where the raster has no data)."""
    return coarsen(reproject_to_grid(load_raster(path), grid.crs), grid)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if np.isnan(values).any():
        raise ValueError("zscore: input contains NaN")
    mean, std = values.mean(), values.std()
    if std == 0:
        raise ValueError("zscore: zero variance")
    return ((values - mean) / std).astype(np.float32)


def monthly_climatology(variable: str, years: list[int], monthly_dir: Path) -> list[xr.DataArray]:
    """Mean of each calendar month across years, for tmin, tmax or prec."""
    climatology = []
    for month in range(1, 13):
        rasters = [
            load_raster(monthly_dir / f"wc2.1_cruts4.09_10m_{variable}_{year}-{month:02d}.tif")
            for year in years
        ]
        climatology.append(xr.concat(rasters, dim="year").mean(dim="year"))
    return climatology


def bioclim_from_monthly(
    tmin: list[xr.DataArray], tmax: list[xr.DataArray], prec: list[xr.DataArray],
) -> dict[str, xr.DataArray]:
    """BIO1, BIO4, BIO12 and BIO15 from 12 monthly normals."""
    tmean = xr.concat([(tmin[m] + tmax[m]) / 2.0 for m in range(12)], dim="month")
    prec_stack = xr.concat(prec, dim="month")
    return {
        "bio_1": tmean.mean(dim="month"),
        "bio_4": tmean.std(dim="month") * 100.0,
        "bio_12": prec_stack.sum(dim="month"),
        "bio_15": (prec_stack.std(dim="month") / prec_stack.mean(dim="month")) * 100.0,
    }


def window_bioclim(grid: AnalysisGrid, years: range | list[int], monthly_dir: Path) -> dict[str, np.ndarray]:
    """Bioclim variables for a window of years on every cell of the grid."""
    years = list(years)
    bio = bioclim_from_monthly(
        monthly_climatology("tmin", years, monthly_dir),
        monthly_climatology("tmax", years, monthly_dir),
        monthly_climatology("prec", years, monthly_dir),
    )
    return {name: coarsen(reproject_to_grid(raster, grid.crs), grid) for name, raster in bio.items()}


def pairwise_features(
    coords_m: np.ndarray, bio_z: np.ndarray, elev_z: np.ndarray, cell_km: float,
) -> dict[str, torch.Tensor]:
    """d_geo in grid-cell units, d_climate (Euclidean in standardized bioclim), d_elev (absolute difference)."""
    return {
        "d_geo": pairwise_euclidean(coords_m / 1000.0) / cell_km,
        "d_climate": pairwise_euclidean(bio_z),
        "d_elev": pairwise_abs_difference(elev_z),
    }
