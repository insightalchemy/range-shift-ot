from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr
from pyproj import Transformer

LON_MIN, LON_MAX = -100.0, -66.5
LAT_MIN, LAT_MAX = 24.5, 49.5
CELL_SIZE_M = 50_000.0
CRS = "EPSG:5070"


@dataclass(frozen=True)
class AnalysisGrid:
    crs: str
    x_min: float
    y_min: float
    cell_size: float
    n_x: int
    n_y: int

    @property
    def n_cells(self) -> int:
        return self.n_x * self.n_y

    def cell_indices(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Map lon/lat to (ix, iy); points outside the grid get -1."""
        transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        x, y = transformer.transform(np.asarray(lon), np.asarray(lat))
        x, y = np.asarray(x), np.asarray(y)
        ix = np.floor((x - self.x_min) / self.cell_size).astype(int)
        iy = np.floor((y - self.y_min) / self.cell_size).astype(int)
        outside = (ix < 0) | (ix >= self.n_x) | (iy < 0) | (iy >= self.n_y)
        return np.where(outside, -1, ix), np.where(outside, -1, iy)

    def flat_index(self, ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
        ix, iy = np.asarray(ix), np.asarray(iy)
        return np.where((ix < 0) | (iy < 0), -1, iy * self.n_x + ix)

    def cell_centers(self) -> np.ndarray:
        ix, iy = np.meshgrid(np.arange(self.n_x), np.arange(self.n_y), indexing="xy")
        x = self.x_min + (ix.ravel() + 0.5) * self.cell_size
        y = self.y_min + (iy.ravel() + 0.5) * self.cell_size
        return np.stack([x, y], axis=1).astype(np.float32)


def build_eastern_us_grid(cell_size_m: float = CELL_SIZE_M) -> AnalysisGrid:
    # Albers is conic, so transform the bbox boundary rather than just its corners.
    n = 200
    lons = np.concatenate([
        np.linspace(LON_MIN, LON_MAX, n), np.linspace(LON_MIN, LON_MAX, n),
        np.full(n, LON_MIN), np.full(n, LON_MAX),
    ])
    lats = np.concatenate([
        np.full(n, LAT_MIN), np.full(n, LAT_MAX),
        np.linspace(LAT_MIN, LAT_MAX, n), np.linspace(LAT_MIN, LAT_MAX, n),
    ])
    x, y = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lons, lats)
    x_min, y_min = float(np.min(x)), float(np.min(y))
    n_x = int(np.ceil((np.max(x) - x_min) / cell_size_m))
    n_y = int(np.ceil((np.max(y) - y_min) / cell_size_m))
    return AnalysisGrid(CRS, x_min, y_min, cell_size_m, n_x, n_y)


def grid_from_meta(path: Path) -> tuple[AnalysisGrid, np.ndarray]:
    """Load the grid and consistent-coverage mask saved by the distribution step."""
    meta = np.load(path)
    grid = AnalysisGrid(
        CRS, float(meta["x_min"]), float(meta["y_min"]),
        float(meta["cell_size"]), int(meta["n_x"]), int(meta["n_y"]),
    )
    return grid, meta["mask"].astype(bool)


def load_raster(path: Path) -> xr.DataArray:
    # masked=True turns the nodata sentinel into NaN so it is not averaged into cells.
    raster = rioxarray.open_rasterio(path, masked=True)
    if "band" in raster.dims and raster.sizes["band"] == 1:
        raster = raster.squeeze("band", drop=True)
    return raster


def point_sample_raster(raster: xr.DataArray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Nearest-pixel sample of an EPSG:4326 raster."""
    points = raster.sel(
        x=xr.DataArray(np.asarray(lon), dims="points"),
        y=xr.DataArray(np.asarray(lat), dims="points"),
        method="nearest",
    )
    return np.asarray(points.values, dtype=np.float32)


def reproject_to_grid(raster: xr.DataArray, target_crs: str = CRS) -> xr.DataArray:
    return raster.rio.reproject(target_crs)


def coarsen(raster: xr.DataArray, grid: AnalysisGrid) -> np.ndarray:
    """Mean of the raster pixels falling in each grid cell; NaN where none."""
    x = raster.x.values
    y = raster.y.values
    values = np.asarray(raster.values, dtype=np.float64)
    if values.ndim == 3:
        values = values[0]
    xx, yy = np.meshgrid(x, y)

    ix = np.floor((xx - grid.x_min) / grid.cell_size).astype(int)
    iy = np.floor((yy - grid.y_min) / grid.cell_size).astype(int)
    valid = (ix >= 0) & (ix < grid.n_x) & (iy >= 0) & (iy < grid.n_y) & np.isfinite(values)

    flat_idx = iy[valid] * grid.n_x + ix[valid]
    sums = np.bincount(flat_idx, weights=values[valid], minlength=grid.n_cells)
    counts = np.bincount(flat_idx, minlength=grid.n_cells)

    out = np.full(grid.n_cells, np.nan, dtype=np.float32)
    covered = counts > 0
    out[covered] = (sums[covered] / counts[covered]).astype(np.float32)
    return out


def support_indices(mask: np.ndarray) -> np.ndarray:
    return np.flatnonzero(mask)


def assert_aligned(support_idx: np.ndarray, *arrays: np.ndarray) -> None:
    """Check that 1D arrays have length n and 2D arrays shape (n, n) for the compacted support."""
    n = len(support_idx)
    for i, arr in enumerate(arrays):
        if arr.ndim == 1:
            assert arr.shape[0] == n, f"array {i}: expected length {n}, got {arr.shape[0]}"
        elif arr.ndim == 2:
            assert arr.shape == (n, n), f"array {i}: expected shape {(n, n)}, got {arr.shape}"
        else:
            raise ValueError(f"array {i}: expected 1D or 2D, got ndim={arr.ndim}")
