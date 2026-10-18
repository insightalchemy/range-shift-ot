"""Build the pairwise cost features (d_geo, d_climate, d_elev) on the 50 km analysis support."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.features import BIO_VARS, extract_cell_values, pairwise_features, window_bioclim, zscore
from rangeshift.grid import assert_aligned
from rangeshift.pipeline import CELL_KM, LATE_YEARS, add_path_args, load_support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    args = parser.parse_args()
    support = load_support(args.processed)
    grid, idx = support.grid, support.idx

    out_dir = args.processed / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    late = window_bioclim(grid, LATE_YEARS, args.data / "worldclim_monthly")
    np.savez(out_dir / "late_bioclim.npz", **late)

    baseline = args.data / "worldclim"
    early_raw = [extract_cell_values(baseline / f"wc2.1_10m_{v}.tif", grid)[idx] for v in BIO_VARS]
    bio_z = np.stack([zscore(x) for x in early_raw], axis=1)
    elev_z = zscore(extract_cell_values(baseline / "wc2.1_10m_elev.tif", grid)[idx])

    # Period-2 climate is standardized with period-1 statistics, as for the SDM.
    clim_t0, clim_t1 = [], []
    for x, v in zip(early_raw, BIO_VARS):
        mean, std = x.mean(), x.std()
        clim_t0.append(((x - mean) / std).astype(np.float32))
        clim_t1.append(((late[v][idx] - mean) / std).astype(np.float32))

    features = pairwise_features(support.coords_m, bio_z, elev_z, CELL_KM)
    assert_aligned(idx, *[f.numpy() for f in features.values()])
    np.savez(
        out_dir / "features.npz",
        support_idx=idx,
        d_geo=features["d_geo"].numpy(), d_climate=features["d_climate"].numpy(), d_elev=features["d_elev"].numpy(),
        bio_z=bio_z, elev_z=elev_z,
        clim_t0=np.stack(clim_t0, axis=1), clim_t1=np.stack(clim_t1, axis=1),
    )
    print(f"{len(idx)} cells; d_climate max {features['d_climate'].max():.2f}, d_elev max {features['d_elev'].max():.2f}")


if __name__ == "__main__":
    main()
