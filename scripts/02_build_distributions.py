"""Select species, build the coverage mask and mu_t0 / mu_t1_obs on the analysis grid, and split train / test."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rangeshift.bbs import (
    apply_coverage_mask,
    build_distribution,
    consistent_coverage_mask,
    grid_raw_values,
    load_routes,
    load_species_counts,
    load_species_names,
    load_surveyed_route_years,
    route_cells,
    route_window_values,
)
from rangeshift.grid import build_eastern_us_grid, support_indices
from rangeshift.pipeline import (
    CURATED,
    CURATED_TEST,
    EARLY_YEARS,
    LATE_YEARS,
    add_path_args,
    make_split,
    read_json,
    slugify,
    write_json,
)

MIN_OCCUPIED_CELLS = 150
MAX_SPECIES = 100
SPLIT_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_path_args(parser)
    parser.add_argument("--resplit", action="store_true", help="regenerate results/split.json even if it exists")
    args = parser.parse_args()
    bbs_dir = args.data / "bbs"

    grid = build_eastern_us_grid()
    routes = load_routes(bbs_dir)
    surveyed = load_surveyed_route_years(bbs_dir)
    mask = consistent_coverage_mask(routes, surveyed, grid, EARLY_YEARS, LATE_YEARS)
    support_idx = support_indices(mask)
    print(f"grid {grid.n_x}x{grid.n_y}, support {len(support_idx)} cells")

    args.processed.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.processed / "grid_meta.npz",
        x_min=grid.x_min, y_min=grid.y_min, cell_size=grid.cell_size,
        n_x=grid.n_x, n_y=grid.n_y, mask=mask, cell_centers=grid.cell_centers(),
    )

    names = load_species_names(bbs_dir)
    counts = load_species_counts(bbs_dir)
    cells = route_cells(routes, grid)

    def masked_raw(aou: str, years: range) -> np.ndarray:
        values = route_window_values(routes, counts[aou], surveyed, years)
        return apply_coverage_mask(grid_raw_values(values, cells, grid.n_cells), mask)

    rows = []
    for aou in counts:
        early = int((masked_raw(aou, EARLY_YEARS)[support_idx] > 0).sum())
        late = int((masked_raw(aou, LATE_YEARS)[support_idx] > 0).sum())
        name = names.get(aou.zfill(5), f"AOU {aou}")
        rows.append({
            "aou": aou, "name": name, "slug": slugify(name),
            "n_occupied_early": early, "n_occupied_late": late, "min_occupied": min(early, late),
        })
    rows.sort(key=lambda r: (-r["min_occupied"], r["aou"]))

    qualifying = [r for r in rows if r["min_occupied"] >= MIN_OCCUPIED_CELLS]
    print(f"{len(qualifying)} species with >= {MIN_OCCUPIED_CELLS} occupied cells in both windows")
    selected = qualifying[:MAX_SPECIES]
    by_aou = {r["aou"]: r for r in rows}
    forced = [a for a in CURATED_TEST if a not in {r["aou"] for r in selected}]
    selected += [by_aou[a] for a in forced]
    missing = [a for a in CURATED if a not in {r["aou"] for r in selected}]
    assert not missing, f"curated species not selected: {missing}"

    write_json(args.results / "inclusion.json", {
        "criterion": f"at least {MIN_OCCUPIED_CELLS} occupied cells in both windows, top {MAX_SPECIES} by occupied cells",
        "force_included": forced,
        "species": selected,
    })

    out_dir = args.processed / "distributions"
    out_dir.mkdir(exist_ok=True)
    for r in selected:
        mu_t0 = build_distribution(masked_raw(r["aou"], EARLY_YEARS))[support_idx]
        mu_t1_obs = build_distribution(masked_raw(r["aou"], LATE_YEARS))[support_idx]
        np.savez(out_dir / f"{r['aou']}_{r['slug']}.npz", mu_t0=mu_t0, mu_t1_obs=mu_t1_obs)
    print(f"wrote {len(selected)} species to {out_dir}")

    split_path = args.results / "split.json"
    if split_path.exists() and not args.resplit:
        split = read_json(split_path)
        assert set(split["train"]) | set(split["test"]) == {r["aou"] for r in selected}, "split.json does not match the selected species"
        print(f"kept existing {split_path}: {len(split['train'])} train / {len(split['test'])} test")
    else:
        split = make_split(selected, CURATED_TEST, SPLIT_SEED)
        write_json(split_path, split)
        print(f"wrote {split_path}: {len(split['train'])} train / {len(split['test'])} test")


if __name__ == "__main__":
    main()
