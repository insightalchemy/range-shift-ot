# Learning Ecological Cost Surfaces of Climate-Driven Range Shifts via Inverse Optimal Transport

Ishan Nagpal and Ziheng Sun

Conservation planning relies on resistance surfaces that are usually assumed from expert opinion or geographic distance. This repository learns an interpretable resistance surface from observed bird range shifts. For each species, an unbalanced optimal-transport model moves the early-period distribution (Breeding Bird Survey, 1997-2002) toward a climate-feasible distribution from a species distribution model, under a cost that is a weighted sum of geographic distance, climate dissimilarity and elevation difference. The weights are fit by backpropagating a Sinkhorn-divergence loss against the later observed distribution (2017-2022) through the Sinkhorn iterations, and are evaluated on held-out species against persistence, climate-only, climate-velocity and uniform-cost baselines.

## Setup

Python 3.11.

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Everything runs on CPU. The pairwise matrices are float32 on a 1,097-cell support (50 km grid, EPSG:5070), and the longest fits peak at about 3.7 GB of RAM.

## Data

- Breeding Bird Survey, 2026 release: USGS ScienceBase item `6a0b0b0ab66b0188da36aedd` (`Routes.csv`, `Weather.csv`, `States.zip`, `SpeciesList.csv`).
- WorldClim 2.1 at 10 arc-minutes: the 1970-2000 bioclimatic variables and elevation, plus the CRU-TS 4.09 monthly historical series (1995-2000, 2006-2011, 2017-2022) used to compute period climates.
- eBird Status and Trends (optional): only used to cross-check the BBS distributions for five species. It needs a key in the `EBIRD_API_KEY` environment variable and is not read by any other step.

`scripts/01_download.py` fetches all of the above into `data/` (about 1 GB). Pass `--ebird` to also fetch the eBird layers.

## Reproduce

Run from the repository root. Paths default to `data/`, `processed/` and `results/`; every script takes `--data`, `--processed` and `--results`. Runtimes are rough, on an 8 GB laptop (M3 Air, CPU).

| Step | Script | Runtime |
| --- | --- | --- |
| Download data | `python scripts/01_download.py` | 10-20 min |
| Species selection, coverage mask, distributions, train/test split | `python scripts/02_build_distributions.py` | 2 min |
| Pairwise features and period-2 climate | `python scripts/03_build_features.py` | 1 min |
| Species distribution models | `python scripts/04_fit_sdm.py` | 1 min |
| Fit the cost weights | `python scripts/05_fit_cost.py --config unbiased` | 3.7 h |
| | `python scripts/05_fit_cost.py --config curated` | 1 h |
| Evaluate on held-out species | `python scripts/06_evaluate.py --config unbiased` | 5 min |
| | `python scripts/06_evaluate.py --config curated [--ebird-crossval]` | 1 min |
| Temporal transfer (1995-2000, 2006-2011, 2017-2022) | `python scripts/07_temporal_transfer.py` | 6 h |
| Sensitivity to SDM and resolution | `python scripts/08_sensitivity.py` | 1 h |
| Ratio identifiability profile | `python scripts/09_ratio_profile.py` | 3 h |
| Figures | `python scripts/10_figures.py --config unbiased` | 1 min |
| Website data | `python scripts/export_web.py` | 1 min |

The `unbiased` configuration uses the 101 species selected only by data sufficiency (at least 150 occupied cells in both windows, plus Loggerhead Shrike, which is held out), split 68 train / 33 test with seed 42. The `curated` configuration uses 13 species chosen from the literature, split 8 train / 5 test. Both use eps = 0.05 and tau = 5. `results/split.json` and `results/inclusion.json` are the lists used in the paper; `02_build_distributions.py` keeps an existing `split.json` and regenerates it only with `--resplit`.

Tests: `pytest` (under a minute).

## Results

Learned cost against each baseline on the 33 held-out species (Wasserstein distance to the observed 2017-2022 distribution; one-sided sign test that the learned cost is better).

| Baseline | Learned cost wins | Sign-test p | Mean Wasserstein of baseline (km) |
| --- | --- | --- | --- |
| Persistence | 7 / 33 | 0.9998 | 106.3 |
| Climate only (SDM) | 33 / 33 | 1.2e-10 | 257.9 |
| Climate velocity | 30 / 33 | 7.0e-7 | 228.9 |
| Uniform-cost OT | 33 / 33 | 1.2e-10 | 167.8 |

For reference, the learned cost has a mean Wasserstein distance of 140.4 km. It improves on every model-based baseline but not on persistence, which beats it for 26 of the 33 species.

## License

MIT. See `LICENSE`.
