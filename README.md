# Residential Population — auditing Meta's crisis population baseline against WorldPop

Meta publishes a "Population During Crisis" baseline used by humanitarian responders to
judge where people are before and during a disaster. This project asks whether that
baseline is **biased against deprived places**, using WorldPop as the reference and GRDI as
the deprivation measure, across **18 cities in 8 countries**.

**Headline finding.** Meta's baseline is population-representative but **geographically
censored**, and the censoring tracks deprivation. Among the tiles Meta publishes, it
allocates people essentially correctly (no bias). But conditional on population and
settlement type, a tile one standard deviation more deprived than its city average has
about **one-eighth the odds of being published at all** — the same sign in 14 of 14
estimable cities.

Full method, results, robustness and limitations: **[`analysis/README.md`](analysis/README.md)**.

---

## What is and isn't in this repository

**Code only.** The raw inputs (Meta PDC extracts, WorldPop rasters, GRDI), the harmonised
grids, and the generated tables and figures are all gitignored — the inputs because they are
large and carry their own licence terms, the outputs because they are regenerable from them.

So this repository documents and reproduces the analysis, but does not distribute its
results. To rerun it you need the source data described under [Data](#data), then
`./run --all` followed by the analysis scripts below.

## Repository layout

```
pipeline/    two steps that build the grids (01 harmonise, 01b independent grid)
analysis/    7 Python analysis scripts + 2 R figure scripts, plus the write-up
config/      regions.json — per-city paths, clip shapes, baseline hours
data_prep/   one-off helpers (build Meta baseline medians, extract boundaries)
data/        raw inputs, harmonised grids, Meta baselines  (gitignored)
outputs/     tables   -> outputs/analysis/                 (gitignored)
figure/      figures  -> figure/analysis/                  (gitignored)
```

## Study sample

| Country | Code | Cities | Meta hour (Pacific) |
|---------|------|--------|---------------------|
| Philippines | PHL | Cagayan de Oro, Davao City, Zamboanga City, General Santos | 8 |
| Kenya | KEN | Nairobi, Mombasa | 16 |
| Mexico | MEX | Mexico City, Puebla, León | 0 |
| Indonesia | IDN | Medan, Banda Aceh | 8 |
| Sri Lanka | LKA | Colombo, Kandy | 8 |
| Colombia | COL | Barranquilla, Cartagena | 0 |
| Ecuador | ECU | Cuenca, Guayaquil | 0 |
| South Africa | ZAF | Cape Town | 16 |

Hour is chosen so the snapshot sits near evening locally. Baseline method is `n_baseline`.
Nakuru and Garden Route stay in `regions.json` with `in_sample: false` — Meta coverage there
is so sparse (11–13%) that they would dominate any pooled model. **Kisumu is excluded for a
different reason**: 303 of its 434 tiles never appear in any timestamp of the Kenya Floods
extract despite holding ~1,600 residents each, so their absence marks the edge of the event
AOI rather than a Meta coverage decision. See `analysis/README.md` §2.

## Prerequisites

- **Python 3.9+** — `conda activate geo_env_LLM`, then `pip install -r requirements.txt`
- Main dependencies: geopandas, rasterio, rasterstats, pandas, numpy, scipy, statsmodels,
  libpysal, esda, matplotlib, pyarrow

**R 4.0+** for figures: `install.packages(c("ggplot2","sf","dplyr","tidyr","patchwork","forcats","readr","scales","ragg","ggrepel"))`.
Figures use the `rcartocolor::ArmyRose` palette; see `analysis/R/theme_armyrose.R`.

## Running it

### 1. Build the grids

`./run` is a thin wrapper around `pipeline/run_all.sh`. It produces two artefacts per city:

| Artefact | What it is |
|---|---|
| `01/harmonised_meta_worldpop.gpkg` | Meta + WorldPop + GRDI on the Meta quadkey grid |
| `01b_coverage/independent_grid.gpkg` | the **independent** city grid, including the tiles Meta never published |

The second is the one that matters. Step 01 starts from published Meta tiles, so a
suppressed tile cannot appear in it. Step 01b rebuilds the grid from the clip polygon so
suppressed tiles are present and flagged — which is what makes the coverage analysis
possible at all.

```bash
./run --region PHL          # all selected cities in a country
./run --all                 # every selected city in every country
./run --one KEN_Nairobi     # a single city
./run --region KEN --ref-hour 8
```

Both artefacts are already built for all cities, so you can skip straight to step 2.

### 2. Run the analysis

```bash
python analysis/01_build_panel.py             # pooled tile panel
python analysis/02_selection_models.py        # extensive margin + robustness ladder
python analysis/03_two_margin_decomposition.py  # reconciles the null
python analysis/04_operational_consequence.py   # blind spots, clustering, targeting
python analysis/05_city_models.py             # per-city models (feeds F0b, F1b)
python analysis/06_refhour_robustness.py      # baseline-hour check, and why not RWI
python analysis/07_censoring_bounds.py        # bounds, imputation, Tobit, placebo
python analysis/09_sensitivity.py             # floor, jackknife, MAUP, functional form
python analysis/12_aoi_check.py               # suppression vs event-AOI edge
Rscript analysis/10_figures_main.R            # F0, F0b, F1, F1b, F2-F5
Rscript analysis/11_figures_robustness.R      # F6-F9
```

Tables land in `outputs/analysis/`, figures in `figure/analysis/`. The whole chain runs in
about a minute from the cached grids.

## Data

Poverty is **GRDI v1.10** (`data/raw/povmap-grdi-v1-10.tif`, higher = more deprived).
WorldPop country rasters live in `data/raw/worldpop/`. Meta baselines are built per country
per hour into `data/baselines/{COUNTRY}/fb_baseline_median_h{00|08|16}.gpkg`. Source files
and the external `data_root` are documented in [`data/README.md`](data/README.md) and
[`config/README.md`](config/README.md).

Note: **Meta RWI is deliberately not used** as an alternative deprivation measure. It is
estimated from Facebook data and is therefore missing in precisely the tiles whose
publication we are modelling (95.4% coverage where Meta published, 47.3% where it did not).
See `analysis/README.md` §5.6.

## Scope

Findings are **associational**. There is no exogenous variation in deprivation, and GRDI
shares unobserved causes with connectivity, so no causal claim is made anywhere in this
repository. A prior `03e_causal` step was removed for that reason.

## License

No `LICENSE` file yet. Add one before public release or redistribution.
