# Data

Sources used by this project, and where to get them. Large rasters are not committed.

```
data/
  raw/          # downloads and source files
  processed/    # pipeline GPKGs (regenerate with ./run)
  baselines/    # built Meta median GPKGs (from data_prep/build_fb_baseline_median.py)
```

## In the pipeline now

| Dataset           | What we use                                                                           | Local path                                                                        | Source                                                                                                                                                                                                                                     |
| ----------------- | ------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| WorldPop Global 2 | Constrained 100 m population (`*_pop_*_CN_100m_R2025A_v1.tif`). Pipeline: KEN, PHL, MEX, IDN, LKA, COL, ECU, ZAF. Extra rasters on disk are listed in [`raw/worldpop_log.csv`](raw/worldpop_log.csv) | `data/raw/worldpop/` (gitignored)                                                 | [Hub listing (100 m, R2025A v1)](https://hub.worldpop.org/geodata/listing?id=135) · [direct files](https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/) · DOI [10.5258/SOTON/WP00839](https://doi.org/10.5258/SOTON/WP00839) |
| Meta PDC          | Facebook Population During Crisis baseline (quadkey counts) | Built GPKGs in `data/baselines/{COUNTRY}/`; raw event zips under `data_root` / `Meta_Event/Pop` | [Meta Data for Good — Population maps](https://dataforgood.facebook.com/dfg/tools/facebook-population-maps) |
| GRDI v1.10        | Poverty / deprivation (default). Higher = more deprived                               | `data/raw/povmap-grdi-v1-10.tif` (gitignored)                                     | [GRDI Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TEM9JH)                                                                                                                                          |
| Meta RWI          | Optional poverty layer (`--poverty-source rwi`)                                       | `data_root` → `Meta_Event/RWI/relative-wealth-index-april-2021/`                  | [Meta Relative Wealth Index](https://dataforgood.facebook.com/dfg/tools/relative-wealth-index)                                                                                                                                             |
| City boundaries   | Clip polygons for each study city                                                     | `data/raw/boundaries/{Ken,Phi,Mex}/`                                              | Local extracts; optional [OSM](https://www.openstreetmap.org/) or [geoBoundaries](https://www.geoboundaries.org/) via `--clip-source`                                                                                                      |


WorldPop file URL pattern:

`https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/{YEAR}/{ISO3}/v1/100m/constrained/{iso}_pop_{YEAR}_CN_100m_R2025A_v1.tif`

## Also in this folder

| Path | Notes |
|------|--------|
| `raw/worldpop/` | Country rasters used by the pipeline (see [`raw/worldpop_log.csv`](raw/worldpop_log.csv)) |
| [`raw/worldpop_log.csv`](raw/worldpop_log.csv) | Log of all PDC events and WorldPop rasters. `meta_event_id` is Meta’s crisis ID (same on every CSV in that extract; different events have different IDs). `year` is the WorldPop raster year; `worldpop_present` is yes/no for `data/raw/worldpop/`; `in_pipeline` is whether the region is already in `config/regions.json` |
| `raw/ghsl/` | **GHS-SMOD** 2020 (R2023A, 30 arc-sec raster). Degree of Urbanisation: urban / suburban / rural per pixel. More relevant for stratifying quadkeys. [JRC download](https://human-settlement.emergency.copernicus.eu/download.php?ds=smod). Not in the main pipeline yet |
| `raw/GHS_STAT_UCDB2015MT_GLOBE_R2019A/` | **GHS-UCDB** 2015 (R2019A v1.2). Urban Centre Database: one polygon per city (~13k centres), city-level attributes. Less relevant here (city list, not a within-city grid). [JRC dataset](http://data.europa.eu/89h/53473144-b88c-44bc-b4a3-4583ed1f547e). Not in the main pipeline |
| `processed/` | Harmonised GPKGs from `./run` (`city/{COUNTRY}/{city}/`) plus footprint aligned parquet under `processed/footprints/` |
| `baselines/` | Meta baseline GPKGs built from PDC zips (`{COUNTRY}/fb_baseline_median_h{00\|08\|16}.gpkg`) |


`data_root` is `/Users/wenlanzhang/Downloads/PhD_UCL/Data` in `config/regions.json`. Override with `RESIDENTIAL_DATA_ROOT` without editing the file.
