# Region configuration

`regions.json` defines data paths and map settings for each study region.

**How to run the pipeline:** see the [top-level README](../README.md). This file focuses on **paths and adding regions**.

## Usage

```bash
# All selected cities in a country (18 in-sample cities across PHL, KEN, MEX, IDN, LKA, COL, ECU, ZAF)
./run --region PHL
./run --region KEN
./run --region MEX
./run --region IDN
./run --region LKA
./run --region COL
./run --region ECU
./run --region ZAF
```

**Event footprints** (the Meta crisis AOI, not a city clip) are no longer built by `./run`. The eight harmonised footprint extracts are kept as `data/processed/footprints/{CODE}_aligned.parquet` for the replication described in `analysis/README.md` §7; the ISO3-only keys remain in `regions.json` as the data donors that produced them.

## Path resolution

- **data_root** (top-level): Base path for external data. Paths for `poverty` (RWI CSV) and `pdc_raw_dir` are relative to this.
- **poverty_source** (top-level): `grdi` (default) or `rwi`. Select with `./run --poverty-source rwi`.
- **poverty_grdi** (top-level): Path to the global GRDI GeoTIFF, relative to the **project root** (default `data/raw/povmap-grdi-v1-10.tif`). Not under `data_root`.
- **clip_source** (top-level): `local` (default), `osm`, or `geob`. Select with `./run --clip-source osm`. Per-region override is allowed.
- **Project paths**: `worldpop`, `meta`, `clip_shape`, `pdc_processed_csv` are relative to the project root. WorldPop GeoTIFFs live in `data/raw/worldpop/`.
- **meta** includes the reference hour: `data/baselines/{COUNTRY}/fb_baseline_median_h{00|08|16}.gpkg`. Use the default-hour file for the pipeline, or pass `--meta` to 01_harmonise when using a different hour.
- To use a different location for PDC/RWI, change `data_root` only. WorldPop and GRDI stay under `data/` unless you change those paths in config.

## Adding a new region

1. Add an entry to `regions.json`:

```json
"XXX": {
  "name": "Country Name",
  "worldpop": "data/raw/worldpop/xxx_pop_2025_CN_100m_R2025A_v1.tif",
  "meta": "outputs/fb_baseline_median_XXX.gpkg",
  "poverty": "/path/to/rwi.csv",
  "pdc_raw_dir": "Meta_Event/Pop/event.zip",
  "pdc_processed_csv": "outputs/PDC_XXX.csv",
  "pdc_use_baseline_column": false,
  "map_bbox": [xmin, ymin, xmax, ymax],
  "map_bbox_label": "Study area name",
  "lon_range": [min_lon, max_lon],
  "lat_range": [min_lat, max_lat]
}
```

2. Paths can be absolute or relative to the project root.
3. **City boundary (`clip_source`)**: which polygon clips the study area. Only quadkeys intersecting this boundary are analysed. When a clip is applied, figures use the clipped data extent instead of `map_bbox`.

   | Source | Flag | What it uses |
   |--------|------|----------------|
   | `local` (default) | `--clip-source local` | `clip_shape` file already on disk (`.gpkg`, `.shp`, `.geojson`). Set `clip_shape` to `null` for no clip (full extent). Extract from a .gdb with `data_prep/extract_boundary_from_gdb.py`. |
   | `osm` | `--clip-source osm` | [OSMnx](https://osmnx.readthedocs.io/) `geocode_to_gdf` (Nominatim). Falls back to Nominatim directly if osmnx is not installed. Query: `clip_osm_place` (e.g. `"Nairobi, Kenya"`). First download is cached in `data/raw/boundaries/cache/osm/`. |
   | `geob` | `--clip-source geob` | [geoBoundaries](https://www.geoboundaries.org/) gbOpen API. Filter by `clip_geob_iso3`, `clip_geob_adm`, `clip_geob_name`. Cached in `data/raw/boundaries/cache/geob/`. |

   Re-download with `--clip-refresh`. Step 01 also writes `data/processed/city/{COUNTRY}/{city}/01/clip_boundary.gpkg` so you can see the polygon that was used.

   Example:

   ```bash
   ./run --region KEN_Nairobi --clip-source local   # downloaded Nairobi.gpkg (default)
   ./run --region KEN_Nairobi --clip-source osm
   ./run --region KEN_Nairobi --clip-source geob
   python pipeline/01_harmonise_datasets.py --region KEN_Nairobi --clip-source osm --clip-osm-place "Nairobi City, Kenya"
   ```
4. `lon_range` / `lat_range`: used for auto-detecting region from data centroid.
5. **PDC (Meta baseline)**: `pdc_raw_dir` = Meta event `.zip` or unzipped folder under `data_root` (CSVs are read from the zip in memory; unzipping is optional). `pdc_processed_csv` = optional intermediate. `pdc_use_baseline_column` (optional): if omitted, auto-detected — data spans 14+ days → 7-day shift; under 14 days → use n_baseline from CSV (if present). Set `true` or `false` to override.

6. **Shared PDC extracts**: Cities in one country share WorldPop, poverty, and the country Meta baseline GPKG; only the clip differs. `./run --region IDN` (etc.) runs the **cities**. ISO3-only keys with `clip_shape` unset (`IDN`, `LKA`, `COL`, `ECU`, `ZAF`) stay in `regions.json` as data donors for those cities; they are not themselves a city run. PHL / KEN / MEX cities use local `clip_shape` files; IDN onward use `clip_source: geob`.

## Output layout

`./run --region KEN` writes grids to `data/processed/city/KEN/{Nairobi,Mombasa,Kisumu}/` under `01/` and `01b_coverage/`. Nakuru is skipped because it is `in_sample: false`; add `--include-out-of-sample` to build it too.

**Two different exclusions, in two different places.** `in_sample: false` in this file (Nakuru, Garden Route) controls what `./run` *builds*. `OUT_OF_SAMPLE` in `analysis/01_build_panel.py` controls what enters the *study panel*, and it additionally excludes **Kisumu** — which is therefore built by `./run --region KEN` but dropped from `tile_panel.parquet`. Kisumu is excluded on evidence rather than configuration (its absent tiles mark the edge of the published event AOI, not a coverage decision), so the reason lives next to the diagnostic that established it. See `analysis/README.md` §2.

`./run --region MEX` runs `MEX_MexicoCity`, `MEX_Puebla`, and `MEX_Leon`. Same pattern for IDN, LKA, COL, ECU, ZAF.  
Meta snapshot hour is Pacific time, chosen to sit near evening locally: **16** Kenya / South Africa; **8** Philippines / Indonesia / Sri Lanka; **0** Mexico / Colombia / Ecuador. Default baseline method is **n_baseline**.  
Analysis tables and figures go to `outputs/analysis/` and `figure/analysis/`.
