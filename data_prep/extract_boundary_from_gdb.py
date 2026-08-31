#!/usr/bin/env python3
"""
Extract city/municipality boundaries from a .gdb (File Geodatabase), .shp (Shapefile), or .geojson for use as clip_shape.

Use this to create boundary files (GPKG, SHP) from admin boundary GDBs/GeoJSON before running
the harmonisation pipeline. The output can be set as clip_shape in config/regions.json.

Workflow (GDB):
  1. List layers:     python data_prep/extract_boundary_from_gdb.py -i path/to.gdb --list
  2. Inspect layer:  python data_prep/extract_boundary_from_gdb.py -i path/to.gdb --inspect LAYER
  3. Extract & save:  python data_prep/extract_boundary_from_gdb.py -i path/to.gdb -l LAYER -c CODE1,CODE2 -o output.gpkg
  4. One file per city: add --split to save each feature as a separate .gpkg

Workflow (GeoJSON — single layer, no -l needed):
  python data_prep/extract_boundary_from_gdb.py -i mex_admin2.geojson --inspect
  python data_prep/extract_boundary_from_gdb.py -i mex_admin2.geojson -n "México" --name-col adm1_name -o data/raw/boundaries/Mex/Mexico_state.gpkg

Kenya example (Shapefile, ADM1):
  python data_prep/extract_boundary_from_gdb.py -i ken_admbnda_adm1_iebc_20191031.shp -n Mombasa --name-col ADM1_EN -o data/raw/boundaries/Kenya/Mombasa.gpkg

Philippines example (PSA NAMRIA GDB):
  python data_prep/extract_boundary_from_gdb.py -i phl_adm_psa_namria_20231106_GDB2.gdb --list
  python data_prep/extract_boundary_from_gdb.py -i phl_adm_psa_namria_20231106_GDB2.gdb -l phl_admbnda_adm2_psa_namria_20231106 -c PH1102402,PH1004305 -o data/raw/boundaries/mindanao_cities.gpkg --split

Dependencies: geopandas, fiona (fiona only for .gdb)
"""

import argparse
import re
import sys
from pathlib import Path

import geopandas as gpd

try:
    import fiona
except ImportError:
    fiona = None


def _is_geojson(path: Path) -> bool:
    """True if input is GeoJSON (single-layer, no fiona needed)."""
    return path.suffix.lower() in (".geojson", ".json")


def _is_single_layer(path: Path) -> bool:
    """True if input is single-layer (GeoJSON or Shapefile, no layer name needed)."""
    return path.suffix.lower() in (".geojson", ".json", ".shp")


def list_layers(input_path: Path) -> list[str]:
    """List all layers. For GDB returns layer names; for GeoJSON/Shapefile returns single placeholder."""
    if _is_single_layer(input_path):
        return ["(single layer)"]
    if fiona is None:
        raise ImportError("fiona required for .gdb: pip install fiona")
    return fiona.listlayers(str(input_path))


def inspect_layer(input_path: Path, layer_name: str | None = None, n_rows: int = 5) -> None:
    """Print layer schema and sample rows."""
    if _is_single_layer(input_path):
        gdf = gpd.read_file(input_path, rows=n_rows)
        layer_name = layer_name or input_path.name
    else:
        if not layer_name:
            raise ValueError("--layer required for .gdb. Use --list to see layers.")
        gdf = gpd.read_file(input_path, layer=layer_name, rows=n_rows)
    print(f"\nLayer: {layer_name}")
    print(f"  CRS: {gdf.crs}")
    print(f"  Rows (sample): {len(gdf)}")
    print("  Columns:", gdf.columns.tolist())
    code_cols = [c for c in gdf.columns if "PCODE" in c.upper() or "CODE" in c.upper()]
    name_cols = [c for c in gdf.columns if "NAME" in c.upper() or c.upper().endswith("_EN")]
    print("  Possible code columns:", code_cols or "(none found)")
    print("  Possible name columns:", name_cols or "(none found)")
    print("\n  Sample:")
    print(gdf.head().to_string())


def _sanitize_filename(s: str) -> str:
    """Make string safe for use as filename."""
    s = re.sub(r'[^\w\s-]', '', str(s))
    s = re.sub(r'[\s_]+', '_', s).strip('_')
    return s or "feature"


def extract_and_save(
    input_path: Path,
    layer_name: str | None,
    codes: list[str] | None = None,
    names: list[str] | None = None,
    code_col: str | None = None,
    name_col: str | None = None,
    name_exact: bool = False,
    output_path: Path | None = None,
    split: bool = False,
) -> gpd.GeoDataFrame:
    """
    Extract features by code or name and optionally save.

    If code_col/name_col are not given, auto-detect from layer schema.
    If split=True, save each feature to a separate .gpkg in the output directory.
    For GeoJSON, layer_name is ignored.
    """
    if _is_single_layer(input_path):
        gdf = gpd.read_file(input_path)
    else:
        if not layer_name:
            raise ValueError("--layer required for .gdb")
        gdf = gpd.read_file(input_path, layer=layer_name)

    # Auto-detect code/name columns if not provided
    if code_col is None:
        candidates = [c for c in gdf.columns if "PCODE" in c.upper() or "CODE" in c.upper()]
        code_col = candidates[0] if candidates else None
    if name_col is None:
        candidates = [c for c in gdf.columns if "NAME" in c.upper() or c.upper().endswith("_EN")]
        name_col = candidates[0] if candidates else None

    if codes:
        if code_col is None:
            raise ValueError("No code column found. Specify --code-col.")
        subset = gdf[gdf[code_col].astype(str).isin([str(c).strip() for c in codes])].copy()
    elif names:
        if name_col is None:
            raise ValueError("No name column found. Specify --name-col.")
        if name_exact:
            subset = gdf[gdf[name_col].astype(str).isin([str(n).strip() for n in names])].copy()
        else:
            pattern = "|".join(re.escape(n) for n in names)
            subset = gdf[gdf[name_col].astype(str).str.contains(pattern, case=False, na=False)].copy()
    else:
        raise ValueError("Provide --codes or --names to filter.")

    if len(subset) == 0:
        print("Warning: No features matched. Check codes/names and column.")
        # Diagnostic: show what we looked for and sample values in the layer
        if codes and code_col:
            uniq = gdf[code_col].astype(str).dropna().unique()
            sample = uniq[:15].tolist() if len(uniq) > 15 else uniq.tolist()
            print(f"  Looked for codes: {codes}")
            print(f"  In column: {code_col}")
            print(f"  Sample values in layer: {sample}")
            print("  Tip: Try --inspect LAYER to see schema. Davao/Cagayan may be in ADM3 layer.")
        elif names and name_col:
            print(f"  Looked for names: {names}")
            print(f"  In column: {name_col}")
        return subset

    print(f"Extracted {len(subset)} features")
    if code_col and code_col in subset.columns:
        print(subset[[c for c in [name_col, code_col] if c in subset.columns]].to_string(index=False))

    if output_path:
        output_path = Path(output_path)
        if split:
            # Save each feature as separate .gpkg in output directory
            out_dir = output_path.parent / output_path.stem if output_path.suffix else output_path
            out_dir.mkdir(parents=True, exist_ok=True)
            geom_col = subset.geometry.name
            for idx, row in subset.iterrows():
                single = gpd.GeoDataFrame([row], geometry=geom_col, crs=subset.crs)
                if code_col and code_col in subset.columns:
                    fname = _sanitize_filename(str(row[code_col])) + ".gpkg"
                elif name_col and name_col in subset.columns:
                    fname = _sanitize_filename(str(row[name_col])) + ".gpkg"
                else:
                    fname = f"feature_{idx}.gpkg"
                out_file = out_dir / fname
                single.to_file(out_file, driver="GPKG")
                print(f"  Saved: {out_file}")
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ext = output_path.suffix.lower()
            if ext == ".gpkg":
                subset.to_file(output_path, driver="GPKG")
            elif ext == ".shp":
                subset.to_file(output_path, driver="ESRI Shapefile")
            else:
                subset.to_file(output_path)
            print(f"Saved: {output_path}")

    return subset


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract city/municipality boundaries from .gdb for clip_shape",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-i", "--input", type=Path, required=True, help="Path to .gdb geodatabase")
    p.add_argument("--list", action="store_true", help="List all layers and exit")
    p.add_argument("--inspect", type=str, nargs="?", metavar="LAYER", const=".", help="Inspect layer (schema, sample rows). For .geojson, LAYER optional.")
    p.add_argument("-l", "--layer", type=str, help="Layer name to read (required for extract)")
    p.add_argument("-c", "--codes", type=str, help="Comma-separated codes to filter (e.g. PH1102402,PH1004305)")
    p.add_argument("-n", "--names", type=str, help="Comma-separated name substrings to filter (partial match)")
    p.add_argument("--code-col", type=str, help="Column for code filter (auto-detect if omitted)")
    p.add_argument("--name-col", type=str, help="Column for name filter (auto-detect if omitted)")
    p.add_argument("--exact", action="store_true", help="Exact match for --names (default: partial/substring match)")
    p.add_argument("-o", "--output", type=Path, help="Output path (.gpkg or .shp). With --split, creates a directory with one .gpkg per feature.")
    p.add_argument("--split", action="store_true", help="Save each city/feature as a separate .gpkg file")
    return p.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    if args.list:
        layers = list_layers(input_path)
        print("Layers:")
        for lyr in layers:
            print(" -", lyr)
        return

    if args.inspect is not None:
        inspect_layer(input_path, args.inspect if not _is_single_layer(input_path) else None)
        return

    # Extract mode
    is_geo = _is_single_layer(input_path)
    if not is_geo and not args.layer:
        print("Error: --layer required for .gdb. Use --list to see layers, --inspect LAYER to inspect.")
        sys.exit(1)

    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None
    names = [n.strip() for n in args.names.split(",")] if args.names else None

    if not codes and not names:
        print("Error: Provide --codes or --names to filter.")
        sys.exit(1)

    extract_and_save(
        input_path,
        args.layer if not is_geo else None,
        codes=codes,
        names=names,
        code_col=args.code_col,
        name_col=args.name_col,
        name_exact=args.exact,
        output_path=args.output,
        split=args.split,
    )


if __name__ == "__main__":
    main()
