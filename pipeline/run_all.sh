#!/bin/bash
# Build the harmonised grids that analysis/ consumes.
#
# The pipeline is deliberately short. It produces exactly two artefacts per city:
#   01/harmonised_meta_worldpop.gpkg   Meta + WorldPop + GRDI on the Meta quadkey grid
#   01b_coverage/independent_grid.gpkg the *independent* city grid, including the tiles
#                                      Meta never published — this is the analysis input
#
# The second one is the point. Step 01 starts from published Meta tiles, so a tile Meta
# suppressed cannot appear in it. Step 01b rebuilds the grid from the clip polygon so the
# suppressed tiles are present and flagged, which is what makes the coverage analysis in
# analysis/ possible at all.
#
# Usage (from the repository root):
#   ./run --region PHL          all selected cities in a country
#   ./run --all                 every selected city in every country
#   ./run --one KEN_Nairobi     a single city
#   ./run --region KEN --ref-hour 8
#   ./run --all --include-out-of-sample      also the in_sample:false cities
#
# Then:  python analysis/01_build_panel.py, then 02-09 and 12.

if [ -n "${ZSH_VERSION:-}" ]; then
  exec /bin/bash "$0" "$@"
fi

set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
SCRIPTS="$(pwd)/pipeline"
REGIONS=()
REF_HOUR=""
POVERTY_SOURCE=""
CLIP_SOURCE=""
EXTRA=()

usage() {
  cat <<'EOF'
Usage: ./run [--region COUNTRY | --all | --one REGION_CODE] [options]

  --region COUNTRY     All selected cities in that country
                       (PHL, KEN, MEX, IDN, LKA, COL, ECU, ZAF)
  --all                All selected cities in every country
  --one REGION_CODE    A single city, e.g. KEN_Nairobi
  --include-out-of-sample
                       Also build the cities marked in_sample:false (Nakuru,
                       Garden Route). analysis/09_sensitivity.py needs them for
                       its out-of-sample row, so a full reproduction wants
                       ./run --all --include-out-of-sample
  --ref-hour HOUR      Meta baseline hour: 0, 8 or 16 (default: per-country)
  --poverty-source S   grdi (default) or rwi
  --clip-source S      local (default), osm or geob
  --rebuild            Rebuild the 01b independent grid even if cached
EOF
}

# Parsed before the loop so it applies whatever order the flags come in.
IN_SAMPLE_ONLY=True
for arg in "$@"; do
  [[ "$arg" == "--include-out-of-sample" ]] && IN_SAMPLE_ONLY=False
done

while [[ $# -gt 0 ]]; do
  case $1 in
    --region)
      mapfile -t found < <("$PYTHON" -c "
import sys; sys.path.insert(0,'pipeline')
import region_config as rc
print('\n'.join(c for c in rc.list_cities(in_sample_only=$IN_SAMPLE_ONLY) if c.startswith('$2_')))")
      REGIONS+=("${found[@]}")
      shift 2 ;;
    --all)
      mapfile -t found < <("$PYTHON" -c "
import sys; sys.path.insert(0,'pipeline')
import region_config as rc
print('\n'.join(rc.list_cities(in_sample_only=$IN_SAMPLE_ONLY)))")
      REGIONS+=("${found[@]}")
      shift ;;
    --include-out-of-sample) shift ;;
    --one)      REGIONS+=("$2"); shift 2 ;;
    --ref-hour) REF_HOUR="$2"; shift 2 ;;
    --poverty-source) POVERTY_SOURCE="$2"; shift 2 ;;
    --clip-source)    CLIP_SOURCE="$2"; shift 2 ;;
    -h|--help)  usage; exit 0 ;;
    *)          EXTRA+=("$1"); shift ;;
  esac
done

if [[ ${#REGIONS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

HARMONISE_ARGS=()
[[ -n "$REF_HOUR" ]]       && HARMONISE_ARGS+=(--ref-hour "$REF_HOUR")
[[ -n "$POVERTY_SOURCE" ]] && HARMONISE_ARGS+=(--poverty-source "$POVERTY_SOURCE")
[[ -n "$CLIP_SOURCE" ]]    && HARMONISE_ARGS+=(--clip-source "$CLIP_SOURCE")

echo "Cities to build: ${#REGIONS[@]}"
for REGION in "${REGIONS[@]}"; do
  echo ""
  echo "=========================================="
  echo "  $REGION"
  echo "=========================================="

  echo "[1/2] 01 harmonise Meta + WorldPop + GRDI ..."
  "$PYTHON" "$SCRIPTS/01_harmonise_datasets.py" --region "$REGION" \
    "${HARMONISE_ARGS[@]+"${HARMONISE_ARGS[@]}"}" "${EXTRA[@]+"${EXTRA[@]}"}"

  echo "[2/2] 01b independent grid + coverage QA ..."
  "$PYTHON" "$SCRIPTS/01b_meta_coverage_qa.py" --region "$REGION" \
    "${EXTRA[@]+"${EXTRA[@]}"}"
done

echo ""
echo "Done. Next:"
echo "  python analysis/01_build_panel.py         # writes both tile panels"
echo "  python analysis/02_selection_models.py    # then 03-09 and 12"
