"""
Quickstart example: met-timeseries procedural pipeline.

This script demonstrates how to use the met-timeseries package to:

1. Read a catchment vector file and dissolve catchments into metzones.
2. Inspect the dissolved polygons.
3. Configure and run the pipeline (here, against a small date range for
   illustration; replace with your actual years and paths for real use).

Run from the repository root::

    python examples/quickstart.py

You will need valid NASA Earthdata credentials in your ~/.netrc for the NLDAS
fetch calls to succeed.  Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD, or
configure ~/.netrc as described at:
https://disc.gsfc.nasa.gov/data-access#python-requests
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configure logging so progress is visible on stdout
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)

from met_timeseries.config import PipelineConfig
from met_timeseries.pipeline import run_pipeline
from met_timeseries.polygons import load_polygons

# ---------------------------------------------------------------------------
# 1. Load and dissolve catchments
# ---------------------------------------------------------------------------
CATCHMENT_PATH = Path(__file__).parent / "example_catchments.geojson"

polygons = load_polygons(
    catchment_path=str(CATCHMENT_PATH),
    metzone_column="metzone_id",
    target_crs="EPSG:4326",
)

print(f"\nLoaded {len(polygons)} metzones:")
print(polygons[["metzone_id", "geometry"]].to_string())
print()

# ---------------------------------------------------------------------------
# 2. Inspect metzone bounding boxes
# ---------------------------------------------------------------------------
for _, row in polygons.iterrows():
    bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
    print(f"  {row['metzone_id']}: bounds={bounds}")

print()

# ---------------------------------------------------------------------------
# 3. Configure the pipeline
# ---------------------------------------------------------------------------
config = PipelineConfig(
    catchment_path=str(CATCHMENT_PATH),
    metzone_column="metzone_id",
    output_dir="output/quickstart",
    cache_dir=".cache/quickstart",
    start_year=2010,
    end_year=2010,  # single year for quick demo
    variables=["APCP", "TMP", "DSWRF"],
    ledger_path="ledger_quickstart.csv",
)

print("Pipeline config:")
print(f"  catchment_path : {config.catchment_path}")
print(f"  metzone_column : {config.metzone_column}")
print(f"  output_dir     : {config.output_dir}")
print(f"  date range     : {config.start_year}–{config.end_year}")
print()

# ---------------------------------------------------------------------------
# 4. Run the pipeline
#    (Comment this out if you don't have Earthdata credentials configured.)
# ---------------------------------------------------------------------------
# run_pipeline(config)
#
# After running, Parquet files are written to:
#   output/quickstart/nldas/<metzone_id>/<variable>/<year>_<month:02d>.parquet
#
# Load one with:
#   import pandas as pd
#   df = pd.read_parquet("output/quickstart/nldas/UPPER_COLUMBIA_1/precip_mm/2010_01.parquet")
#   print(df)

print("Done.  Uncomment run_pipeline(config) to actually fetch data.")
