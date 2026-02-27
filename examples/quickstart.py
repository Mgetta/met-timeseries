"""Minimal quickstart example for met-timeseries.

Run from the examples/ directory:

    python quickstart.py
"""

from pathlib import Path

from met_timeseries import Pipeline, PipelineConfig

# Configure the pipeline for a short date range
config = PipelineConfig(
    polygon_path=Path(__file__).parent / "example_polygons.geojson",
    output_dir=Path(__file__).parent / "output",
    start_date="2020-01-01",
    end_date="2020-01-31",
)

print(f"Configuration:")
print(f"  Polygons: {config.polygon_path}")
print(f"  Output:   {config.output_dir}")
print(f"  Period:   {config.start_date} to {config.end_date}")
print(f"  Sources:  {config.sources}")
print()

# Create and run the pipeline
pipeline = Pipeline(config)

# Check what months still need processing
from met_timeseries.ledger import Ledger
ledger = Ledger(config.output_dir)
incomplete = ledger.get_incomplete("nldas", config.start_year, config.end_year)
print(f"Months remaining to process: {len(incomplete)}")
print()

# Uncomment to actually run (requires internet access for NLDAS/PRISM data):
# pipeline.run()

print("To run the pipeline, uncomment the pipeline.run() line.")
print("This will download NLDAS-2 and PRISM data for January 2020.")
