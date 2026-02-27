"""
Command-line interface for met-timeseries.

Usage example::

    met-timeseries \\
        --catchment-path catchments.gpkg \\
        --metzone-column metzone_id \\
        --output-dir output \\
        --start-year 2000 \\
        --end-year 2005
"""

from __future__ import annotations

import logging

import click

from met_timeseries.config import PipelineConfig
from met_timeseries.pipeline import run_pipeline


@click.command()
@click.option(
    "--catchment-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the vector file containing model catchments "
    "(shapefile, GeoPackage, GeoJSON, Parquet, …).",
)
@click.option(
    "--metzone-column",
    default="metzone_id",
    show_default=True,
    help="Attribute column that groups catchments into metzones.",
)
@click.option(
    "--output-dir",
    default="output",
    show_default=True,
    type=click.Path(),
    help="Directory for output Parquet timeseries files.",
)
@click.option(
    "--cache-dir",
    default=".cache",
    show_default=True,
    type=click.Path(),
    help="Directory for caching downloaded source data.",
)
@click.option(
    "--start-year",
    default=2000,
    show_default=True,
    type=int,
    help="First calendar year to process (inclusive).",
)
@click.option(
    "--end-year",
    default=2020,
    show_default=True,
    type=int,
    help="Last calendar year to process (inclusive).",
)
@click.option(
    "--ledger-path",
    default="ledger.csv",
    show_default=True,
    type=click.Path(),
    help="Path to the CSV ledger tracking completed steps.",
)
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def main(
    catchment_path: str,
    metzone_column: str,
    output_dir: str,
    cache_dir: str,
    start_year: int,
    end_year: int,
    ledger_path: str,
    log_level: str,
) -> None:
    """Download and process meteorological timeseries for metzone polygons.

    Reads a vector file containing model catchments, dissolves them by
    METZONE_COLUMN, then fetches NLDAS-2 (and optionally PRISM) data for
    each dissolved polygon over the requested date range.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )

    config = PipelineConfig(
        catchment_path=catchment_path,
        metzone_column=metzone_column,
        output_dir=output_dir,
        cache_dir=cache_dir,
        start_year=start_year,
        end_year=end_year,
        ledger_path=ledger_path,
    )
    run_pipeline(config)


if __name__ == "__main__":
    main()
