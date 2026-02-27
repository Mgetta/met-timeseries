"""Command-line interface for met-timeseries."""

from __future__ import annotations

import sys
from pathlib import Path

import click


@click.group()
@click.version_option()
def cli() -> None:
    """met-timeseries: Meteorological timeseries extraction and aggregation pipeline."""


@cli.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to a JSON/YAML config file.")
@click.option("--polygon-path", type=click.Path(), default=None,
              help="Path to polygon GeoJSON/GeoPackage.")
@click.option("--output-dir", type=click.Path(), default="output",
              show_default=True, help="Root output directory.")
@click.option("--start-date", default="1996-01-01", show_default=True,
              help="Start date (YYYY-MM-DD).")
@click.option("--end-date", default="present", show_default=True,
              help="End date (YYYY-MM-DD or 'present').")
@click.option("--steps", default="all", show_default=True,
              help="Comma-separated steps to run: nldas,prism,derive,aggregate or 'all'.")
@click.option("--polygon-id-column", default="polygon_id", show_default=True,
              help="Column name for polygon IDs.")
def run(
    config_path: str | None,
    polygon_path: str | None,
    output_dir: str,
    start_date: str,
    end_date: str,
    steps: str,
    polygon_id_column: str,
) -> None:
    """Run the met-timeseries processing pipeline."""
    from met_timeseries.config import PipelineConfig  # noqa: PLC0415
    from met_timeseries.pipeline import Pipeline  # noqa: PLC0415

    if polygon_path is None and config_path is None:
        click.echo("Error: --polygon-path is required when --config is not provided.", err=True)
        sys.exit(1)

    step_list = None if steps == "all" else steps.split(",")

    config = PipelineConfig(
        polygon_path=Path(polygon_path) if polygon_path else Path("."),
        output_dir=Path(output_dir),
        start_date=start_date,
        end_date=end_date,
        polygon_id_column=polygon_id_column,
    )
    pipeline = Pipeline(config)
    pipeline.run(steps=step_list)
    click.echo("Pipeline completed successfully.")


@cli.command()
@click.option("--output-dir", type=click.Path(), default="output", show_default=True,
              help="Root output directory to inspect.")
def status(output_dir: str) -> None:
    """Show processing status from the ledger."""
    from met_timeseries.ledger import Ledger  # noqa: PLC0415

    ledger = Ledger(Path(output_dir))
    df = ledger.load()
    if df.empty:
        click.echo("No completed steps found in ledger.")
        return

    click.echo(f"Ledger: {ledger.path}")
    click.echo(f"Total completed entries: {len(df)}")
    click.echo("\nCompleted steps summary:")
    summary = df.groupby("step").agg(count=("year", "count")).reset_index()
    click.echo(summary.to_string(index=False))


@cli.command()
@click.option("--polygon-path", type=click.Path(exists=True), required=True,
              help="Path to polygon GeoJSON/GeoPackage.")
@click.option("--id-column", default="polygon_id", show_default=True,
              help="Column name for polygon IDs.")
@click.option("--crs", default=4326, show_default=True,
              help="Target CRS EPSG code.")
def validate(polygon_path: str, id_column: str, crs: int) -> None:
    """Validate polygon inputs."""
    from met_timeseries.polygons import load_polygons  # noqa: PLC0415

    click.echo(f"Validating {polygon_path!r} ...")
    try:
        gdf = load_polygons(polygon_path, id_column=id_column, target_crs=crs)
        click.echo(f"OK Loaded {len(gdf)} polygons successfully.")
        click.echo(f"  CRS: {gdf.crs}")
        click.echo(f"  ID column: {id_column!r} ({gdf[id_column].nunique()} unique values)")
        click.echo(f"  Bounds: {gdf.total_bounds}")
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"FAIL Validation failed: {exc}", err=True)
        sys.exit(1)
