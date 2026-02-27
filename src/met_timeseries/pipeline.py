"""
Procedural pipeline: orchestrates data fetching, derivation, aggregation,
and output writing for all metzones and months in the configured date range.

Entry point: :func:`run_pipeline`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import geopandas as gpd
import xarray as xr

from met_timeseries.config import PipelineConfig
from met_timeseries.polygons import load_polygons
from met_timeseries.ledger import is_complete, mark_complete, get_incomplete
from met_timeseries.aggregation import aggregate_over_polygon
from met_timeseries.io import save_timeseries
from met_timeseries.sources.nldas import fetch_nldas
from met_timeseries.sources.prism import fetch_prism
from met_timeseries.sources.base import BoundingBox
from met_timeseries.derivations import derive_variables

logger = logging.getLogger(__name__)


def run_pipeline(config: PipelineConfig) -> None:
    """Run the full met-timeseries pipeline.

    Steps for each (year, month) that is not already in the ledger:

    1. Fetch NLDAS-2 data for each metzone bounding box.
    2. Derive secondary variables.
    3. Aggregate each variable over each metzone polygon.
    4. Save per-metzone timeseries to Parquet.
    5. Optionally fetch PRISM data and repeat steps 3–4.
    6. Mark the (step, year, month) complete in the ledger.

    Parameters
    ----------
    config:
        Fully-specified :class:`~met_timeseries.config.PipelineConfig`.
    """
    logger.info("Loading polygons from '%s'", config.catchment_path)
    polygons = load_polygons(
        config.catchment_path,
        metzone_column=config.metzone_column,
        target_crs=config.target_crs,
    )

    nldas_months = get_incomplete(
        config.ledger_path, "nldas", config.start_year, config.end_year
    )
    logger.info("%d NLDAS month(s) remaining", len(nldas_months))

    for year, month in nldas_months:
        process_nldas_month(
            polygons=polygons,
            metzone_column=config.metzone_column,
            year=year,
            month=month,
            output_dir=config.output_dir,
            cache_dir=config.cache_dir,
            variables=config.variables,
            ledger_path=config.ledger_path,
        )


def process_nldas_month(
    polygons: gpd.GeoDataFrame,
    metzone_column: str,
    year: int,
    month: int,
    output_dir: str,
    cache_dir: str,
    variables: list[str],
    ledger_path: str,
) -> None:
    """Fetch NLDAS-2 data, aggregate over each metzone, and save outputs.

    Parameters
    ----------
    polygons:
        GeoDataFrame of dissolved metzone polygons (from
        :func:`~met_timeseries.polygons.load_polygons`).
    metzone_column:
        Column in *polygons* containing the metzone identifier.
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    output_dir:
        Root directory for output Parquet files.
    cache_dir:
        Directory used to cache downloaded NLDAS files.
    variables:
        NLDAS-2 variable names to process.
    ledger_path:
        Path to the CSV ledger file.
    """
    if is_complete(ledger_path, "nldas", year, month):
        logger.debug("NLDAS %d-%02d already complete; skipping", year, month)
        return

    logger.info("Processing NLDAS %d-%02d", year, month)

    for _, row in polygons.iterrows():
        metzone_id = row[metzone_column]
        geom = row.geometry
        bounds = BoundingBox(
            west=geom.bounds[0],
            south=geom.bounds[1],
            east=geom.bounds[2],
            north=geom.bounds[3],
        )

        raw = fetch_nldas(bounds, year, month, variables=variables, cache_dir=cache_dir)
        derived = derive_variables(raw)

        for var_name, da in derived.items():
            stats = aggregate_over_polygon(xr.Dataset({var_name: da}), geom)
            row_df = pd.DataFrame(
                [{"year": year, "month": month, **stats}]
            )
            save_timeseries(
                df=row_df,
                output_dir=output_dir,
                metzone_id=metzone_id,
                step="nldas",
                variable=var_name,
                year=year,
                month=month,
            )

    mark_complete(ledger_path, "nldas", year, month)


def process_prism_month(
    polygons: gpd.GeoDataFrame,
    metzone_column: str,
    year: int,
    month: int,
    output_dir: str,
    cache_dir: str,
    variables: list[str] | None,
    ledger_path: str,
) -> None:
    """Fetch PRISM data, aggregate over each metzone, and save outputs.

    Parameters
    ----------
    polygons:
        GeoDataFrame of dissolved metzone polygons.
    metzone_column:
        Column in *polygons* containing the metzone identifier.
    year:
        Calendar year.
    month:
        Calendar month (1–12).
    output_dir:
        Root directory for output Parquet files.
    cache_dir:
        Directory used to cache downloaded PRISM files.
    variables:
        PRISM variable short-names to process.  Defaults to
        ``["ppt", "tmax", "tmin"]``.
    ledger_path:
        Path to the CSV ledger file.
    """
    if is_complete(ledger_path, "prism", year, month):
        logger.debug("PRISM %d-%02d already complete; skipping", year, month)
        return

    if variables is None:
        variables = ["ppt", "tmax", "tmin"]

    logger.info("Processing PRISM %d-%02d", year, month)

    for _, row in polygons.iterrows():
        metzone_id = row[metzone_column]
        geom = row.geometry
        bounds = BoundingBox(
            west=geom.bounds[0],
            south=geom.bounds[1],
            east=geom.bounds[2],
            north=geom.bounds[3],
        )

        ds = fetch_prism(bounds, year, month, variables=variables, cache_dir=cache_dir)

        for var_name in ds.data_vars:
            stats = aggregate_over_polygon(ds[[var_name]], geom)
            row_df = pd.DataFrame(
                [{"year": year, "month": month, **stats}]
            )
            save_timeseries(
                df=row_df,
                output_dir=output_dir,
                metzone_id=metzone_id,
                step="prism",
                variable=str(var_name),
                year=year,
                month=month,
            )

    mark_complete(ledger_path, "prism", year, month)


def assemble_final_output(
    nldas_df: pd.DataFrame,
    prism_df: pd.DataFrame,
    year: int,
    month: int,
) -> pd.DataFrame:
    """Merge NLDAS and PRISM monthly summaries for a single time step.

    Parameters
    ----------
    nldas_df:
        Monthly NLDAS statistics (must include ``metzone_id`` column).
    prism_df:
        Monthly PRISM statistics (must include ``metzone_id`` column).
    year:
        Calendar year (added as a column).
    month:
        Calendar month (added as a column).

    Returns
    -------
    pandas.DataFrame
        Merged DataFrame with one row per metzone.
    """
    merged = nldas_df.merge(prism_df, on="metzone_id", how="outer", suffixes=("_nldas", "_prism"))
    merged["year"] = year
    merged["month"] = month
    return merged
