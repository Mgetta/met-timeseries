"""
forcings/nldas.py
Single-pass processing for NLDAS forcing timeseries. Acts as a fallback for when PRISM data is unavailable, but can also be used in conjunction with PRISM for a more complete forcing dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr

from met_timeseries import derivations, disaggregation
from met_timeseries.weights import weighted_mean_timeseries

logger = logging.getLogger(__name__)

CST_OFFSET = -6

@dataclass
class CatchmentResult:
    name: str
    metzone_id: str
    data: dict[str, pd.Series]
    error: Exception | None = None


# --- Step 1: Domain-Wide Physics ---
def derive_grids(nldas: xr.Dataset) -> tuple[dict, dict]:
    """Domain-wide pixel-wise derivations using dictionaries to avoid grid alignment issues."""

    # --- NLDAS Derived Variables ---
    temp_c = derivations.kelvin_to_celsius(nldas["Tair"])
    wind = derivations.adjust_wind_height(
        derivations.wind_speed(nldas["Wind_E"], nldas["Wind_N"]), z_from=10.0, z_to=2.0
    )
    dew_c = derivations.dewpoint_august_roche_magnus(temp_c, nldas["Qair"], nldas["PSurf"])
    
    # Pre-calculate NLDAS daily aggregates (still on NLDAS grid)
    t_daily_mean = temp_c.resample(time="1D").mean()
    t_daily_min = temp_c.resample(time="1D").min()
    t_daily_max = temp_c.resample(time="1D").max()
    wind_daily_mean = wind.resample(time="1D").mean()
    sw_daily_mj = (nldas["SWdown"] * 0.0036).resample(time="1D").sum()
    
    # Compute PET on NLDAS grid
    pet_knb = derivations.penman_knb(
        t_daily_mean, t_daily_min, t_daily_max, 
        dew_c.resample(time="1D").mean(), 
        wind_daily_mean, 
        sw_daily_mj
    )

    # --- Construct Dictionaries ---
    # These maintain their respective NLDAS coordinates
    hourly_grids = {
        "temperature_c": temp_c,
        "dewpoint_c": dew_c,
        "wind_speed_ms": wind,
        "shortwave_wm2": nldas["SWdown"],
        "hourly_precip_mm": nldas["Rainf"], 
    }

    daily_grids = {
        "cloud_cover": derivations.cloud_cover_davis(nldas["SWdown"], min_clearsky=10, k=0.6).resample(time="1D").mean(),
        "pet_knb_mm": pet_knb,
    }

    return hourly_grids, daily_grids


# --- Step 2: 1D Timeseries Math ---
def derive_timeseries(hourly_agg: xr.Dataset, daily_agg: xr.Dataset) -> xr.Dataset:
    """1D operations (temporal disaggregation) on spatially lumped timeseries."""
    pet_hourly = disaggregation.disaggregate(
        coarse=daily_agg["pet_knb_mm"],
        conservation="sum",
        fine_pattern=hourly_agg["shortwave_wm2"],
        weight_method="proportional"
    )


    return xr.merge([
        hourly_agg,
        daily_agg, 
        pet_hourly.rename("pet_final_mm"),
    ])


# --- Step 3: HSPF Standard Units ---
def to_hspf_units(final_ts: xr.Dataset) -> dict[str, pd.Series]:
    """Convert the final 1D timeseries to HSPF standard units."""
    return {
        "temperature_f":      final_ts["temperature_c"].to_series() * 9/5 + 32,
        "dewpoint_f":         final_ts["dewpoint_c"].to_series() * 9/5 + 32,
        "wind_speed_mph":     final_ts["wind_speed_ms"].to_series() * 2.23694,
        "shortwave_langley":  final_ts["shortwave_wm2"].to_series() / 11.63,
        "cloud_cover_tenths": final_ts["cloud_cover"].reindex_like(final_ts["temperature_c"], method="ffill").to_series() * 10,
        "pet_in":             final_ts["pet_final_mm"].to_series() / 25.4,
        "precip_in":          final_ts["hourly_precip_mm"].to_series() / 25.4,
    }


# --- Step 4: The Main Loop ---
def process_chunk(
    nldas: xr.Dataset,
    polygons: gpd.GeoDataFrame,
    metzone_column: str = "metzone_id"
) -> list[CatchmentResult]:
    """
    Process a chunk of time across the domain, then aggregate by polygon.
    """
    results = []

    # 1. Domain-wide Grid Derivation
    try:
        logger.info("Deriving physical grids for domain...")
        hourly_grid, daily_grid = derive_grids(nldas)
    except Exception as e:
        logger.error(f"Grid derivation failed: {e}")
        return [CatchmentResult(row[metzone_column], {}, error=e) for _, row in polygons.iterrows()]

    # 2. Polygon-by-polygon spatial aggregation and 1D disaggregation
    logger.info(f"Aggregating grids for {len(polygons)} polygons...")
    for _, row in polygons.iterrows():
        metzone_id = row[metzone_column]
        geom = row.geometry
        
        #try:
        # Spatial Aggregation
        hourly_agg = {k: weighted_mean_timeseries(v, geom) for k, v in hourly_grid.items()}
        daily_agg = {k: weighted_mean_timeseries(v, geom) for k, v in daily_grid.items()}

        # 1D Timeseries Derivations (Temporal Disaggregation)
        final_ts = derive_timeseries(hourly_agg, daily_agg)

        # HSPF Unit Conversion
        hspf_inputs = to_hspf_units(final_ts)
        
        results.append(CatchmentResult(metzone_id, hspf_inputs))
        
        #except Exception as e:
        #    logger.error(f"Failed processing catchment {metzone_id}: {e}")
        #    results.append(CatchmentResult(metzone_id, {}, error=e))

    return results


def save_to_parquet(results: list[CatchmentResult], output_root: str | Path, model_name: str):        
    # Ensure output_root is a Path object so we can use the '/' operator
    output_root = Path(output_root)
    
    for res in results:
        # Skip failed catchments so we don't crash the whole save process
        if res.error:
            # You might want to use logging.warning here instead
            print(f"Skipping save for metzone {res.metzone_id} due to error: {res.error}")
            continue
            
        # 1. Convert the dict of pandas Series into a single DataFrame
        # res.data contains the variables (temperature_f, precip_in, etc.)
        df = pd.DataFrame(res.data)
        
        # Ensure the index is named for clarity when reading back
        df.index.name = "datetime"
        
        # 2. Construct the partition directory path
        # CatchmentResult doesn't have a 'name' attribute, so pass model_name as an argument
        partition_dir = output_root / f"model={model_name}" / f"metzone={res.metzone_id}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        
        # 3. Write this specific temporal chunk to the directory
        start_date = df.index.min().strftime("%Y%m%d")
        end_date = df.index.max().strftime("%Y%m%d")
        file_path = partition_dir / f"forcings_{start_date}_{end_date}.parquet"
        
        # Use pyarrow engine for best performance and compatibility
        df.to_parquet(file_path, engine='pyarrow')