"""
forcings/hybrid.py
Single-pass processing for NLDAS/PRISM forcing timeseries.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from met_timeseries import weights
from met_timeseries.forcings import nldas
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


# --- Step 1: Domain-Wide Physics with weights ---
def _derive_grids(nldas: xr.Dataset, prism: xr.Dataset) -> tuple[dict, dict]:
    """Domain-wide pixel-wise derivations using dictionaries to avoid grid alignment issues."""

    # --- NLDAS Derived Variables ---
    temp_c = derivations.kelvin_to_celsius(nldas["Tair"])
    # wind = derivations.adjust_wind_height(
    #     derivations.wind_speed(nldas["Wind_E"], nldas["Wind_N"]), z_from=10.0, z_to=2.0
    # )
    wind =derivations.wind_speed(nldas["Wind_E"], nldas["Wind_N"])
    dew_c = derivations.dewpoint_august_roche_magnus(temp_c, nldas["Qair"], nldas["PSurf"])
    shortwave = nldas["SWdown"]
    precip_hourly = nldas["Rainf"]

    # Pre-calculate NLDAS daily aggregates (still on NLDAS grid)
    t_daily_mean = temp_c.resample(time="1D").mean()
    t_daily_min = temp_c.resample(time="1D").min()
    t_daily_max = temp_c.resample(time="1D").max()
    wind_daily_mean = wind.resample(time="1D").mean()
    sw_daily_mj = (shortwave * 0.0036).resample(time="1D").sum()
    # Compute PET on NLDAS grid
    pet_knb = derivations.penman_knb(
        t_daily_mean, t_daily_min, t_daily_max, 
        dew_c.resample(time="1D").mean(), 
        wind_daily_mean, 
        sw_daily_mj
    )
    cloud_cover = derivations.cloud_cover_davis(shortwave, min_clearsky=10, k=0.6).resample(time="1D").mean()
    
    nldas_grids = {
        "temperature_c": temp_c,
        "dewpoint_c": dew_c,
        "wind_speed_ms": wind,
        "shortwave_wm2": shortwave,
        "hourly_precip_mm": precip_hourly,
        "cloud_cover": cloud_cover,
        "pet_knb_mm": pet_knb}
    


    # --- PRISM Derived Variables ---
    precip_daily = prism['ppt']
    prism_grids = {
        "daily_precip_mm": precip_daily}
    
    return nldas_grids, prism_grids
    
    
def _weight_grids(nldas_grids, prism_grids, nldas_weights, prism_weights):
    
    # Apply the weights to the gridded data before spatial aggregation
    nldas_grids_weighted = {k: weights._weight_dataset(v, nldas_weights) for k, v in nldas_grids.items()}
    prism_grids_weighted = {k: weights._weight_dataset(v, prism_weights) for k, v in prism_grids.items()}

    hourly_agg = {'temperature_c': nldas_grids_weighted["temperature_c"],
                  'dewpoint_c': nldas_grids_weighted["dewpoint_c"],
                  'wind_speed_ms': nldas_grids_weighted["wind_speed_ms"],
                  'shortwave_wm2': nldas_grids_weighted["shortwave_wm2"],
                  'hourly_precip_mm': nldas_grids_weighted["hourly_precip_mm"]}
    daily_agg = {'cloud_cover': nldas_grids_weighted["cloud_cover"],
                 'pet_knb_mm': nldas_grids_weighted["pet_knb_mm"],
                 'daily_precip_mm': prism_grids_weighted["daily_precip_mm"]}

    return hourly_agg, daily_agg




# --- Step 2: 1D Timeseries Math ---
def _derive_timeseries(hourly_agg: xr.Dataset, daily_agg: xr.Dataset) -> xr.Dataset:
    """1D operations (temporal disaggregation) on spatially lumped timeseries."""
    pet_hourly = disaggregation.disaggregate(
        coarse=daily_agg["pet_knb_mm"],
        conservation="sum",
        fine_pattern=hourly_agg["shortwave_wm2"],
        weight_method="proportional"
    )

    precip_hourly = disaggregation.disaggregate_precipitation_hybrid(
        coarse=daily_agg["daily_precip_mm"],  
        fine_pattern=hourly_agg["hourly_precip_mm"], 
        enforce_coarse_mass=False,
        cascade_method="molnar_burlando"
    )

    hourly_agg['cloud_cover'] = daily_agg['cloud_cover'].reindex_like(hourly_agg["temperature_c"], method="ffill")
    return xr.merge([
        hourly_agg,
        pet_hourly.rename("pet_final_mm"),
        precip_hourly.rename("precip_final_mm")
    ])


# --- Step 3: HSPF Standard Units ---
def to_hspf_units(final_ts: xr.Dataset) -> dict[str, pd.Series]:
    """Convert the final 1D timeseries to HSPF standard units."""
    return {
        "temperature_f":      final_ts["temperature_c"].to_series() * 9/5 + 32,
        "dewpoint_f":         final_ts["dewpoint_c"].to_series() * 9/5 + 32,
        "wind_speed_mph":     final_ts["wind_speed_ms"].to_series() * 2.23694,
        "shortwave_langley":  final_ts["shortwave_wm2"].to_series() / 11.63,
        "cloud_cover_tenths": final_ts["cloud_cover"].to_series() * 10,
        "pet_in":             final_ts["pet_final_mm"].to_series() / 25.4,
        "precip_in":          final_ts["precip_final_mm"].to_series() / 25.4,
    }


# --- Step 4: The Main Loop ---
def _process_chunk(nldas_data: xr.Dataset, 
                   prism_data: xr.Dataset, 
                   nldas_weights: xr.Dataset,
                   prism_weights: xr.Dataset) -> list[CatchmentResult]:
    """
    This is a placeholder for a more optimized version of process_chunk that applies the weights directly to the gridded data before spatial aggregation, rather than applying them to the timeseries after aggregation. This would require reworking the weighted_mean_timeseries function to handle the weighted grid data and ensure proper alignment of coordinates.
    """

    nldas_grids, prism_grids = _derive_grids(nldas_data,prism_data)
    hourly_agg,daily_agg = _weight_grids(nldas_grids, prism_grids, nldas_weights, prism_weights)

    final_ts = pd.DataFrame(to_hspf_units(_derive_timeseries(hourly_agg,daily_agg)))
    return final_ts


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