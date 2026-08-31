#%% imports
import dask.dataframe as dd
from met_timeseries import derivations, disaggregation
import xarray as xr
from pathlib import Path
import geopandas as gpd

def get_filepaths(cache_dir, year):
    files = list(Path(cache_dir).glob("*.nc"))
    selected_files = [file for file in files if int(file.stem.split('_')[-1][0:4]) == year]
    return selected_files

#%% paths
prism_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\800m") 
nldas_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas")
mrms_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\mrms")
narr_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\narr")
aorc_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\aorc")
output_dir = Path("C:/Users/mfratki/Documents/met_extension/")


#%% Load in driving datasets
gdf = gpd.read_file("C:/Users/mfratki/Documents/github/hspf_spatial/files/env_watershed_hspfmodel_catch.gdb", layer="watershed_hspfmodel_catchments", driver='OpenFileGDB')
catchments_subset = gdf.loc[gdf['sam_name'] == model_name].dissolve(by='hydrozone').to_crs("EPSG:4326")

nldas_weights = xr.open_dataset("C:/Users/mfratki/Documents/github/hspf_spatial/files/nldas_weights.nc",chunks={"time": 720, "lat": -1, "lon": -1})
prism_weights = xr.open_dataset("C:/Users/mfratki/Documents/github/hspf_spatial/files/prism_weights_800m.nc",chunks={"time": 720, "lat": -1, "lon": -1})

#%%

# 1. LAZY LOAD
# Instead of feeding in chunks manually, load the whole multi-year cache lazily.
nldas_ds = xr.open_mfdataset(get_filepaths(nldas_cache_dir, 2001), chunks={"time": 720, "lat": -1, "lon": -1})
prism_ds = xr.open_mfdataset(get_filepaths(prism_cache_dir, 2001), chunks={"time": 720, "lat": -1, "lon": -1})

nldas_hourly, nldas_daily, prism_daily = _derive_grids(nldas_ds, prism_ds)


# 3. LAZY SPATIAL AGGREGATION 
hourly_nldas_zonal = (nldas_hourly * nldas_weights["weights"]).sum(dim=["lat", "lon"])
daily_nldas_zonal = (nldas_daily * nldas_weights["weights"]).sum(dim=["lat", "lon"])
daily_prism_zonal = (prism_daily * prism_weights["weights"]).sum(dim=["lat", "lon"])


# 2. MERGE INTO SOURCE-AGNOSTIC FREQUENCY DATASETS
# -----------------------------------------------------------------
# We use .assign_attrs() to tag the source directly onto the DataArray metadata

hourly_zonal = xr.Dataset({
    "temperature_c": hourly_nldas_zonal["temperature_c"].assign_attrs(source="NLDAS"),
    "wind_speed_ms": hourly_nldas_zonal["wind_speed_ms"].assign_attrs(source="NLDAS"),
    "precip_mm": hourly_nldas_zonal["hourly_precip_mm"].assign_attrs(source="NLDAS"),
    # If you ever swap to AORC, you just change the inputs here. 
    # Downstream code won't care.
})

daily_zonal = xr.Dataset({
    "precip_mm": daily_prism_zonal["daily_precip_mm"].assign_attrs(source="PRISM"),
    "pet_knb_mm": daily_nldas_zonal["pet_knb_mm"].assign_attrs(source="NLDAS"),
    "pet_oudin_mm": daily_nldas_zonal["pet_oudin_mm"].assign_attrs(source="NLDAS"),
    #"cloud_cover": daily_nldas_zonal["cloud_cover"].assign_attrs(source="NLDAS")
})

final_zonal_ds = disaggregation.disaggregate_precipitation_hybrid(
    coarse=daily_zonal["precip_mm"],
    fine_pattern=hourly_zonal["precip_mm"]
)




# 2. LAZY PHYSICS (Domain-wide)
# This replaces _derive_grids. No data is computed yet.
def _derive_grids(nldas: xr.Dataset, prism: xr.Dataset) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    """Domain-wide pixel-wise derivations returning separate hourly and daily Datasets."""

    # 1. Create empty "cargo ships" for our specific output frequencies
    nldas_hourly = xr.Dataset()
    nldas_daily = xr.Dataset()
    prism_daily = xr.Dataset()

    # --- NLDAS Hourly Derived Variables ---
    # Windspeed and Temperature
    nldas_hourly["temperature_c"] = derivations.kelvin_to_celsius(nldas["Tair"])
    nldas_hourly["wind_speed_ms"] = derivations.wind_speed(nldas["Wind_E"], nldas["Wind_N"])
    nldas_hourly["dewpoint_c"] = derivations.dewpoint_august_roche_magnus(
        nldas_hourly["temperature_c"], nldas["Qair"], nldas["PSurf"]
    )
    nldas_hourly["shortwave_wm2"] = nldas["SWdown"]
    nldas_hourly["hourly_precip_mm"] = nldas["Rainf"]
    # --- NLDAS Daily Derived Variables ---
    # PET calculations
    t_daily_mean = nldas_hourly["temperature_c"].resample(time="1D").mean()
    t_daily_min = nldas_hourly["temperature_c"].resample(time="1D").min()
    t_daily_max = nldas_hourly["temperature_c"].resample(time="1D").max()
    wind_daily_mean = nldas_hourly["wind_speed_ms"].resample(time="1D").mean()
    sw_daily_mj = (nldas_hourly["shortwave_wm2"] * 0.0036).resample(time="1D").sum()
    nldas_daily["pet_knb_mm"] = derivations.penman_knb(
        t_daily_mean, t_daily_min, t_daily_max, 
        nldas_hourly["dewpoint_c"].resample(time="1D").mean(), 
        wind_daily_mean, 
        sw_daily_mj
    )
    # nldas_daily["cloud_cover"] = derivations.cloud_cover_davis(
    #     nldas_hourly["shortwave_wm2"], min_clearsky=10, k=0.6
    # ).resample(time="1D").mean()

    nldas_daily['pet_oudin_mm'] = derivations.oudin(t_daily_mean)

    # --- PRISM Derived Variables ---
    prism_daily["daily_precip_mm"] = prism['ppt']
    
    return nldas_hourly, nldas_daily, prism_daily


# 4. LAZY TEMPORAL DISAGGREGATION
# Apply your disaggregation to the 1D polygon timeseries.
# This replaces _derive_timeseries.
final_zonal_ds = disaggregation.disaggregate_precipitation_hybrid(
    coarse=prism_zonal["ppt"],
    fine_pattern=nldas_zonal["Rainf"]
)

# 5. THE GREAT PIVOT (Xarray to Dask DataFrame)
# We flatten the multi-dimensional array into a tabular format, still lazily.
ddf = final_zonal_ds.to_dask_dataframe()

# 6. DISTRIBUTED EXECUTION
# This single command replaces your entire save_to_parquet loop.
# Dask spins up its workers, computes the math chunk-by-chunk, 
# and builds the Hive-partitioned directory structure automatically.
ddf.to_parquet(
    "staged_metzones_output.parquet",
    engine="pyarrow",
    partition_on=["polygon_id"], # Automatically creates the folders you need for WDM
    write_index=True
)