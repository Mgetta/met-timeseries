

#%% imports
import geopandas as gpd
from met_timeseries.sources.base import BoundingBox
from met_timeseries.sources import prism
from met_timeseries.spatial import  weights
from met_timeseries.utils import clip_dataset, hdf5WDM
import xarray as xr
import pandas as pd
from met_timeseries.sources.stations import ndawn
import matplotlib.pyplot as plt
from pathlib import Path
import dask.dataframe as dd
from met_timeseries import derivations, disaggregation
import xarray as xr
from pathlib import Path
import geopandas as gpd

def get_filepaths(cache_dir, year):
    files = list(Path(cache_dir).glob("*.nc"))
    selected_files = [file for file in files if int(file.stem.split('_')[-1][0:4]) == year]
    return selected_files

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

    nldas_daily["cloud_cover"] = derivations.cloud_cover_davis(
        nldas_hourly["shortwave_wm2"], min_clearsky=10, k=0.6
    ).resample(time="1D").mean()

    nldas_daily['pet_oudin_mm'] = derivations.oudin(t_daily_mean)

    # --- PRISM Derived Variables ---
    prism_daily["daily_precip_mm"] = prism['ppt']

    return nldas_hourly, nldas_daily, prism_daily


#%% paths
prism_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\800m") 
nldas_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas")
mrms_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\mrms")
narr_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\narr")
aorc_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\aorc")

nldas_zarr_path = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas.zarr")
prism_zarr_path = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\prism.zarr")

output_dir = Path("C:/Users/mfratki/Documents/met_extension/")


#%% Load in driving datasets
model_name = 'LOWUS'
gdf = gpd.read_file("C:/Users/mfratki/Documents/github/hspf_spatial/files/env_watershed_hspfmodel_catch.gdb", layer="watershed_hspfmodel_catchments", driver='OpenFileGDB')
catchments_subset = gdf.loc[gdf['sam_name'] == model_name].dissolve(by='hydrozone').to_crs("EPSG:4326").reset_index()
#catchments_subset = gdf.dissolve(by = ['hydrozone','sam_name']).to_crs("EPSG:4326").reset_index()
bounds = BoundingBox(
    west=catchments_subset.total_bounds[0], 
    south=catchments_subset.total_bounds[1],
    east=catchments_subset.total_bounds[2],
    north=catchments_subset.total_bounds[3])

# wdm_path = Path(f'C:/Users/mfratki/Documents/Projects/pipeline/{model_name}/model/{model_name}_Met_2024.hdf5')
# wdm = hdf5WDM(wdm_path)
# hydrozone_config = {
#     15: {  # NDAWN station_id 95 (Williams)
#         'PEVT': 152,
#         'PREC': 151,
#         'ATEM': 154,
#         'SOLR': 155,
#         'WIND': 156,
#         'DEWP': 157,
#         'CLOU': 159
#     },
#     17: {  # NDAWN station_id 95 (Williams)
#         'PEVT': 172,
#         'PREC': 171,
#         'ATEM': 174,
#         'SOLR': 175,
#         'WIND': 176,
#         'DEWP': 177,
#         'CLOU': 179
#     },
#     11: {
#         'PEVT': 112,
#         'PREC': 111,
#         'ATEM': 114,
#         'SOLR': 115,
#         'WIND': 116,
#         'DEWP': 117,
#         'CLOU': 119
#     },
#     13: {
#         'PEVT': 132,
#         'PREC': 131,
#         'ATEM': 134,
#         'SOLR': 135,
#         'WIND': 136,
#         'DEWP': 137,
#         'CLOU': 139
#     }
# }

#%% Load in gridded products


# nldas_ds = xr.open_mfdataset(
#     str(nldas_cache_dir / "*.nc"),
#     engine="h5netcdf",       # Bypasses the C-library lock
#     chunks={"time": 720, "lat": -1, "lon": -1}, 
#     parallel=False,          # Turn off Dask for the metadata crawl
#     concat_dim="time",       
#     combine="nested",        
#     coords="minimal",        
#     data_vars="minimal",
#     compat="override"        
# )

nldas_ds = xr.open_mfdataset(nldas_zarr_path, engine="zarr")
prism_ds = xr.open_mfdataset(prism_zarr_path, engine="zarr")
prism_ds = prism._shift_time_coord(prism_ds)

# Only grab a year of data
nldas_ds = nldas_ds.sel(time=slice("2001-01-01", "2001-12-31"))
prism_ds = prism_ds.sel(time=slice("2001-01-01", "2001-12-31"))

# nldas_ds = xr.open_mfdataset(get_filepaths(nldas_cache_dir, 2001), chunks={"time": 720, "lat": -1, "lon": -1})
# prism_ds = xr.open_mfdataset(get_filepaths(prism_cache_dir, 2001), chunks={"time": 720, "lat": -1, "lon": -1})
# prism_ds = prism._shift_time_coord(prism_ds)
#nldas_ds = xr.open_mfdataset(Path(nldas_cache_dir).glob("*.nc"),chunks={"time": 720, "lat": -1, "lon": -1} )
#prism_ds = xr.open_mfdataset(Path(prism_cache_dir).glob("*.nc"), chunks={"time": 720, "lat": -1, "lon": -1})
# nldas_ds = nldas_ds.chunk({"time": 720})
# prism_ds = prism_ds.chunk({"time": 30})

# nldas_ds = nldas_ds.chunk({"time": 720})
nldas_data = clip_dataset(nldas_ds,bounds)
prism_data = clip_dataset(prism_ds,bounds)

print("Datasets loaded and clipped to bounds. Proceeding to build weightmaps...")

nldas_weights = weights.build_weightmap(nldas_data['Rainf'],catchments_subset,poly_id_columns=['sam_name','hydrozone'])
prism_weights = weights.build_weightmap(prism_data['ppt'],catchments_subset,poly_id_columns=['sam_name','hydrozone'])

print("Weightmaps built. Proceeding to derive grids and perform spatial aggregation...")
#%%
nldas_hourly, nldas_daily, prism_daily = _derive_grids(nldas_data, prism_data)

print("Grids derived. Proceeding to spatial aggregation...")
# 3. LAZY SPATIAL AGGREGATION 
hourly_nldas_zonal = (nldas_hourly * nldas_weights["weights"]).sum(dim=["lat", "lon"])
daily_nldas_zonal = (nldas_daily * nldas_weights["weights"]).sum(dim=["lat", "lon"])
daily_prism_zonal = (prism_daily * prism_weights["weights"]).sum(dim=["lat", "lon"])

# 1. Cleanly slice out the variables you need and unify the precip name
hourly_zonal = hourly_nldas_zonal[[
    "temperature_c", "wind_speed_ms", "dewpoint_c", "shortwave_wm2", "hourly_precip_mm"
]].rename({"hourly_precip_mm": "precip_mm"})

daily_zonal = daily_nldas_zonal[["pet_knb_mm", "pet_oudin_mm", "cloud_cover"]]

# 2. Bulk-assign source attributes (Keeps code DRY)
for var in hourly_zonal.data_vars:
    hourly_zonal[var].attrs["source"] = "NLDAS"

daily_zonal["pet_knb_mm"].attrs["source"] = "NLDAS"
daily_zonal["pet_oudin_mm"].attrs["source"] = "NLDAS"
daily_zonal["cloud_cover"].attrs["source"] = "NLDAS"
# 3. The Matrix Transpose (Prepare chunks for 1D timeseries math)
hourly_zonal = hourly_zonal.chunk({"time": -1, "polygon_index": 10})
daily_zonal = daily_zonal.chunk({"time": -1, "polygon_index": 10}) 
daily_prism_zonal = daily_prism_zonal.chunk({"time": -1, "polygon_index": 10})
# Initial Compute of simple vector math
print("Starting initial compute to trigger any lazy loading and inspect chunking...")
#%%
hourly_ds = hourly_zonal.load()
daily_ds = daily_zonal.load()
daily_prism_ds = daily_prism_zonal.load()

print("Initial compute complete. Proceeding to build computationally complex compute graph...")
#%%
# Build computationally complex compute graph
# 4. Temporal Disaggregation (Inject results directly back into the hourly dataset)


# daily_ds["cloud_cover"] = derivations.cloud_cover_davis(
#     hourly_ds["shortwave_wm2"], min_clearsky=10, k=0.6
# ).resample(time="1D").mean()


hourly_zonal["hourly_precip_mm"] = disaggregation.disaggregate_precipitation_hybrid(
    coarse=daily_prism_zonal["daily_precip_mm"], # Pass the native PRISM array directly
    fine_pattern=hourly_zonal["precip_mm"], 
    enforce_coarse_mass=False
)

hourly_zonal["pet_knb_mm"] = disaggregation.disaggregate(
    coarse=daily_zonal["pet_knb_mm"],
    conservation="sum",
    fine_pattern=hourly_zonal["shortwave_wm2"],
    weight_method="proportional"
)

hourly_zonal["cloud_cover"] = daily_zonal["cloud_cover"].reindex_like(hourly_zonal['shortwave_wm2'], method="ffill")

hourly_zonal["pet_oudin_mm"] = disaggregation.disaggregate(
    coarse=daily_zonal["pet_oudin_mm"],
    conservation="sum",
    fine_pattern=hourly_zonal["shortwave_wm2"],
    weight_method="proportional"
)

# 5. Finalize
final_ds = hourly_zonal.chunk({"time": -1, "polygon_index": 10})

#convert units
final_ds = final_ds.assign(
    precipitation = final_ds['hourly_precip_mm']/25.4,
    solar_radiation = final_ds['shortwave_wm2'] * 0.0864,
    wind_speed = final_ds['wind_speed_ms']*2.23694,
    dewpoint = final_ds['dewpoint_c'] * 1.8 + 32,
    temperature = final_ds['temperature_c'] * 1.8 + 32,
    pet_knb = final_ds['pet_knb_mm']/25.4,
    pet_oudin = final_ds['pet_oudin_mm']/25.4,
    cloud_cover = final_ds['cloud_cover']*10
)

#shift to utc-6
final_ds = final_ds.assign_coords(time=final_ds.time - pd.Timedelta(hours=6))

# # Stage for storing as parquet files
ddf = final_ds.to_dask_dataframe()
#%%
print("Final dataset prepared. Proceeding to write to Parquet and then WDM...")
ddf.to_parquet(
    output_dir / "test.parquet",
    engine="pyarrow",
    compression="snappy",
    partition_on=["sam_name", "hydrozone"]
)


# #%% To dataframe
df = ddf.compute()
df['hydrozone'] = pd.to_numeric(df['hydrozone'])
df = df.sort_values(by=['sam_name', 'hydrozone', 'time'])
df = df.set_index('time')
model_name = 'Pine'
wdm_path = f'C:/Users/mfratki/Documents/Projects/pipeline/{model_name}/model/{model_name}_Met_2024.wdm'
#precip
fig, ax = plt.subplots()
#(hourly_zonal.sel(hydrozone=hydrozone)['precip_mm']/25.4).plot(ax=ax)#.shift(-6).plot(ax = ax)
df_wdm = wdmtoolbox.extract(wdm_path,151).plot(ax=ax)
df_new = df.query("hydrozone == 15")['precipitation']
df_wdm.plot(ax=ax)
df_new.plot(ax=ax)
ax.set_xlim(pd.Timestamp('01-01-2001'),pd.Timestamp('12-31-2001'))


#Dewpoint
fig, ax = plt.subplots()
df_wdm = wdmtoolbox.extract(wdm_path,157).plot(ax=ax)
df_new = df.query("hydrozone == 15")['dewpoint']
df_wdm.plot(ax=ax)
df_new.plot(ax=ax)
ax.set_xlim(pd.Timestamp('01-01-2001'),pd.Timestamp('12-31-2001'))


#cloud cover
fig, ax = plt.subplots()
(final_ds.sel(hydrozone=hydrozone)['cloud_cover']).shift(time =-6).plot(ax=ax)#.shift(-6).plot(ax = ax)
wdm.series(hydrozone_config[hydrozone]['CLOU']).plot(ax = ax)
#daily_zonal.sel(hydrozone=hydrozone)['cloud_cover'].plot(ax=ax)
ax.set_xlim(pd.Timestamp('01-01-2001'),pd.Timestamp('12-31-2001'))

#shortwave
fig, ax = plt.subplots()
(final_ds.sel(polygon_index=2)['shortwave_wm2'] * 0.0864).plot(ax=ax)#.shift(-6).plot(ax = ax)
wdm.series(hydrozone_config[hydrozone]['SOLR']).shift(6).plot(ax = ax)
ax.set_xlim(pd.Timestamp('01-01-2001'),pd.Timestamp('12-31-2001'))

#cloud cover
fig, ax = plt.subplots()
(final_ds.sel(polygon_index=2)['cloud_cover']).plot(ax=ax)#.shift(-

# # date = '04-16-2001'
# # fig, ax = plt.subplots()
# # (final_ds.sel(polygon_index=2)['precip_mm']/25.4).plot(ax=ax)#.shift(-6).plot(ax = ax)
# # wdm.series(dsns['PREC']).shift(6).plot(ax = ax)
# # ax.set_xlim(pd.Timestamp(date),pd.Timestamp(date)+pd.Timedelta(days=1))
# # (hourly_zonal.sel(time = date).sel(polygon_index=2)['precip_mm'].to_series()/25.4).plot(ax=ax)
# # prism_total = daily_prism_zonal.sel(time = date).sel(polygon_index=2)['daily_precip_mm'].values
# # plt.hlines(prism_total/24.5, xmin=pd.Timestamp(date), xmax=pd.Timestamp(date)+pd.Timedelta(days=1), colors='r', linestyles='dashed')


# fig, ax = plt.subplots()
# (final_ds.sel(polygon_index=2)['wind_speed_ms']*2.23694).plot(ax=ax)#.shift(-6).plot(ax = ax)
# wdm.series(dsns['WIND']).shift(6).plot(ax = ax)
# #clip to 2001
# ax.set_xlim(pd.Timestamp("2001-01-01"), pd.Timestamp("2001-12-31"))

# fig, ax = plt.subplots()
# hourly_zonal.sel(time = '05-07-2001')['precip_mm'].to_series().plot(ax=ax)
# daily_zonal.sel(time = '05-07-2001')['precip_mm'].to_series().plot(ax=ax)
# #%%

# wdm_path = Path('C:/Users/mfratki/Documents/Projects/Tests/Pine/model/UpperMissMet.hdf5')
# wdm = hdf5WDM(wdm_path)
# hydrozone = 41 #NDAWN station_id 95 (Williams)
# dsns = {'PEVT': 6020,
#         'PREC': 4110,
#         'ATEM': 3930,
#         'SOLR': 6040,
#         'WIND': 6050,
#         'DEWP': 6060,
#         'CLOU': 6070}
# #%%

stations = ndawn.get_stations()
stations = pd.concat([ms.stations.nearby(ms.Point(point.y,point.x), limit=4) for point in list(catchments_subset.centroid)])
stations = stations.reset_index().drop_duplicates('id')
stations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.longitude, stations.latitude), crs="EPSG:4326")

# Set time period
start = datetime(2018, 1, 1)
end = datetime(2018, 12, 31, 23, 59)

# Get hourly data
ts = ms.hourly(ms.Station(id='KPWC0'), start, end)
df = ts.fetch()

#plot stations over catchments clipping to catchment bounds
fig, ax = plt.subplots(figsize=(10,10))
catchments_subset.plot(ax=ax, facecolor='none', edgecolor='red')
stations.plot(ax=ax, color='blue', markersize=50)
#stations2.plot(ax=ax, color='green', markersize=50) 
ax.set_xlim(catchments_subset.total_bounds[0] - 0.1, catchments_subset.total_bounds[2] + 0.1)
ax.set_ylim(catchments_subset.total_bounds[1] - 0.1, catchments_subset.total_bounds[3] + 0.1)
#label polygons with hydrozone
for idx, row in catchments_subset.reset_index().iterrows():
    centroid = row.geometry.centroid
    ax.annotate(text=str(row['hydrozone']), xy=(centroid.x, centroid.y), ha='center', va='center', fontsize=12, fontweight='bold', color='red')
ax.set_title("NDAWN Stations (blue) over Catchments (red)")



#%%
import wdmtoolbox as wdm
from hspf.uci import UCI
import geopandas as gpd
import pandas as pd
from hspf.wdm import hdf5WDM
from pathlib import Path


parquet_path = 'C:/Users/mfratki/Documents/met_extension/hspf.parquet'
wdm_path = 'C:/Users/mfratki/Documents/Projects/pipeline/Pine/model/Pine_Met_2025.wdm'
model_name = 'Pine'
wdm_path = f'C:/Users/mfratki/Documents/Projects/pipeline/{model_name}/model/{model_name}_Met_2025.wdm'
old_wdm = f'C:/Users/mfratki/Documents/Projects/pipeline/{model_name}/model/UpperMissMet.wdm'

df_met = pd.read_parquet(parquet_path,
                         filters = [('sam_name','==',model_name)])
df_met['hydrozone'] = pd.to_numeric(df_met['hydrozone']).astype(int)
df_met.set_index('time', inplace=True)
#precipitation
fig, ax = plt.subplots()
df_wdm = wdmtoolbox.extract(old_wdm,4110).plot(ax=ax)
df_new = df_met.query("hydrozone == 41")['precipitation']
df_wdm.plot(ax=ax)
df_new.plot(ax=ax)
ax.set_xlim(pd.Timestamp('01-01-1999'),pd.Timestamp('12-31-1999'))

#prec
fig, ax = plt.subplots()
df_wdm = wdmtoolbox.extract(wdm_path,39).plot(ax=ax)
df_new = df_met.query("hydrozone == 15")['precipitation']
df_wdm.plot(ax=ax)
df_new.plot(ax=ax)
ax.set_xlim(pd.Timestamp('01-01-2001'),pd.Timestamp('12-31-2001'))

#Pet evaporation
fig, ax = plt.subplots()
df_wdm = wdmtoolbox.extract(old_wdm,4320).plot(ax=ax)
df_new = df_met.query("hydrozone == 43")['pet_oudin']
df_wdm.plot(ax=ax)
df_new.plot(ax=ax)
ax.set_xlim(pd.Timestamp('01-01-1999'),pd.Timestamp('12-31-1999'))





extsources = uci.table('EXT SOURCES')
extsources = extsources.loc[extsources['SMEMN'].isin(['PREC','PEVT','ATEM','SOLR','WIND','DEWP','CLOU'])]
dsn_map = extsources.drop_duplicates(['SVOLNO','SMEMN'])


variable_mapping = {
    'precipitation': 'PREC',
    'pet_knb': 'PEVT',
    'pet_oudin': 'PEVT',
    'temperature': 'ATEM',
    'solar_radiation': 'SOLR',
    'wind_speed': 'WIND',
    'dewpoint': 'DEWP',
    'cloud_cover': 'CLOU'
}

variable_mapping = {v: k for k, v in variable_mapping.items()}

wdm_file = uci.filepath.parent / f"{model_name}_Met_2025.wdm"
wdm.createnewwdm(wdm_file,True)
for index,row in dsn_map.iterrows():
    dsn = row['SVOLNO']
    hydrozone = int(str(dsn)[0:2])
    if hydrozone == 60:
        hydrozone = 41
    #dsn = int(str(hydrozone) + str(dsn)[2:])
    smnem = row['SMEMN']
    ts_name = variable_mapping.get(smnem)
    ts_series = df_met.loc[df_met['hydrozone'] == hydrozone].set_index('time')[ts_name]

    try:
        wdm.createnewdsn(
                wdmpath=wdm_file.as_posix(), 
                dsn=dsn, 
                tstype=smnem, 
                base_year=1994, 
                tcode=3, 
                tsstep=1,
                scenario="GRIDDED",               # 8 char max
                description=ts_name  # 48 char max
                )
        wdm.csvtowdm(
                wdmpath=wdm_file.as_posix(), 
                dsn=dsn, 
                input_ts=ts_series
                )
    except wdm.DSNExistsError as e:
        print(f"Error processing dsn {dsn}: {e}")   





#%% Creating WDM
import wdmtoolbox as wdm


prism_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\800m") 
nldas_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas")
mrms_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\mrms")
narr_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\narr")
aorc_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\aorc")
output_dir = Path("C:/Users/mfratki/Documents/met_extension/")
# Function to determine the dsn number given a metzone and ts_name
def _dsn(ts_name,metzone = None):
    if metzone is None:
        metzone = 0
    assert(metzone<100)
    ts_number = { 'WIND':11,
                  'SOLR':12,
                  'PREC':10,
                  'PEVT':13,
                  'DEWP':14,
                  'CLOU':15,
                  'ATEM':16,
                  'NH4D':17,
                  'NH4W':18,
                  'NO3D':19,
                  'NO3W':20}
    return int(str(ts_number[ts_name]) + str(metzone))


variable_mapping = {
    'precipitation': 'PREC',
    'pet_knb': 'PEVT',
    'pet_oudin': 'PEVT',
    'temperature': 'ATEM',
    'solar_radiation': 'SOLR',
    'wind_speed': 'WIND',
    'dewpoint': 'DEWP',
    #'cloud_cover': 'CLOU'
}

wdm_file = output_dir / "test.wdm"
wdm.createnewwdm(wdm_file,True)
df = pd.read_parquet(output_dir / "test.parquet")
dsn = 1
for hydrozone in df['hydrozone'].unique():
    for variable in variable_mapping.keys():
        ts_name = variable_mapping[variable]
        #dsn = _dsn(ts_name, int(float(hydrozone)))
        wdmtoolbox.createnewdsn(
        wdmpath=wdm_file.as_posix(), 
        dsn=dsn, 
        tstype=ts_name, 
        base_year=1994, 
        tcode=3, 
        tsstep=1,
        scenario="GRIDDED",               # 8 char max
        description=variable  # 48 char max
        )
        series = df.loc[df['hydrozone'] == hydrozone].set_index('time')[variable]
        wdmtoolbox.csvtowdm(
        wdmpath=wdm_file.as_posix(), 
        dsn=dsn, 
        input_ts=series
        )
        dsn += 1





# 4. Write the DataFrame to the WDM file
# Pass the pandas DataFrame directly to the 'input_ts' argument.
wdmtoolbox.csvtowdm(
    wdmpath=wdm_file, 
    dsn=dsn, 
    input_ts=df
)



# nldas_weights = weights.build_weightmap(nldas_data,catchments, lat_dim="lat", lon_dim="lon", poly_id_columns=['sam_name',"hydrozone"])
# prism_weights = weights.build_weightmap(prism_data,catchments, lat_dim="lat", lon_dim="lon", poly_id_columns=['sam_name',"hydrozone"])

# nldas_path = Path("C:/Users/mfratki/Documents/github/hspf_spatial/files/nldas_weights.nc")
# prism_path = Path("C:/Users/mfratki/Documents/github/hspf_spatial/files/prism_weights_800m.nc")

# encoding = {
#     "weights": {
#         "zlib": True,       # Turn on compression
#         "complevel": 5,     # 1 is fastest, 9 is smallest. 4-5 is the sweet spot.
#         # Optional but highly recommended: chunking
#         # This makes reading individual polygons much faster
#         "chunksizes": (1, len(nldas_weights.lat), len(nldas_weights.lon)) 
#     }
# }
# nldas_weights.to_netcdf(nldas_path, engine="netcdf4", encoding=encoding)
# encoding = {
#     "weights": {
#         "zlib": True,       # Turn on compression
#         "complevel": 5,     # 1 is fastest, 9 is smallest. 4-5 is the sweet spot.
#         # Optional but highly recommended: chunking
#         # This makes reading individual polygons much faster
#         "chunksizes": (1, len(prism_weights.lat), len(prism_weights.lon)) 
#     }
# }
# prism_weights.to_netcdf(prism_path, engine="netcdf4", encoding=encoding)

# %%
