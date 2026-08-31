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
#%%
from met_timeseries import forcings
from met_timeseries.sources import nldas, prism
import pandas as pd

start_date = "2018-06-01"
end_date = '2018-06-30'
dates = [date.date().strftime("%Y-%m-%d") for date in pd.date_range(start_date,end_date)]


#%% Imports
import pandas as pd
from met_timeseries.sources.stations import ndawn
from met_timeseries import forcings
from met_timeseries import derivations, disaggregation
from met_timeseries.spatial.weights import weighted_mean_timeseries
from met_timeseries.sources.nldas import BoundingBox
from met_timeseries.sources import nldas, prism, narr
from met_timeseries.utils import mem_gb, hdf5WDM
import geopandas as gpd
from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import gc
#%% Load Target Metzone
model_name = 'LOWUS'
gdf = gpd.read_file("C:/Users/mfratki/Documents/github/hspf_spatial/files/env_watershed_hspfmodel_catch.gdb", layer="watershed_hspfmodel_catchments", driver='OpenFileGDB')
#statewide = gpd.read_file(Path('C:/Users/mfratki/Documents/github/hspf_spatial/files/statewide_subwatersheds.gpkg')).to_crs("EPSG:4326")
catchments = gdf.query('sam_name == @model_name')

CATCHMENT_PATH = catchments.to_crs("EPSG:4326")
wdm_path = Path('C:/Users/mfratki/Documents/Projects/Tests/LOWUS/model/LOWUS_Met_2024.hdf5')
wdm = hdf5WDM(wdm_path)

hydrozone = 15 #NDAWN station_id 95 (Williams)
dsns = {'PEVT': 152,
        'PREC': 151,
        'ATEM': 154,
        'SOLR': 155,
        'WIND': 156,
        'DEWP': 157,
        'CLOU': 159}

polygon = CATCHMENT_PATH[CATCHMENT_PATH['hydrozone'] == hydrozone].dissolve()
bounds = polygon.bounds
bounds = BoundingBox(
    west=bounds.minx.min(),
    south=bounds.miny.min(),
    east=bounds.maxx.max(),
    north=bounds.maxy.max(),)





#%% Read met products
# ── Step 0: fetch grids (before spatial aggregation) ──────────
prism_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism"
prism_resolution = '800m'
nldas_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas"
mrms_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\mrms"
narr_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\narr"


nldas_data = xr.open_mfdataset("path/to/nldas/*.nc")
prism_data = xr.open_mfdataset("path/to/prism/*.nc")
polygons = gpd.read_file("mn_catchments.shp")

# Run the pipeline
catchment_results = hybrid.process_chunk(
    nldas=nldas_data,
    prism=prism_data,
    polygons=polygons,
    metzone_column="HUC12"
)
start_date = "2001-01-01"
end_date = '2001-12-31'
dates = [date.date().strftime("%Y-%m-%d") for date in pd.date_range(start_date,end_date)]
nldas_ds = []
prism_ds = []
mrms_ds = []
for i, date in enumerate(dates):
    prism_data = prism.fetch_prism(date=date, cache_dir=prism_cache_dir, resolution=prism_resolution,bounds = bounds)
    nldas_data = nldas.fetch_nldas(date, nldas_cache_dir,bounds=bounds)
    # del nldas_data
    # del prism_data
    # gc.collect()
    #mrms_data = mrms.fetch_mrms(date,mrms_cache_dir,bounds=bounds)
    #narr_data = narr.fetch_narr(date, narr_cache_dir,bounds=bounds,variables = ['tcdc'])
    prism_ds.append(prism_data)
    nldas_ds.append(nldas_data)
    #mrms_ds.append(mrms_data)

    if i % 30 == 0:  # every ~month
        print(f"Day {i:3d} | {date} | Memory: {mem_gb():.2f} GB | "
              f"PRISM list: {len(prism_ds)} | NLDAS list: {len(nldas_ds)}")

#TODO earthaccess 504 Server Error: Gateway Time-out for url   
#%% Merge datasets
prism_ds = [ds.reset_coords('spatial_ref') for ds in prism_ds]  # Avoid coordinate conflicts when concatenating
#narr_data = narr.fetch_narr(2001, narr_cache_dir,bounds=bounds,variables = ['tcdc'])
nldas_data = xr.concat(nldas_ds, dim="time")
prism_data = xr.concat(prism_ds, dim="time")

nldas_data = nldas_data.shift(time=-6)
prism_data = prism_data.shift(time=-6)
#Plot an example variable and the metzone polygon to verify things look reasonable

#%%
fig, ax = plt.subplots()
ax.set_facecolor("lightgray")
ax.patch.set_alpha(0.5)
# Plot the xarray grid
prism_data.isel(time=180)['ppt'].plot(ax=ax)
# Plot the vector layer on top
CATCHMENT_PATH.loc[~(CATCHMENT_PATH['hydrozone'] == 15)].plot(ax=ax, facecolor="none", edgecolor="black",alpha = .5)
CATCHMENT_PATH.loc[CATCHMENT_PATH['hydrozone'] == 15].plot(ax=ax, facecolor="none", edgecolor="red",alpha = .5)

minx, miny, maxx, maxy = CATCHMENT_PATH.total_bounds
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
plt.title('LOWUS: PRISM Precipitation on 2001-07-01')
plt.show()

#%% Generate forcings
# Load data
nldas_data = xr.open_mfdataset("path/to/nldas/*.nc")
prism_data = xr.open_mfdataset("path/to/prism/*.nc")
polygons = gpd.read_file("mn_catchments.shp")

# Run the pipeline
catchment_results = hybrid.process_chunk(
    nldas=nldas_data,
    prism=prism_data,
    polygons=polygons,
    metzone_column="HUC12"
)

#%% Precipitation 



hourly_ts = weighted_mean_timeseries(
    hourly_precipitation,
    polygon.geometry.iloc[0],
)
daily_ts = weighted_mean_timeseries(daily_precipitation, polygon.geometry.iloc[0])
precip, counts = dl.disaggregate_precipitation(daily_ts['ppt'], hourly_ts['Rainf'], method='hybrid')

date = '2001-05-25'

hourly_precipitation =  weighted_mean_timeseries(nldas_data['Rainf'], polygon.geometry.iloc[0])
daily_precipitation = weighted_mean_timeseries(prism_data["ppt"], polygon.geometry.iloc[0])

#hourly_precipitation.sel(time=date).sum()
#daily_precipitation.sel(time=date).sum()

precip = disaggregation.disaggregate_precipitation_hybrid(daily_precipitation,
                                                          hourly_precipitation, 
                                                          enforce_coarse_mass=False)
precip = precip.to_pandas()/25.4

precip_old, counts = disaggregation_legacy.disaggregate_precipitation(daily_precipitation.to_pandas(), hourly_precipitation.to_pandas(), method='hybrid')

#pd.concat([precip_old.loc[date], precip.loc[date]], axis=1).rename(columns={0: 'old', 1: 'new'})
precip_old = precip_old/25.4
precip_wdm = wdm.series(dsns['PREC'])
precip_wdm = precip_wdm.loc[precip_wdm.index.year == 2001]
fig, ax = plt.subplots()
precip_wdm.plot(label='respec', ax=ax)
(precip.shift(-6)).plot(label='new', ax=ax)
precip_old.shift(-6).plot(label='legacy',ax=ax)


precip_wdm.loc[date].plot(label='respec', ax=ax)
(precip.loc[date].shift(-6)).plot(label='new', ax=ax)
plt.legend(['RESPEC','New'])
plt.ylabel('Precipitation (in)')
plt.xlabel('Date')

fig, ax = plt.subplots(figsize=(10, 4))

s_respec = precip_wdm.loc[precip.index.intersection(precip_wdm.index)].squeeze()
s_new = (precip.shift(-6) / 25.4).squeeze()

# Use ax.vlines instead of ax.bar
# Syntax: ax.vlines(x, ymin, ymax)
ax.vlines(s_respec.index, ymin=0, ymax=s_respec, label='RESPEC', 
          color='blue', alpha=0.6, linewidth=1.5)

ax.vlines(s_new.index, ymin=0, ymax=s_new, label='New', 
          color='orange', alpha=0.6, linewidth=1.5)

ax.set_ylabel('Precipitation (in)')
ax.set_xlabel('Date')
ax.legend()

plt.show()


#%% Wind Speed
"""
Can match RESPEC extension exactly:
    - RESPEC does not adjust from 10m to 2m
    - RESPEC does the polygon area weighting after converting to wind speed, while we do it before.  This should not make a difference for the mean, but could affect the variability if there are strong spatial gradients in wind speed across the catchment.
"""
wind_ms    = derivations.wind_speed(nldas_data["Wind_E"], nldas_data["Wind_N"])
#wind_ms = adjust_wind_height(wind_ms, z_from=10, z_to=2)
wind_nldas = weighted_mean_timeseries(wind_ms.to_dataset(), polygon.geometry.iloc[0])
#wind_nldas = np.sqrt(weighted_mean_timeseries(nldas_data["Wind_E"].to_dataset(), polygon.geometry.iloc[0])['Wind_E']**2 + 
#                     weighted_mean_timeseries(nldas_data["Wind_N"].to_dataset(), polygon.geometry.iloc[0])['Wind_N']**2) * 2.23694 # convert m/s to knots for comparison with WDM

wind_nldas = wind_nldas['wind_speed_ms'] * 2.23694 # convert m/s to knots for comparison with WDM
wind = wdm.series(dsns['WIND'])

fig, ax = plt.subplots()
wind_nldas.shift(-6).plot(label='nldas', ax=ax)
plt.legend()
wind.loc[wind.index.intersection(wind_nldas.index)].plot(label='RESPEC', ax=ax)
plt.legend()

#%% Air Temperature
# ── Step 3: disaggregate (source-agnostic Series in, Series out) ─
atem = derivations.kelvin_to_fahrenheit(nldas_data['Tair'])
atem = weighted_mean_timeseries(atem.to_dataset(), polygon.geometry.iloc[0])['temp_f']
atem_wdm = wdm.series(dsns['ATEM'])

fig, ax = plt.subplots()
atem.shift(-6).plot(label='nldas', ax=ax)
atem_wdm.loc[atem.index.intersection(atem_wdm.index)].plot(label='RESPEC', ax=ax)


#%% Solar Radiation
# ── Step 3: disaggregate (source-agnostic Series in, Series out) ─
solar = nldas_data['SWdown']

solar = weighted_mean_timeseries(solar.to_dataset(), polygon.geometry.iloc[0])['SWdown']
#convert to langleys
solar = solar/11.63
solar_wdm = wdm.series(dsns['SOLR'])

fig, ax = plt.subplots()
solar.shift(-6).plot(label='nldas', ax=ax)
solar_wdm.loc[solar.index.intersection(solar_wdm.index)].plot(label='RESPEC', ax=ax)
plt.legend()
#%% Dewpoint Temperature

dewpoint_arm = derivations.dewpoint_august_roche_magnus(
                 temperature=derivations.kelvin_to_celsius(nldas_data["Tair"]),
                 specific_humidity=nldas_data["Qair"],
                 pressure=nldas_data.get("PSurf", 101325))
dewpoint_arm = weighted_mean_timeseries(dewpoint_arm.to_dataset(), polygon.geometry.iloc[0])['dewpoint_c']
dewpoint_arm = dewpoint_arm*9/5 + 32 # convert to Fahrenheit for comparison with WDM

dewp = derivations.dewpoint_from_specific_humidity(
                 nldas_data["Qair"], pressure=nldas_data.get("PSurf"))
dewp = weighted_mean_timeseries(dewp.to_dataset(), polygon.geometry.iloc[0])['dewpoint_c']
dewp = dewp*9/5 + 32 # convert to Fahrenheit for comparison with WDM
dewp_wdm = wdm.series(dsns['DEWP'])

fig, ax = plt.subplots()
dewp_wdm.loc[dewp.index.intersection(dewp_wdm.index)].plot(label='RESPEC', ax=ax)
#dewp.shift(-6).plot(label='nldas', ax=ax)
dewpoint_arm.shift(-6).plot(label='august-roche-magnus', ax=ax)

#%% Cloud Cover
#cloud_linear = weighted_mean_timeseries(narr_data["tcdc"].to_dataset(), polygon.geometry.iloc[0])['tcdc'].resample('D').mean()/10

defaults = (1.353, -3.231, 1.878)
cloud_thompson = derivations.cloud_cover_thompson(
                 nldas_data["SWdown"],
                 coeffs=defaults)
cloud_thompson = weighted_mean_timeseries(cloud_thompson.to_dataset(), polygon.geometry.iloc[0])['cloud_cover_fraction_thompson'].resample('D').mean()*100
#cloud_thompson = cloud_thompson.loc[cloud_thompson > 0].resample('D').mean()*10

cloud_davis = derivations.cloud_cover_davis(
                 nldas_data["SWdown"],
                 min_clearsky=10,
                 k = .6
                 )
cloud_davis = weighted_mean_timeseries(cloud_davis.to_dataset(), polygon.geometry.iloc[0])['cloud_cover_fraction_davis']
cloud_davis = cloud_davis.loc[cloud_davis > 0].shift(-6).resample('D').mean()*10


cloud_wdm = wdm.series(dsns['CLOU'])
fig, ax = plt.subplots()
cloud_wdm.loc[cloud_thompson.index.intersection(cloud_wdm.index)].plot(label='RESPEC', ax=ax)

#cloud_linear.plot(label='linear method', ax=ax)
cloud_davis.plot(label='davis method', ax=ax)
#cloud_thompson.plot(label='thompson method', ax=ax)
#cloud.plot(label='narr', ax=ax)
#%% PET
shortwave = nldas_data["SWdown"]
longwave = nldas_data["LWdown"]
temperature_c = derivations.kelvin_to_celsius(nldas_data["Tair"])
pressure = nldas_data.get("PSurf", 101325)  # Use surface pressure if available, otherwise assume standard atmospheric pressure
wind_speed = derivations.adjust_wind_height(derivations.wind_speed(nldas_data["Wind_E"], nldas_data["Wind_N"]))
wind_speed = derivations.wind_speed(nldas_data["Wind_E"], nldas_data["Wind_N"])
specific_humidity = nldas_data["Qair"]
dewpoint_c = derivations.dewpoint_august_roche_magnus(
                 temperature=temperature_c,
                 specific_humidity=specific_humidity,
                 pressure=pressure)
penman_monteith = derivations.penman_monteith_asce(
                shortwave,
                longwave,
                temperature_c,
                dewpoint_c,
                wind_speed,
                pressure)
                #cn = 60,
                #cd_day = .24,
                #cd_night = 1.7) # convert from Pa to kPa for comparison with WDM
penman_monteith = weighted_mean_timeseries(penman_monteith.to_dataset(), polygon.geometry.iloc[0])['pet_penman_monteith_hourly_mm']/25.4


pet_penman_knb = derivations.penman_knb(temperature_c.resample(time="1D").mean(),
                                            temperature_c.resample(time="1D").min(),
                                            temperature_c.resample(time="1D").max(),
                                            dewpoint_c.resample(time="1D").mean(),
                                            wind_speed.resample(time="1D").mean(),
                                            (shortwave*.0036).resample(time="1D").sum())

pet_penman_knb_hourly = disaggregation.disaggregate_pevt(pet_penman_knb)

pet_penman_knb = weighted_mean_timeseries(pet_penman_knb.to_dataset(), polygon.geometry.iloc[0])['penman_knb']/25.4
pet_penman_knb_hourly = weighted_mean_timeseries(pet_penman_knb_hourly.to_dataset(), polygon.geometry.iloc[0])['pet_hourly_trapezoidal']/25.4


pet_oudin = derivations.oudin(temperature_c.shift(time = -6).resample(time="1D").mean())
pet_oudin = weighted_mean_timeseries(pet_oudin.to_dataset(), polygon.geometry.iloc[0])['pet_oudin']/25.4

 # convert from W/m2 to MJ/m2/day for comparison with the Penman Pan formula

pet_wdm = wdm.series(dsns['PEVT'])
pet_wdm = pet_wdm.loc[pet_wdm.index.intersection(penman_monteith.index)]
pet_wdm = pet_wdm.rename(columns={'values': 'Target (RESPEC)'})
fig, ax = plt.subplots()
(pet_wdm).plot(label='RESPEC', ax=ax)
pet_penman_knb_hourly.plot(label = 'PET KBN', ax = ax)
# penman_monteith.shift(-6).plot(label='Penman-Monteith', ax=ax)


fig, ax = plt.subplots()
pet_wdm.resample('D').sum().plot(label='RESPEC (daily)', ax=ax)
pet_penman_knb_hourly.resample('D').sum().plot(label='Kohler (daily)', ax=ax)
(pet_penman_knb).plot(label='Kohler', ax=ax)
#pet_oudin.plot(label='Penman Montieth (daily)', ax=ax)
#penman_monteith.shift(-6).resample('D').sum().plot(label='Penman-Monteith', ax=ax)
#%%
ds_list = [
    temperature_c.resample(time="1D").mean().rename("temp_mean"),
    temperature_c.resample(time="1D").min().rename("temp_min"),
    temperature_c.resample(time="1D").max().rename("temp_max"),
    dewpoint_c.resample(time="1D").mean().rename("dewpoint_mean"),
    wind_speed.resample(time="1D").mean().rename("wind_speed_mean"),
    (shortwave * .0036).resample(time="1D").sum().rename("shortwave_sum")
]

ds = xr.merge(ds_list)
inputs = weighted_mean_timeseries(ds, polygon.geometry.iloc[0])

target_pet = pet_wdm.resample('D').sum()['Target (RESPEC)']
calculated_pet = pet_penman_knb
ratio = target_pet / calculated_pet        # multiplicative component
residual = target_pet - calculated_pet     # additive component


#%%
matching_day = '2018-07-26'
mismatch_day = '2018-07-26'


ds = xr.concat([temperature_c.shift(time = -6).resample(time="1D").mean(),
      temperature_c.shift(time = -6).resample(time="1D").min(),
      temperature_c.shift(time = -6).resample(time="1D").max().rename('mean'),
      dewpoint_c.shift(time = -6).resample(time="1D").mean(),
      wind_speed.shift(time = -6).resample(time="1D").mean(),
      (shortwave*.0036).shift(time = -6).resample(time="1D").sum()], dim = 'time')

print(
{'temp_c' : temperature_c.shift(time = -6).resample(time="1D").max().sel(time = mismatch_day).mean().values,
'dewp_c' : dewpoint_c.shift(time = -6).resample(time="1D").mean().sel(time = mismatch_day).mean().values,
'wind_speed': wind_speed.shift(time = -6).resample(time="1D").mean().sel(time = mismatch_day).mean().values,
'shortwave' : (shortwave*.0036).shift(time = -6).resample(time="1D").sum().sel(time = mismatch_day).mean().values,
'pet_result': .18,
'pet_target': .217,
'date': '2018-07-26'})
# #pevt_penman.shift(-6).resample('D').sum().plot(label='Penman Pan (daily)', ax=ax)
# #penman_monteith.shift(-6).resample('D').sum().plot(label='Penman Montieth (daily)', ax=ax)
# pevt_kohler.shift(-6).resample('D').sum().plot(label='Kohler', ax=ax)
# ds_station['Penman PET'].plot(label='Station Penman PET', ax=ax)
# plt.legend()
#%% Plot PET
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Set up a wider, more professional figure canvas
fig, ax = plt.subplots(figsize=(14, 6))
# Select a nice 10-day summer window
zoom_slice = slice('2018-07-01', '2018-07-10')
# 2. Plot the data with distinct colors, line widths, and transparency (alpha)
# Make the target (RESPEC) bold and black/dark gray so it stands out as the baseline
pet_wdm[zoom_slice].plot(ax=ax, label='Target (RESPEC)', color='black', linewidth=2, alpha=0.8)

# Use complementary colors for the methods you are testing
pevt_penman[zoom_slice].shift(-6).plot(ax=ax, label='Penman Pan', color='#1f77b4', linewidth=1.5, alpha=0.7)
penman_monteith[zoom_slice].shift(-6).plot(ax=ax, label='Penman-Monteith', color='#2ca02c', linewidth=1.5, alpha=0.7)
pevt_kohler[zoom_slice].shift(-6).plot(ax=ax, label='Kohler (Disaggregated)', color='#ff7f0e', linewidth=1.5, alpha=0.7)

# 3. Add clean labels and a title
ax.set_title('Comparison of Hourly Evapotranspiration Methods', fontsize=14, fontweight='bold', pad=15)
ax.set_ylabel('Potential Evapotranspiration (mm/hr)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)

# 4. Add a subtle grid to help the eye track values
ax.grid(True, which='major', color='gray', linestyle='--', alpha=0.4)

# 5. Move the legend outside the plot so it doesn't cover your data
ax.legend(
    title='PET Method', 
    title_fontsize='11',
    loc='upper left', 
    bbox_to_anchor=(1.02, 1), 
    frameon=False # Removes the box around the legend for a cleaner look
)

# Removes extra white space and ensures the legend fits in the exported image
plt.tight_layout() 
plt.show()
#%% Download Gauge Data


ds_station = ndawn.get_station_data_daily(95,variables = ['ddtpetp','ddtpetjh'],as_dataset=False)
ds_station = ds_station.rename({'Latitude': 'lat', 'Longitude': 'lon'})
ds_station.set_index('time', inplace=True)

#%%Plot station data

# 1. Set up a wider, more professional figure canvas
fig, ax = plt.subplots()
# Select a nice 10-day summer window
zoom_slice = slice('2018-01-01', '2018-12-31')
# 2. Plot the data with distinct colors, line widths, and transparency (alpha)
# Make the target (RESPEC) bold and black/dark gray so it stands out as the baseline
pet_wdm.resample('D').sum()[zoom_slice].plot(ax=ax, label='Target (RESPEC)', color='red', linewidth=2, alpha=0.8)
pet_oudin[zoom_slice].resample('D').sum().plot(ax=ax, label='Oudin', color="#bb4d22", linewidth=1.5, alpha=0.7)
pet_penman_knb[zoom_slice].resample('D').sum().plot(label='Penman Kohler',color="#147338", linewidth=1.5, alpha=0.7)
ds_station['Penman PET'][zoom_slice].plot(ax=ax, label='Station Penman PET', color="#072F4C", linewidth=1.5, alpha=0.7)
ds_station['Jensen-Haise PET'][zoom_slice].plot(ax=ax, label='Station Jensen-Haise PET', color='#1f77b4', linewidth=1.5, alpha=0.7)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Daily Potential Evapotranspiration (in)', fontsize=12)
plt.title('Daily Potential Evapotranspiration: RESPEC vs. Station Penman PET', fontsize=14, fontweight='bold', pad=15)
ax.grid(True, which='major', color='gray', linestyle='--', alpha=0.4)
ax.legend(title='PET Method', title_fontsize='11', loc='upper left', frameon=True)










#%%










#%%






































station_penman_monteith = derivations.pet_penman_monteith_hourly(ds_station['hdt']*9/5 + 32, # convert to Fahrenheit
                                                         ds_station['hdws']*0.44704, # convert from mph to m/s for the calculation, then back to mph for comparison with WDM
                                                         ds_station['hdsr'], #lys is already in W/m2, which is what the formula expects, but WDM stores it as langleys, so we don't need to convert for comparison with WDM
                                                         ds_station['hddp']*9/5 + 32, # convert to Fahrenheit
                                                         ds_station.attrs['Elevation']*3.28084) # convert from meters to feet for comparison with WDM

#rename lat/lon coordinates
station_penman = derivations.pet_penman_pyet_hourly(ds_station['hdsr'],
                                                     ds_station['hdt']*9/5 + 32,
                                                     ds_station['hddp']*9/5 + 32,
                                                     ds_station['hdws']*0.44704,
                                                     ds_station.attrs['Elevation']*3.28084)

#create polygon from ds lat/long coords
polygon_station = gpd.GeoDataFrame(geometry=gpd.points_from_xy(ds_station['Longitude'].values, ds_station['Latitude'].values), crs="EPSG:4326").buffer(.1)
bounds = polygon_station.total_bounds


start_date = '2019-01-01'
end_date = '2019-12-31'
dates = [date.date().strftime("%Y-%m-%d") for date in pd.date_range(start_date,end_date)]
nldas_data = []
for date in dates:
    nldas_data.append(nldas.fetch_nldas(date = date, cache_dir=nldas_cache_dir, bounds=BoundingBox(west=bounds[0], south=bounds[1], east=bounds[2], north=bounds[3])))
nldas_data = xr.concat(nldas_data, dim="time")
shortwave = nldas_data["SWdown"]
longwave = nldas_data["LWdown"]
temperature_c = derivations.kelvin_to_celsius(nldas_data["Tair"])
pressure = nldas_data.get("PSurf", 101325)  # Use surface pressure if available, otherwise assume standard atmospheric pressure
wind_speed = derivations.wind_speed(nldas_data["Wind_E"], nldas_data["Wind_N"])
specific_humidity = nldas_data["Qair"]
dewpoint_c = derivations.dewpoint_august_roche_magnus(
                 temperature=temperature_c,
                 specific_humidity=specific_humidity,
                 pressure=pressure)
penman_monteith = derivations.pet_penman_monteith_hourly(temperature_c,
                                                         wind_speed,
                                                         shortwave,
                                                         dewpoint_c,
                                                         ds_station.attrs['Elevation'])
penman_monteith = weighted_mean_timeseries(penman_monteith.to_dataset(), polygon.geometry.iloc[0])['pet_penman_monteith_hourly_mm']



temperature_c = derivations.kelvin_to_celsius(nldas_data["Tair"])
temperature_c = weighted_mean_timeseries(temperature_c.to_dataset(), polygon_station.geometry.iloc[0])['temp_c']

fig, ax = plt.subplots()
temperature_c.shift(-6).plot(label='nldas temp', ax=ax)
#convert degF to degC for comparison with NLDAS-derived temperature
((ds_station['hdt'] - 32) * 5/9).plot(label='station temp', ax=ax)
plt.legend()

station_penman_monteith.plot(label='station', ax=ax)
penman_monteith.shift(-6).plot(label='nldas', ax=ax)  


# %%
"""Monte Carlo analysis of cascade disaggregation uncertainty."""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class CascadeDayReport:
    """Uncertainty statistics for a single cascade day from N realizations."""
    timestamp: pd.Timestamp
    prism_total_mm: float

    # Per-hour statistics across realizations (24 values)
    hourly_mean: pd.Series
    hourly_std: pd.Series
    hourly_cv: pd.Series
    hourly_q05: pd.Series
    hourly_q25: pd.Series
    hourly_q50: pd.Series
    hourly_q75: pd.Series
    hourly_q95: pd.Series

    # Scalar summaries across realizations
    peak_hour_mean: float
    peak_hour_std: float
    peak_intensity_mean: float
    peak_intensity_std: float
    wet_hours_mean: float
    wet_hours_std: float

    # Raw ensemble for this day: shape (n_realizations, n_day_hours)
    ensemble: np.ndarray | None


@dataclass
class CascadeUncertaintyReport:
    """Collection of per-day uncertainty reports from N cascade realizations."""
    n_realizations: int
    n_cascade_days: int
    days: dict[pd.Timestamp, CascadeDayReport]


def run_cascade_monte_carlo(
    daily_total: pd.Series,
    hourly_pattern: pd.Series,
    cascade_method: str = "molnar_burlando",
    n_realizations: int = 500,
    wet_threshold: float = 0.1,
    store_ensemble: bool = True,
) -> CascadeUncertaintyReport:
    """Run N cascade realizations and compute per-day uncertainty statistics.

    Only days where PRISM > 0 and NLDAS == 0 (cascade days) produce
    variability.  Each cascade day gets its own :class:`CascadeDayReport`.
    Days with NaN in either input are skipped.

    Parameters
    ----------
    daily_total:
        Daily precipitation totals (PRISM) in mm.
    hourly_pattern:
        Hourly precipitation (NLDAS) in mm.
    cascade_method:
        ``"molnar_burlando"`` or ``"olsson"``.
    n_realizations:
        Number of Monte Carlo runs.
    wet_threshold:
        Minimum mm/h to count as a wet hour.
    store_ensemble:
        If True, store the per-day (n_realizations, n_day_hours) array.
    """
    from met_timeseries.disaggregation import disaggregate_precipitation

    hourly_index = hourly_pattern.index
    n_hours = len(hourly_index)
    ensemble = np.zeros((n_realizations, n_hours))

    # Identify cascade days
    cascade_days = []
    for timestamp, prism_total in daily_total.items():
        if pd.isna(prism_total):
            continue

        day_range = pd.date_range(
            start=timestamp,
            end=timestamp + pd.Timedelta(hours=23),
            freq="h",
        )
        matched_hours = hourly_pattern.index.intersection(day_range)

        if len(matched_hours) == 0:
            continue

        nldas_day = hourly_pattern[matched_hours]
        if nldas_day.isna().any():
            continue

        if prism_total > 0 and float(nldas_day.sum()) == 0:
            cascade_days.append(timestamp)

    print(f"Identified {len(cascade_days)} cascade days out of {len(daily_total)} total days")
    print(f"Running {n_realizations} realizations...")

    for i in range(n_realizations):
        if (i + 1) % 100 == 0:
            print(f"  realization {i + 1}/{n_realizations}")
        result, _ = disaggregate_precipitation_hybrid(
            daily_total, hourly_pattern, cascade_method=cascade_method,
        )
        ensemble[i, :len(result)] = result.values[:n_hours]

    # Build per-day reports
    days = {}
    for timestamp in cascade_days:
        prism_total = float(daily_total[timestamp])

        day_range = pd.date_range(
            start=timestamp,
            end=timestamp + pd.Timedelta(hours=23),
            freq="h",
        )
        matched_hours = hourly_index.intersection(day_range)
        hour_indices = np.array([hourly_index.get_loc(h) for h in matched_hours])

        if len(hour_indices) == 0:
            continue

        # Slice ensemble for this day: (n_realizations, n_day_hours)
        day_ensemble = ensemble[:, hour_indices]

        # Per-hour stats
        h_mean = pd.Series(day_ensemble.mean(axis=0), index=matched_hours)
        h_std = pd.Series(day_ensemble.std(axis=0), index=matched_hours)
        h_cv = pd.Series(
            np.where(h_mean > 0, h_std / h_mean, 0.0),
            index=matched_hours,
        )

        quantiles = np.percentile(day_ensemble, [5, 25, 50, 75, 95], axis=0)
        h_q05 = pd.Series(quantiles[0], index=matched_hours)
        h_q25 = pd.Series(quantiles[1], index=matched_hours)
        h_q50 = pd.Series(quantiles[2], index=matched_hours)
        h_q75 = pd.Series(quantiles[3], index=matched_hours)
        h_q95 = pd.Series(quantiles[4], index=matched_hours)

        # Scalar summaries across realizations
        peak_idxs = np.argmax(day_ensemble, axis=1)
        peak_vals = np.array([day_ensemble[i, p] for i, p in enumerate(peak_idxs)])
        wet_counts = np.sum(day_ensemble >= wet_threshold, axis=1)

        days[timestamp] = CascadeDayReport(
            timestamp=timestamp,
            prism_total_mm=prism_total,
            hourly_mean=h_mean,
            hourly_std=h_std,
            hourly_cv=h_cv,
            hourly_q05=h_q05,
            hourly_q25=h_q25,
            hourly_q50=h_q50,
            hourly_q75=h_q75,
            hourly_q95=h_q95,
            peak_hour_mean=float(peak_idxs.mean()),
            peak_hour_std=float(peak_idxs.std()),
            peak_intensity_mean=float(peak_vals.mean()),
            peak_intensity_std=float(peak_vals.std()),
            wet_hours_mean=float(wet_counts.mean()),
            wet_hours_std=float(wet_counts.std()),
            ensemble=day_ensemble if store_ensemble else None,
        )

    return CascadeUncertaintyReport(
        n_realizations=n_realizations,
        n_cascade_days=len(cascade_days),
        days=days,
    )
# %%

"""Visualization utilities for cascade uncertainty analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# 1. Fan chart: single cascade day with uncertainty envelope
# ---------------------------------------------------------------------------

def plot_day_fan_chart(day: CascadeDayReport, ax: plt.Axes | None = None) -> plt.Figure:
    """Hourly uncertainty envelope for one cascade day.

    Shows median, IQR, and 90% interval from the ensemble.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    hours = np.arange(len(day.hourly_mean))
    hour_labels = day.hourly_mean.index

    ax.fill_between(hours, day.hourly_q05, day.hourly_q95,
                     alpha=0.15, color="steelblue", label="90% interval")
    ax.fill_between(hours, day.hourly_q25, day.hourly_q75,
                     alpha=0.35, color="steelblue", label="IQR")
    ax.plot(hours, day.hourly_q50, color="steelblue", lw=2, label="Median")

    ax.set_title(f"{day.timestamp.date()} — PRISM: {day.prism_total_mm:.1f} mm")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Precipitation (mm/h)")
    ax.set_xticks(hours[::3])
    ax.set_xticklabels([h.strftime("%H:%M") for h in hour_labels[::3]], rotation=45)
    ax.legend(loc="upper right")
    ax.set_xlim(0, len(hours) - 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Spaghetti plot: overlay individual realizations for one day
# ---------------------------------------------------------------------------

def plot_day_spaghetti(
    day: CascadeDayReport,
    n_traces: int = 50,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Overlay N individual cascade realizations for one day."""
    if day.ensemble is None:
        raise ValueError("Ensemble not stored — rerun with store_ensemble=True")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    hours = np.arange(day.ensemble.shape[1])
    hour_labels = day.hourly_mean.index

    indices = np.random.choice(day.ensemble.shape[0], size=min(n_traces, day.ensemble.shape[0]), replace=False)
    for i in indices:
        ax.plot(hours, day.ensemble[i], color="steelblue", alpha=0.1, lw=0.8)

    ax.plot(hours, day.hourly_q50, color="black", lw=2, label="Median")

    ax.set_title(f"{day.timestamp.date()} — {n_traces} realizations")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Precipitation (mm/h)")
    ax.set_xticks(hours[::3])
    ax.set_xticklabels([h.strftime("%H:%M") for h in hour_labels[::3]], rotation=45)
    ax.legend()
    ax.set_xlim(0, len(hours) - 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 3. Peak hour histogram: where does the storm land across realizations?
# ---------------------------------------------------------------------------

def plot_peak_hour_histogram(day: CascadeDayReport, ax: plt.Axes | None = None) -> plt.Figure:
    """Distribution of the hour containing peak intensity across realizations."""
    if day.ensemble is None:
        raise ValueError("Ensemble not stored — rerun with store_ensemble=True")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))
    else:
        fig = ax.get_figure()

    peak_hours = np.argmax(day.ensemble, axis=1)
    hour_labels = day.hourly_mean.index

    ax.hist(peak_hours, bins=np.arange(day.ensemble.shape[1] + 1) - 0.5,
            color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_title(f"{day.timestamp.date()} — Peak hour distribution")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Count")
    ax.set_xticks(range(0, len(hour_labels), 3))
    ax.set_xticklabels([h.strftime("%H:%M") for h in hour_labels[::3]], rotation=45)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Summary dashboard: fan + spaghetti + peak histogram for one day
# ---------------------------------------------------------------------------

def plot_day_dashboard(day: CascadeDayReport, n_traces: int = 50) -> plt.Figure:
    """Three-panel dashboard for a single cascade day."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    plot_day_fan_chart(day, ax=axes[0])
    plot_day_spaghetti(day, n_traces=n_traces, ax=axes[1])
    plot_peak_hour_histogram(day, ax=axes[2])

    fig.suptitle(
        f"{day.timestamp.date()} — PRISM: {day.prism_total_mm:.1f} mm  |  "
        f"Peak: {day.peak_intensity_mean:.2f}±{day.peak_intensity_std:.2f} mm/h  |  "
        f"Wet hours: {day.wet_hours_mean:.1f}±{day.wet_hours_std:.1f}",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Cross-day comparison: CV and peak spread across all cascade days
# ---------------------------------------------------------------------------

def plot_cascade_days_summary(report: CascadeUncertaintyReport) -> plt.Figure:
    """Compare uncertainty metrics across all cascade days."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    days = list(report.days.values())
    dates = [d.timestamp for d in days]
    prism_totals = [d.prism_total_mm for d in days]
    max_cvs = [float(d.hourly_cv.max()) for d in days]
    peak_stds = [d.peak_hour_std for d in days]
    intensity_stds = [d.peak_intensity_std for d in days]
    wet_hour_stds = [d.wet_hours_std for d in days]

    # Max CV vs PRISM total
    axes[0, 0].scatter(prism_totals, max_cvs, c="steelblue", edgecolor="white", s=50)
    axes[0, 0].set_xlabel("PRISM daily total (mm)")
    axes[0, 0].set_ylabel("Max hourly CV")
    axes[0, 0].set_title("Hourly variability vs. daily total")

    # Peak hour spread vs PRISM total
    axes[0, 1].scatter(prism_totals, peak_stds, c="coral", edgecolor="white", s=50)
    axes[0, 1].set_xlabel("PRISM daily total (mm)")
    axes[0, 1].set_ylabel("Peak hour std (hours)")
    axes[0, 1].set_title("Storm timing uncertainty vs. daily total")

    # Peak intensity spread over time
    axes[1, 0].bar(range(len(days)), intensity_stds, color="steelblue", alpha=0.7)
    axes[1, 0].set_xlabel("Cascade day index")
    axes[1, 0].set_ylabel("Peak intensity std (mm/h)")
    axes[1, 0].set_title("Peak intensity uncertainty by day")

    # Wet hours spread over time
    axes[1, 1].bar(range(len(days)), wet_hour_stds, color="coral", alpha=0.7)
    axes[1, 1].set_xlabel("Cascade day index")
    axes[1, 1].set_ylabel("Wet hours std")
    axes[1, 1].set_title("Storm duration uncertainty by day")

    fig.suptitle(
        f"{report.n_cascade_days} cascade days — {report.n_realizations} realizations each",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Grid of fan charts: all cascade days at a glance
# ---------------------------------------------------------------------------

def plot_all_days_grid(
    report: CascadeUncertaintyReport,
    cols: int = 4,
) -> plt.Figure:
    """Small-multiple fan charts for every cascade day."""
    days = list(report.days.values())
    n = len(days)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), squeeze=False)

    for idx, day in enumerate(days):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        hours = np.arange(len(day.hourly_mean))

        ax.fill_between(hours, day.hourly_q05, day.hourly_q95, alpha=0.15, color="steelblue")
        ax.fill_between(hours, day.hourly_q25, day.hourly_q75, alpha=0.35, color="steelblue")
        ax.plot(hours, day.hourly_q50, color="steelblue", lw=1.5)

        ax.set_title(f"{day.timestamp.date()}\n{day.prism_total_mm:.1f} mm", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"Cascade uncertainty — {report.n_realizations} realizations", fontsize=11)
    fig.tight_layout()
    return fig
# %%
