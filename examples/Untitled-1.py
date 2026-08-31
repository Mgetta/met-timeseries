#%% Imports
from collections import defaultdict
from datetime import datetime

import pandas as pd
from met_timeseries.sources.nldas import BoundingBox
from met_timeseries.sources import nldas, prism, narr
from met_timeseries.utils import mem_gb, hdf5WDM
import geopandas as gpd
from pathlib import Path
import gc
#%% Load Target Metzone
model_name = 'LOWUS'
gdf = gpd.read_file("C:/Users/mfratki/Documents/github/hspf_spatial/files/env_watershed_hspfmodel_catch.gdb", layer="watershed_hspfmodel_catchments", driver='OpenFileGDB')
statewide = gpd.read_file(Path('C:/Users/mfratki/Documents/github/hspf_spatial/files/statewide_subwatersheds.gpkg')).to_crs("EPSG:4326")
catchments = gdf.query('sam_name == @model_name')


CATCHMENT_PATH = catchments.to_crs("EPSG:4326")
# wdm_path = Path('C:/Users/mfratki/Documents/Projects/Tests/LOWUS/model/LOWUS_Met_2024.hdf5')
# wdm = hdf5WDM(wdm_path)

hydrozone = 15 #NDAWN station_id 95 (Williams)
# dsns = {'PEVT': 152,
#         'PREC': 151,
#         'ATEM': 154,
#         'SOLR': 155,
#         'WIND': 156,
#         'DEWP': 157,
#         'CLOU': 159}

polygon = CATCHMENT_PATH[CATCHMENT_PATH['hydrozone'] == hydrozone].dissolve()
bounds = polygon.bounds
bounds = BoundingBox(
    west=bounds.minx.min(),
    south=bounds.miny.min(),
    east=bounds.maxx.max(),
    north=bounds.maxy.max(),)





#%% Download PRISM and NLDAS
# ── Step 0: fetch grids (before spatial aggregation) ──────────
prism_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\800m"
prism_resolution = '800m'
nldas_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas"
mrms_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\mrms"
narr_cache_dir = "C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\narr"



start_date = "1994-12-31" 
end_date = "2026-01-01"
# Cached dates
prism_files = list(Path(prism_cache_dir).glob("*.nc"))
nldas_files = list(Path(nldas_cache_dir).glob("*.nc"))

def get_missing_dates(files, start_date, end_date):
    cached_dates = set([file.stem.split("_")[-1] for file in files])
    # 2. Generate the expected dates in the exact string format as your filenames
    expected_dates = set(pd.date_range(start=start_date, end=end_date).strftime("%Y%m%d"))
    # 3. Find the missing dates using a mathematical set difference
    missing_dates = expected_dates - cached_dates
    return missing_dates

def get_files_by_mod_date(directory_path: str) -> dict:
    """
    Scans a directory and returns a dictionary grouping files by their modification date.
    
    Args:
        directory_path (str): The path to the directory to scan.
        
    Returns:
        dict: Keys are date strings ('YYYY-MM-DD'), values are lists of file names.
    """
    # Using defaultdict makes it easy to append to lists without checking if the key exists yet
    files_by_date = defaultdict(list)
    target_dir = Path(directory_path)
    
    # Check if the directory actually exists
    if not target_dir.is_dir():
        raise NotADirectoryError(f"The path '{directory_path}' is not a valid directory.")

    # Iterate through all items in the directory
    for item in target_dir.iterdir():
        # Make sure we are only looking at files, ignoring subdirectories
        if item.is_file():
            # Get the modification timestamp
            mod_timestamp = item.stat().st_mtime
            
            # Convert the timestamp to a formatted date string (e.g., '2026-06-04')
            mod_date = datetime.fromtimestamp(mod_timestamp).strftime('%Y-%m-%d')
            
            # Append the file name to the corresponding date's list
            files_by_date[mod_date].append(item.resolve())
            
            # Note: If you want the full file path instead of just the name, 
            # change `item.name` to `str(item)` or `item.resolve()`.

    # Convert back to a standard dictionary before returning
    return dict(files_by_date)

files = get_files_by_mod_date(nldas_cache_dir)
len_lat = 58
len_lon = 70
dates_to_download = []
for key,value in files.items():
    ds = xr.open_dataset(value[-1])
    if len(ds.lat) != len_lat or len(ds.lon) != len_lon:
        print(f"{value[0].stem} has {len(ds.lat)} lat and {len(ds.lon)} lon")
        dates = [pd.to_datetime(date.stem.split("_")[-1], format="%Y%m%d").strftime("%Y-%m-%d") for date in value]
        dates_to_download.append(dates)
missing_dates = [date for sublist in dates_to_download for date in sublist] 

missing_dates = sorted(list(get_missing_dates(prism_files, start_date, end_date) | get_missing_dates(nldas_files, start_date, end_date)))
missing_dates = pd.to_datetime(missing_dates,format = "%Y%m%d").strftime("%Y-%m-%d").tolist()
# potentially missing 2002-07-16 for nldas, got a skipping granule warning.
for i, date in enumerate(missing_dates):
    print(f"Processing date {i+1}/{len(missing_dates)}: {date} | Memory: {mem_gb():.2f} GB")
    prism_data = prism.fetch_prism(date=date, cache_dir=prism_cache_dir, resolution=prism_resolution,bounds = bounds)
    nldas_data = nldas.fetch_nldas(date, nldas_cache_dir,bounds=bounds,overwrite_cache=True)
    del nldas_data
    del prism_data
    gc.collect()
    #mrms_data = mrms.fetch_mrms(date,mrms_cache_dir,bounds=bounds)

    #prism_ds.append(prism_data)
    #nldas_ds.append(nldas_data)
    #mrms_ds.append(mrms_data)

    if i % 30 == 0:  # every ~month
        print(f"Day {i:3d} | {date} | Memory: {mem_gb():.2f} GB | ")

#%% Download NARR data