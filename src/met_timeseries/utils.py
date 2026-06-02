import psutil
import os
from pathlib import Path
import h5py
import pandas as pd
import xarray as xr
from met_timeseries.sources.base import BoundingBox

def clip_dataset(
    ds: xr.Dataset, 
    bounds: BoundingBox, 
    lat_dim: str = "lat", 
    lon_dim: str = "lon"
) -> xr.Dataset:
    """
    Clip an xarray Dataset to a spatial bounding box with automatic edge padding.
    Safely handles both ascending and descending latitude coordinates.
    """
    lats = ds[lat_dim].values
    lons = ds[lon_dim].values

    # Pad by half a cell so we include cells whose edges overlap the bounds
    half_dy = abs(float(lats[1] - lats[0])) / 2
    half_dx = abs(float(lons[1] - lons[0])) / 2

    # Check if latitudes are descending (e.g., NLDAS) or ascending (e.g., PRISM)
    if lats[0] > lats[-1]:
        lat_slice = slice(bounds.north + half_dy, bounds.south - half_dy)
    else:
        lat_slice = slice(bounds.south - half_dy, bounds.north + half_dy)

    # Slice the dataset using a dictionary to support dynamic dimension names
    ds_clipped = ds.sel({
        lat_dim: lat_slice,
        lon_dim: slice(bounds.west - half_dx, bounds.east + half_dx),
    })
    
    return ds_clipped
def mem_gb():
    """Return current process RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


class hdf5WDM():
    def __init__(self,wdm_path:list):
        self.wdm_path = Path(wdm_path)
        
        with h5py.File(wdm_path, "r") as f:
            grp = f["/TIMESERIES/SUMMARY"]

            data = grp["table"][:]  # <-- slice the dataset, not the group
            df = pd.DataFrame(data)
            # Decode bytes to strings if needed
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            
            df.index = df['index'].str[2:].astype(int)
            self.summary = df


    def series(self,dsn):
        hdf5_name = self.summary.loc[dsn,'index']
        
        with h5py.File(self.wdm_path, "r") as f:
            grp = f[f"/TIMESERIES/{hdf5_name}"]

            data = grp["table"][:]  # <-- slice the dataset, not the group
            df = pd.DataFrame(data)
            # Decode bytes to strings if needed
            for col in df.columns:
                if df[col].dtype == object:
                    df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            
            df['index'] = pd.to_datetime(df['index'])
            df.set_index('index', inplace=True)
        return df
