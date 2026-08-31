#%% imports
import math

import xarray as xr
from pathlib import Path
import zarr
import numpy as np
#from dask.distributed import Client
#%%

'''
Some notes on the process:
1. Storing the 30 year cache to a zarr required batch loading as I did not have enough RAM to open the entire multi-year dataset at once, even with the optimized open_mfdataset kwargs.
2. Netcdf4 files must have identical coordinate dimensions. ran into issues with different lat/lon lengths and some instances where an hour of the NLDAS data was missing. This is fine when opening individually, but when writing to Zarr or opening using xr.open_mfdataset it crashes due to misaligned time coordinate. I had to write a quick audit function to identify and re download the bad files before the final zarr conversion.
    - Note that this does not mean that the coordinate are correct just that they have the same shape. I did not check the actual values of lat/lon, just that they had the same shape. I spot checked a few files and they looked correct, but there may be some drift in the grid over time that I did not catch.



'''
# def main():
#     # Setup client here
#     #client = Client(n_workers=4, threads_per_worker=1, memory_limit="3GB")
#     #print(client.dashboard_link)
def mem_gb():
    """Return current process RSS in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def get_filepaths(cache_dir, year):
    files = list(Path(cache_dir).glob("*.nc"))
    selected_files = [file for file in files if int(file.stem.split('_')[-1][0:4]) == year]
    return selected_files




prism_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\800m") 
nldas_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas")
files = list(Path(nldas_cache_dir).glob("*.nc"))
nldas_ds =     ds = xr.open_mfdataset(
        files, 
        chunks={},          # Start with no chunking to speed up metadata reading
        parallel=True,      # Crucial: multithreads the metadata crawl
        concat_dim="time", 
        combine="nested", 
        coords="minimal",    # Only read coordinate variables during the initial metadata crawl
        compat="override",   # Skip the expensive compatibility checks since we trust our cache
        join="exact"        # Explicitly tells xarray to pad missing variables with NaNs
        # Notice we removed coords="minimal", data_vars="minimal", and compat="override"
    )

# files = list(Path(prism_cache_dir).glob("*.nc"))
# output_zarr = str(prism_cache_dir.parent /"prism.zarr")
# start_date = '12-31-1994'
# end_date = '12-31-1994'
# time_chunks = 90
# time_steps_per_file = 1
# batch_multiplier = 4


files = list(Path(nldas_cache_dir).glob("*.nc"))
output_zarr = str(nldas_cache_dir.parent /"nldas.zarr")
time_chunks = 90*24
time_steps_per_file = 24
batch_multiplier = 4

files_per_chunk = time_chunks // time_steps_per_file
batch_size = files_per_chunk * batch_multiplier
total_batches = math.ceil(len(files) / batch_size)

for i in range(0, len(files), batch_size):
        batch_files = files[i : i + batch_size]
        is_first_write = (i == 0)
        
        print(f"Processing batch {i//batch_size + 1}/{total_batches}...")
        
        # 3. Open only this specific batch
        # Memory stays low because join="outer" only looks at ~360 files instead of 10,000
        ds = xr.open_mfdataset(
            batch_files, 
            chunks={},          
            parallel=True,      
            concat_dim="time", 
            combine="nested", 
            join="exact",
            engine="h5netcdf" # Highly recommended for parallel reads
        )
        
        # 4. Apply your chunking
        ds = ds.chunk({"time": time_chunks, "lat": -1, "lon": -1})

        print('Batch loaded and chunked. Now writing to Zarr...')
        # 5. Write or Append to Zarr
        if is_first_write:
            # We turn off consolidation during the loop to avoid I/O overhead
            ds.to_zarr(output_zarr, mode="w", consolidated=False)
        else:
            ds.to_zarr(output_zarr, mode="a", append_dim="time", consolidated=False)
            
        # 6. Explicitly free the RAM before the next loop
        ds.close()




times = []
for file in files:
    # Read the metadata headers without loading the heavy data arrays
    ds = xr.open_dataset(file, engine="h5netcdf", chunks={})
    times.append(list(ds.time.values))
times = np.concatenate(times)

np.max(np.diff(times)/ np.timedelta64(1, 'h'))


prism_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\prism\\800m") 
nldas_cache_dir = Path("C:\\Users\\mfratki\\Documents\\github\\met-timeseries\\.cache\\nldas")

files = list(Path(nldas_cache_dir).glob("*.nc"))
nldas_ds =     ds = xr.open_mfdataset(
        files, 
        chunks={},          # Start with no chunking to speed up metadata reading
        parallel=True,      # Crucial: multithreads the metadata crawl
        concat_dim="time", 
        combine="nested", 
        join="exact"        # Explicitly tells xarray to pad missing variables with NaNs
        # Notice we removed coords="minimal", data_vars="minimal", and compat="override"
    )
nldas_ds = nldas_ds.chunk({"time": 24*90, "lat": -1, "lon": -1})
print(f"Writing fully chunked dataset to {nldas_cache_dir.parent /'nldas.zarr'}. This may take a while...")
nldas_ds.to_zarr( str(nldas_cache_dir.parent /"nldas.zarr"), mode="w", consolidated=True)
    
print("Done! Metadata is consolidated and the Zarr store is ready.")


years = list(range(1994, 2026))
for year in years:
    print(f"Processing year {year}...")
    files = get_filepaths(prism_cache_dir, year)
    ds_year = xr.open_mfdataset(
        files, 
        chunks={},          # Start with no chunking to speed up metadata reading
        parallel=True,      # Crucial: multithreads the metadata crawl
        concat_dim="time", 
        combine="nested", 
        join="outer"        # Explicitly tells xarray to pad missing variables with NaNs
        # Notice we removed coords="minimal", data_vars="minimal", and compat="override"
    )
    print(f"Year {year} has {len(ds_year['time'])} time steps.")
    ds_year.close()

bad_files = audit_netcdf_cache(prism_cache_dir)

files = list(Path(prism_cache_dir).glob("*.nc"))
prism_ds =     ds = xr.open_mfdataset(
        files, 
        chunks={},          # Start with no chunking to speed up metadata reading
        parallel=True,      # Crucial: multithreads the metadata crawl
        concat_dim="time", 
        combine="nested", 
        join="outer"        # Explicitly tells xarray to pad missing variables with NaNs
        # Notice we removed coords="minimal", data_vars="minimal", and compat="override"
    )

# Run it overnight
netcdf_to_zarr(str(nldas_cache_dir), str(nldas_cache_dir.parent /"nldas.zarr"), time_chunks=2160)

netcdf_to_zarr(str(prism_cache_dir), str(prism_cache_dir.parent /"prism.zarr"), time_chunks=90)

