#%% imports
import math

import xarray as xr
from pathlib import Path
import zarr
import numpy as np
#from dask.distributed import Client
#%%


# def main():
#     # Setup client here
#     #client = Client(n_workers=4, threads_per_worker=1, memory_limit="3GB")
#     #print(client.dashboard_link)


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

    
# Process in larger batches to speed up I/O, but strictly tied to the chunk multiple.
# E.g., for NLDAS: 720 / 24 = 30 files per chunk. 30 * 12 = 360 files per batch (roughly 1 year).

def netcdf_to_zarr(netcdf_dir: str, output_zarr: str, time_chunks: int):
    print(f"Crawling metadata in {netcdf_dir}...")
    
    # Grab all files across all years at once
    # 1. Open the ENTIRE multi-year cache lazily.
    # We use the optimized kwargs to prevent the 15-minute metadata crawl.
    ds = xr.open_mfdataset(
        files, 
        chunks={},          # Start with no chunking to speed up metadata reading
        parallel=True,      # Crucial: multithreads the metadata crawl
        concat_dim="time", 
        combine="nested", 
        join="outer"        # Explicitly tells xarray to pad missing variables with NaNs
        # Notice we removed coords="minimal", data_vars="minimal", and compat="override"
    )
    
    # 2. Apply your optimal Dask chunking across the continuous timeline
    # Dask will seamlessly chunk the entire 30-year span into perfect 2160-hour blocks.
    ds = ds.chunk({"time": time_chunks, "lat": -1, "lon": -1})
    
    # 3. Write it all to Zarr in one massively parallel shot.
    # Dask streams the data sequentially through RAM; you will not run out of memory.
    print(f"Writing fully chunked dataset to {output_zarr}. This may take a while...")
    ds.to_zarr(output_zarr, mode="w", consolidated=True)
    
    print("Done! Metadata is consolidated and the Zarr store is ready.")

def audit_netcdf_cache(cache_dir: str):
    folder = Path(cache_dir)
    files = list(folder.glob("*.nc"))
    print(f"Auditing {len(files)} files in {cache_dir}...")
    
    bad_files = []
    reference_lon = None
    reference_lat = None
    reference_file = None
    
    for file in files:
        try:
            # Read the metadata headers without loading the heavy data arrays
            ds = xr.open_dataset(file, engine="h5netcdf", chunks={})

            if len(ds.time) < 24:
                print(f"⚠️ WARNING: {file.name} has only {len(ds.time)} time steps. Expected 24.")
                bad_files.append(file)
                ds.close()
                continue
            
            # Check 1: Is 'time' a coordinate?
            if 'time' not in ds.coords:
                print(f"❌ ERROR: Missing 'time' coordinate in {file.name}")
                bad_files.append(file)
                ds.close()
                continue
                
            # Check 2: Do the spatial coordinates exist?
            if 'lon' not in ds.coords or 'lat' not in ds.coords:
                print(f"❌ ERROR: Missing 'lat' or 'lon' coordinates in {file.name}")
                bad_files.append(file)
                ds.close()
                continue
            
            current_lon = ds['lon'].values
            current_lat = ds['lat'].values
            
            # Establish the baseline using the first valid file
            if reference_lon is None or reference_lat is None:
                reference_lon = current_lon
                reference_lat = current_lat
                reference_file = file.name
            else:
                # Check 3A: Does the grid shape match? (Catches bounding box errors)
                if current_lon.shape != reference_lon.shape or current_lat.shape != reference_lat.shape:
                    print(f"❌ ERROR: Grid shape mismatch in {file.name}. "
                          f"Expected lon/lat shapes {reference_lon.shape}/{reference_lat.shape}.")
                    bad_files.append(file)
                    ds.close()
                    continue
                
                # Check 3B: Do the exact Longitude values match? (Catches floating point drift)
                if not np.array_equal(current_lon, reference_lon):
                    print(f"❌ ERROR: 'lon' values drifted from reference ({reference_file}) in {file.name}")
                    bad_files.append(file)
                    ds.close()
                    continue
                    
                # Check 3C: Do the exact Latitude values match?
                if not np.array_equal(current_lat, reference_lat):
                    print(f"❌ ERROR: 'lat' values drifted from reference ({reference_file}) in {file.name}")
                    bad_files.append(file)
                    ds.close()
                    continue
            
            ds.close()
            
        except Exception as e:
            print(f"💥 FATAL: Could not even open {file.name}. Error: {e}")
            bad_files.append(file)

    print("-" * 30)
    if not bad_files:
        print("✅ Audit complete. All files are structurally sound and perfectly aligned.")
    else:
        print(f"🚨 Audit complete. Found {len(bad_files)} corrupted or misaligned files.")
        
    return bad_files


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


if __name__ == '__main__':
    main()