from pathlib import Path

def get_filepaths(cache_dir, year):
    files = list(Path(cache_dir).glob("*.nc"))
    selected_files = [file for file in files if int(file.stem.split('_')[-1][0:4]) == year]
    return selected_files

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