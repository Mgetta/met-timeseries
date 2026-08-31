from pathlib import Path
import h5py
import pandas as pd

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
