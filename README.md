# met-timeseries

Meteorological timeseries downloading and processing for hydrologic models.

## Overview

`met-timeseries` downloads gridded climate data (NLDAS-2, PRISM) and
aggregates it to user-defined **metzone** polygons derived from catchment
vector files, producing per-metzone hourly timeseries stored as Parquet files.

