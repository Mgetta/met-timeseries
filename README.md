# met-timeseries

A Python package for extracting hourly meteorological timeseries from gridded data products
(NLDAS-2, PRISM, etc.) and aggregating them to polygon areas of interest.

## Overview

`met-timeseries` is designed to support hydrologic modeling workflows by providing:

- **7 meteorological variables**: precipitation, solar radiation, wind speed, air temperature,
  dewpoint temperature, cloud cover, and potential evapotranspiration (PET)
- **Two gridded sources**: NLDAS-2 hourly forcing data and PRISM daily precipitation
- **Area-weighted aggregation**: zonal statistics using `exactextract` for ~1,020 polygons
  (between HUC8 and HUC12 size)
- **Long records**: data from 1996-01-01 to present in resumable monthly chunks
- **Future ensemble support**: designed to extend to gridMET, Daymet, ERA5

## Installation

```bash
pip install met-timeseries
```

With optional performance and ensemble dependencies:

```bash
pip install met-timeseries[perf,ensemble]
```

For development:

```bash
pip install met-timeseries[dev]
```

## Quick Start

```python
from met_timeseries import Pipeline, PipelineConfig

config = PipelineConfig(
    polygon_path="example_polygons.geojson",
    output_dir="output",
    start_date="2020-01-01",
    end_date="2020-01-31",
)
pipeline = Pipeline(config)
pipeline.run()
```

## CLI Usage

```bash
# Run the full pipeline
met-timeseries run --polygon-path polygons.geojson --start-date 2020-01-01 --end-date 2020-12-31

# Check processing status
met-timeseries status --output-dir output/

# Validate polygon inputs
met-timeseries validate --polygon-path polygons.geojson
```

## Supported Variables

| Variable | Source | Units | Description |
|---|---|---|---|
| `prcp` | PRISM (disaggregated) | mm/hr | Hourly precipitation |
| `rsds` | NLDAS-2 | W/m² | Downward shortwave radiation |
| `temp` | NLDAS-2 | K | Air temperature at 2m |
| `wind_speed` | NLDAS-2 (derived) | m/s | Wind speed at 10m |
| `dewpoint` | NLDAS-2 (derived) | K | Dewpoint temperature |
| `cloud_cover` | NLDAS-2 (derived) | fraction | Cloud cover fraction (0–1) |
| `pet` | Derived (FAO-56) | mm/hr | Penman-Monteith reference ET |

## Architecture

```
met-timeseries/
├── sources/          # Data fetching (NLDAS-2, PRISM)
│   ├── nldas.py      # pynldas2 wrapper with bounding box fetch
│   └── prism.py      # PRISM BIL downloader with local cache
├── derivations/      # Variable derivation from raw data
│   ├── wind.py       # Wind speed from u/v components
│   ├── dewpoint.py   # Dewpoint from specific humidity
│   ├── cloud_cover.py # Cloud cover from SW radiation ratio
│   ├── pet.py        # FAO-56 Penman-Monteith hourly ET
│   └── disaggregation.py  # PRISM daily → hourly using NLDAS pattern
├── aggregation.py    # exactextract zonal statistics wrapper
├── pipeline.py       # Main orchestrator with resume capability
├── ledger.py         # CSV-based processing tracker
├── config.py         # PipelineConfig dataclass
└── io.py             # Partitioned Parquet I/O helpers
```

## Dependencies

- [pynldas2](https://github.com/cheginit/pynldas2) — NLDAS-2 hourly forcing data (HyRiver stack)
- [exactextract](https://github.com/isciences/exactextract) — Fast area-weighted zonal statistics
- [xarray](https://xarray.pydata.org/) — N-dimensional labeled arrays
- [geopandas](https://geopandas.org/) — Vector data handling
- [rioxarray](https://corteva.github.io/rioxarray/) — CRS-aware xarray extension

## License

MIT License — see [LICENSE](LICENSE) for details.
