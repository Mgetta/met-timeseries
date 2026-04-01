


from pandas import pd


def precipitation(
    polygon, daily_grid: xr.DataArray, hourly_grid: xr.DataArray, 
    method="proportional",
) -> pd.Series:
    hourly_precip = weighted_mean_timeseries(
        xr.Dataset({"precipitation": hourly_grid}),
        polygon,
    )
    daily_precip = weighted_mean_timeseries(
        xr.Dataset({"precipitation": daily_grid}),
        polygon,
    )
    if method == "proportional":
        return disaggregation.disaggregate_precipitation(
            daily_total=daily_precip,
            hourly_pattern=hourly_precip
        )
    return hourly_precip




# def solar_radiation(
#     polygon, daily_grid, hourly_grid=None,
#     method="proportional",           # or "clearsky" if hourly_grid is None
#     lat=None, lon=None,              # needed for clearsky fallback
# ) -> pd.Series:

# def wind_speed(
#     polygon, hourly_grid,            # no daily truth — NLDAS only
#     method="pattern",                # or "equal"
# ) -> pd.Series:

# def air_temperature(
#     polygon, daily_grid, hourly_grid,
#     method="pattern_rescale",        # or "sine"
# ) -> pd.Series:

# def dewpoint_temperature(
#     polygon, hourly_grid,            # derive from NLDAS spfh on grid first
#     daily_grid=None,                 # optional PRISM override
#     method="pattern",                # or "constant"
# ) -> pd.Series:

# def cloud_cover(
#     polygon, hourly_grid,            # derive from NLDAS shortwave on grid
#     method="pattern",
# ) -> pd.Series:

# def pet(
#     polygon, daily_grid, lat,        # daily-only output
#     method="hargreaves",
# ) -> pd.Series: