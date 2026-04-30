
import xarray as xr
from met_timeseries import derivations



CST_OFFSET = -6
def generate_timeseries(nldas_data,prism_data,polygon):

    hourly_precipitation = nldas_data[["Rainf"]].shift(CST_OFFSET)
    daily_precipitation = prism_data[["ppt"]].shift(CST_OFFSET)
    shortwave = nldas_data["SWdown"].shift(CST_OFFSET)
    longwave = nldas_data["LWdown"].shift(CST_OFFSET)
    specific_humidity = nldas_data["Qair"].shift(CST_OFFSET)
    pressure = nldas_data["PSurf"].shift(CST_OFFSET)  # Use surface pressure if available, otherwise assume standard atmospheric pressure
    temperature = nldas_data["Tair"].shift(CST_OFFSET)
    wind_n =  nldas_data["Wind_N"].shift(CST_OFFSET)
    wind_e =  nldas_data["Wind_E"].shift(CST_OFFSET)

    ds = [hourly_precipitation.rename("hourly_precip"),
                    daily_precipitation.rename("daily_precip"),
                    shortwave.rename("shortwave"),
                    longwave.rename("longwave"),
                    specific_humidity.rename("specific_humidity"),
                    temperature.rename("temperature"),
                    pressure.rename("pressure"),
                    wind_n.rename('wind_n'),
                    wind_e.rename('wind_e')]
    ds = xr.merge(ds)


    temperature_c = derivations.kelvin_to_celsius(ds['temperature'])
    wind_speed = derivations.adjust_wind_height(derivations.wind_speed(ds['wind_e'],ds['wind_n']))
    cloud_davis = derivations.cloud_cover_davis(
                    ds['shortwave'],
                    min_clearsky=10,
                    k = .6
                    )
    dewpoint_c = derivations.dewpoint_august_roche_magnus(
                    temperature=temperature_c,
                    specific_humidity=ds['specific_humidity'],
                    pressure=ds['pressure'])
    penman_monteith = derivations.penman_monteith_asce(
                    ds['shortwave'],
                    ds['longwave'],
                    temperature_c,
                    dewpoint_c,
                    wind_speed,
                    ds['pressure'])
                    #cn = 60,
                    #cd_day = .24,
                    #cd_night = 1.7) # convert from Pa to kPa for comparison with WDM

    penman_knb = derivations.penman_knb(temperature_c.resample(time="1D").mean(),
                                            temperature_c.resample(time="1D").min(),
                                            temperature_c.resample(time="1D").max(),
                                            dewpoint_c.resample(time="1D").mean(),
                                            wind_speed.resample(time="1D").mean(),
                                            (ds['shortwave']*.0036).resample(time="1D").sum())


    oudin = derivations.pet_oudin(temperature_c.resample(time="1D").mean())


    
    ds_hourly = [temperature_c*9/5 + 32, # convert to Fahrenheit,
          dewpoint_c*9/5 + 32, # convert to Fahrenheit,
          wind_speed* 2.23694, # convert m/s to knots,
          shortwave/11.63, #convert to langleys,
          penman_monteith,
          hourly_precipitation
          ]

    ds_daily = [cloud_davis.resample('D').mean()*10, #conver to 10ths and as daily
        oudin /25.4,
        penman_knb/25.4,
        daily_precipitation
        ] 

    hourly_inputs = weighted_mean_timeseries(ds_hourly, polygon.geometry.iloc[0])
    daily_inputs = weighted_mean_timeseries(ds_daily, polygon.geometry.iloc[0])

    daily_precip = daily_inputs['precipitation']
    houlry_precip = hourly_inputs['precipitation']
    precip = precip, counts = disaggregation.disaggregate_precipitation(daily_ts['ppt'], hourly_ts['Rainf'], method='hybrid')

    inputs = {'precipitation': precip,
              'wind_speed': hourly_inputs['wind_speed'],
              'cloud_davis': daily_inputs['cloud_davis'],
              'solar_radiation': hourly_inputs['shortwave'],
              'dewpoint' : hourly_inputs['dewpoint'],
              'temperature': hourly_inputs['temperature'],
              'oudin': daily_inputs['oudin'],
              'penman_knb': daily_inputs['penman_knb'],
              'penman_montieth': hourly_inputs['penman_montieth']}