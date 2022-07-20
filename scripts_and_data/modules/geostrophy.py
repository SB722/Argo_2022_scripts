"""Module that calculates the second meridional derivative of density from binned 
density dataset (made by module density_binning)

(2021, Swantje Bastin)
"""

import numpy as np
import xarray as xr
import cftime
import datetime
import sys
sys.path.append("/sfs/fs6/home-geomar/smomw326/Analysis-EDJ/Argo/argo_modules/")
import prepare_and_bin



def density_dy2(density_dataset, dlat, take_further_values=True):
    """Calculates second meridional derivative of density from Argo data
    (discretely from three values, on the equator, and north and south of it)
    
    Input:
    density_dataset (xarray dataset): containing binned in-situ density, named
        binned_density, on coordinates time, pressure, lat, lon
    dlat (scalar): bin size in y direction in degrees
    take_further_values (boolean): whether, if first northern or southern value
        are missing, values further away shall be taken into account
    
    Returns:
    density_dy2 (xarray dataset): containing the second meridional derivative
        of density along the equator
    """
    
    if take_further_values:
        # check for existence of north and south values, if not there, take values one dlat step further
        density_derivative_1 = density_dataset.binned_density.sel(lat=slice(-dlat,dlat)).diff("lat", n=2).values / ((dlat*111000)**2)
        density_derivative_n2 = ((density_dataset.binned_density.sel(lat=[0,2*dlat]).diff("lat", n=1).values / (2*(dlat*111000)))
                                 - (density_dataset.binned_density.sel(lat=[-dlat,0]).diff("lat", n=1).values / (dlat*111000))) / (1.5*(dlat*111000))
        density_derivative_s2 = ((density_dataset.binned_density.sel(lat=[0,dlat]).diff("lat", n=1).values / (dlat*111000))
                                 - (density_dataset.binned_density.sel(lat=[-2*dlat,0]).diff("lat", n=1).values / (2*(dlat*111000)))) / (1.5*(dlat*111000))
        density_derivative_2 = density_dataset.binned_density.sel(lat=[-2*dlat,0,2*dlat]).diff("lat", n=2).values / ((2*(dlat*111000))**2)
        
        density_data_temp1 = np.where(np.isfinite(density_derivative_1), density_derivative_1, density_derivative_n2)
        density_data_temp2 = np.where(np.isfinite(density_data_temp1), density_data_temp1, density_derivative_s2)
        density_data_temp3 = np.where(np.isfinite(density_data_temp2), density_data_temp2, density_derivative_2)
        
        # choose number of measurements (minimum number of the three involved density bins)
        num_1 = density_dataset.number_measurements.sel(lat=slice(-dlat,dlat)).min(dim="lat").values
        num_n2 = density_dataset.number_measurements.sel(lat=[-dlat,0,2*dlat]).min(dim="lat").values
        num_s2 = density_dataset.number_measurements.sel(lat=[-2*dlat,0,dlat]).min(dim="lat").values
        num_2 = density_dataset.number_measurements.sel(lat=[-2*dlat,0,2*dlat]).min(dim="lat").values
        
        number_data_temp1 = np.where(np.isfinite(density_derivative_1), num_1[:,:,None,:], num_n2[:,:,None,:])
        number_data_temp2 = np.where(np.isfinite(density_data_temp1), number_data_temp1, num_s2[:,:,None,:])
        number_data_temp3 = np.where(np.isfinite(density_data_temp2), number_data_temp2, num_2[:,:,None,:])
        number_data_temp4 = np.where(np.isfinite(density_data_temp3), number_data_temp3, np.nan)
        
        # make xarray dataset with the right dimensions
        density_dy2 = xr.Dataset({"density_dy2": (["time", "pressure", "lat", "lon"], density_data_temp3, {"unit": "kg/m^5"}),
                                  "number_measurements": (["time", "pressure", "lat", "lon"], number_data_temp4)},
                                coords={"time": (["time"], density_dataset.time),
                                       "pressure": (["pressure"], density_dataset.pressure, {"unit": "decibar"}),
                                       "lat": (["lat"], [0,]),
                                       "lon": (["lon"], density_dataset.lon)})
        
    else:
        # standard discretisation of second derivative
        density_dy2_data = density_dataset.binned_density.sel(lat=slice(-dlat,dlat)).diff("lat", n=2).values / ((dlat*111000)**2)
        num_1 = density_dataset.number_measurements.sel(lat=slice(-dlat,dlat)).min(dim="lat").values
        # make xarray dataset with the right dimensions
        density_dy2 = xr.Dataset({"density_dy2": (["time", "pressure", "lat", "lon"], density_dy2_data, {"unit": "kg/m^5"}),
                                  "number_measurements": (["time", "pressure", "lat", "lon"], num_1[:,:,None,:])},
                                coords={"time": (["time"], density_dataset.time),
                                       "pressure": (["pressure"], density_dataset.pressure, {"unit": "decibar"}),
                                       "lat": (["lat"], [0,]),
                                       "lon": (["lon"], density_dataset.lon)})
    
    return density_dy2


    
    
def calculate_reference_u(yomaha_dataset):
    """Calculate reference field of zonal velocity on equator, from Yomaha dataset"""
    
    lon, lat, time, depth, u = prepare_and_bin.prepare_argo(yomaha_dataset, 1000, (-3,3))
    
    dt = 7  # in days, i.e. weekly
    dlat = 0.1   # in degrees
    dlon = 0.5

    binned_u_ds = prepare_and_bin.bin_argo(lon, lat, time, u, dlon, dlat, dt, (-3,3))
    # select only Atlantic
    binned_u_ds_Atl = binned_u_ds.sel(lon=slice(-50,10), lat=slice(-3,3))
    # convert time values into datetime objects
    u_ds_date = binned_u_ds_Atl.assign_coords(time=(cftime.num2pydate(binned_u_ds_Atl.time, "days since 2000-01-01")))

    std_x = 3.75    # half of decorrelation scales (Atlantic)
    std_y = 0.15
    def gaussian_ellipse(x0,y0):
        ge = np.exp( - ((binned_u_ds_Atl.lon.values[:, np.newaxis] - x0)**2/(2*std_x**2) + (binned_u_ds_Atl.lat.values - y0)**2/(2*std_y**2)))
        # put all values outside ellipse (more than 2 sigma away) to NaN
        ge_2 = np.where((ge>np.exp(-3)), ge, np.nan)
        return ge_2.T
    
    smoothed_reference_u = np.zeros((len(binned_u_ds_Atl.lon), 19*12+6)) + np.nan  # u, smoothed means for every month
    time_axis = []
    counter=0
    
    for year in np.arange(2001, 2020, 1):
        for month in np.arange(1, 13, 1):
    
            # select month
            #monthly_ds = u_ds_date.sel(time=f"{year:.0f}-{month:02d}")   # only one month
            monthly_ds = u_ds_date.sel(time=slice(datetime.datetime(year, month, 15) - datetime.timedelta(weeks=10),
                                                  datetime.datetime(year, month, 15) + datetime.timedelta(weeks=10)))   # five months running mean

            for i in range(0, len(binned_u_ds_Atl.lon), 1):
                # choose area that lies within an ellipse (or maybe multiply entire field with gaussian?)
                ge = gaussian_ellipse(binned_u_ds_Atl.lon[i].values, 0)
                time_series_temp = np.zeros(len(monthly_ds.time)) + np.nan
                for k in range(0, len(monthly_ds.time)):
                    ge_norm = ge / np.nansum(ge[np.isfinite(monthly_ds.binned_u[k,:,:])])    # normalised weights to retain physical amplitude of averaged u value
                    time_series_temp[k] = np.nansum(monthly_ds.binned_u[k,:,:].values * ge_norm)
                time_series_temp[(time_series_temp == 0)] = np.nan                    # nansum returns zero for all-NaN array... unwanted behavior
            
                # calculate smoothed time average without removing EDJ harmonic
                smoothed_reference_u[i, counter] = np.nanmean(time_series_temp)
            time_axis.append(datetime.datetime(year, month, 15))
            counter = counter+1
            
    # missing 6 months in 2020
    for year in np.arange(2020, 2021, 1):
        for month in np.arange(1, 7, 1):
    
            # select month
            monthly_ds = u_ds_date.sel(time=f"{year:.0f}-{month:02d}")

            for i in range(0, len(binned_u_ds_Atl.lon), 1):
                # choose area that lies within an ellipse (or maybe multiply entire field with gaussian?)
                ge = gaussian_ellipse(binned_u_ds_Atl.lon[i].values, 0)
                time_series_temp = np.zeros(len(monthly_ds.time)) + np.nan
                for k in range(0, len(monthly_ds.time)):
                    ge_norm = ge / np.nansum(ge[np.isfinite(monthly_ds.binned_u[k,:,:])])    # normalised weights to retain physical amplitude of averaged u value
                    time_series_temp[k] = np.nansum(monthly_ds.binned_u[k,:,:].values * ge_norm)
                time_series_temp[(time_series_temp == 0)] = np.nan                    # nansum returns zero for all-NaN array... unwanted behavior
            
                # calculate smoothed time average without removing EDJ harmonic
                smoothed_reference_u[i, counter] = np.nanmean(time_series_temp)
            time_axis.append(datetime.datetime(year, month, 15))
            counter = counter+1     

    
    monthly_smoothed_u = xr.Dataset({"smoothed_u_equator_1000m": (["lon", "time"], smoothed_reference_u / 100)},   # in m/s
                               coords={"lon": (["lon"], binned_u_ds_Atl.lon.values),
                                       "time": (["time"], time_axis)}
                               )
    
    return monthly_smoothed_u

   