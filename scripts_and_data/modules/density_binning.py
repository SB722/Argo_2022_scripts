"""Module that takes care of loading and preparation of Argo float hydrographic
profiles, then calculates density and buoyancy frequency, 
and performs binning in space and time

(2021, Swantje Bastin)
"""

import numpy as np
import xarray as xr
import glob
import gsw
import datetime

def calc_bin_density(argo_monthly_path, dx, dy, dz, density="in-situ"):
    """Calculates in-situ density for all profiles given in all 
    daily argo files belonging to one month, then performs binning.
    
    Input:
    argo_monthly_path (string): path to all argo files belonging to
        one month
    dx (scalar): bin size x, in degrees
    dy (scalar): bin size y, in degrees
    dz (scalar): bin size y, in dbar
    density="in-situ" (string): either "in-situ" or "potential", density to calculate (potential not implemented)
    
    Returns:
    monthly_data (xarray dataset), containing the binned values of density, squared buoyancy frequency, 
        absolute salinity, conservative temperature, number of measurements in the bins,
        the lat, lon and pressure axes, the month and year and the number
        of days that the bin contains
    """
    
    # initialise empty arrays
    SA = []
    CT = []
    rho = []
    longitude = []
    latitude = []
    pressure = []
    
    file_names = glob.glob(argo_monthly_path + "*.nc")
    
    ## 1. data preparation
    
    for file_name in file_names:
        argo_data = xr.open_dataset(file_name)
        for i in argo_data.N_PROF.values:
            # only choose equatorial data
#             if (np.absolute(argo_data.LATITUDE[i].values) < 3) & (argo_data.DATA_MODE[i].values==b'D'):    # this is for only choosing delayed mode data
            if np.absolute(argo_data.LATITUDE[i].values) < 3:
                # Pressure
                if np.all(argo_data.PRES_ADJUSTED[i,:].values > 9000) or np.all(np.isnan(argo_data.PRES_ADJUSTED[i,:].values)):
                    pres = argo_data.PRES[i,:].values
                    # throw out values where quality flag is not 1, 2, 5, or 8, and where pressure values are unrealistic
                    pres[np.isin(argo_data.PRES_QC[i,:].values, [b'1', b'2', b'5', b'8'], invert=True)] = np.nan
                    pres[pres > 6000] = np.nan
                    pres[pres < 0] = np.nan
                else:
                    pres = argo_data.PRES_ADJUSTED[i,:].values
                    # throw out values where quality flag is not 1, 2, 5, or 8, and where pressure values are unrealistic
                    pres[np.isin(argo_data.PRES_ADJUSTED_QC[i,:].values, [b'1', b'2', b'5', b'8'], invert=True)] = np.nan
                    pres[pres > 6000] = np.nan
                    pres[pres < 0] = np.nan
                # Salinity
                if np.all(argo_data.PSAL_ADJUSTED[i,:].values > 9000) or np.all(np.isnan(argo_data.PSAL_ADJUSTED[i,:].values)):
                    psal = argo_data.PSAL[i,:].values
                    # throw out values where quality flag is not 1, 2, 5, or 8, and where salinity values are unrealistic
                    psal[np.isin(argo_data.PSAL_QC[i,:].values, [b'1', b'2', b'5', b'8'], invert=True)] = np.nan
                    psal[psal > 50] = np.nan
                    psal[psal < 0] = np.nan
                else:
                    psal = argo_data.PSAL_ADJUSTED[i,:].values
                    # throw out values where quality flag is not 1, 2, 5, or 8, and where salinity values are unrealistic
                    psal[np.isin(argo_data.PSAL_ADJUSTED_QC[i,:].values, [b'1', b'2', b'5', b'8'], invert=True)] = np.nan
                    psal[psal > 50] = np.nan
                    psal[psal < 0] = np.nan
                # Temperature
                if np.all(argo_data.TEMP_ADJUSTED[i,:].values > 9000) or np.all(np.isnan(argo_data.TEMP_ADJUSTED[i,:].values)):
                    temp = argo_data.TEMP[i,:].values
                    # throw out values where quality flag is not 1, 2, 5, or 8, and where temperature values are unrealistic
                    temp[np.isin(argo_data.TEMP_QC[i,:].values, [b'1', b'2', b'5', b'8'], invert=True)] = np.nan
                    temp[temp > 50] = np.nan
                    temp[temp < 0] = np.nan
                else:
                    temp = argo_data.TEMP_ADJUSTED[i,:].values
                    # throw out values where quality flag is not 1, 2, 5, or 8, and where temperature values are unrealistic
                    temp[np.isin(argo_data.TEMP_ADJUSTED_QC[i,:].values, [b'1', b'2', b'5', b'8'], invert=True)] = np.nan
                    temp[temp > 50] = np.nan
                    temp[temp < 0] = np.nan
                lon = np.zeros(len(temp)) + argo_data.LONGITUDE[i].values
                lat = np.zeros(len(temp)) + argo_data.LATITUDE[i].values
                
                # calculate absolute salinity, conservative temperature, density
                abssal = gsw.SA_from_SP(psal, pres, lon, lat)
                constemp = gsw.CT_from_t(abssal, temp, pres)
                density = gsw.rho_t_exact(abssal, temp, pres)
                
                # append new profile to arrays
                SA.extend(list(abssal))
                CT.extend(list(constemp))
                rho.extend(list(density))
                longitude.extend(list(lon))
                latitude.extend(list(lat))
                pressure.extend(list(pres))
                
    ## 2. binning/averaging
    # to numpy array
    SA2 = np.asarray(SA)
    CT2 = np.asarray(CT)
    rho2 = np.asarray(rho)
    longitude2 = np.asarray(longitude)
    latitude2 = np.asarray(latitude)
    pressure2 = np.asarray(pressure)
    # retain only finite values
    SA3 = SA2[np.isfinite(rho2)]
    CT3 = CT2[np.isfinite(rho2)]
    longitude3 = longitude2[np.isfinite(rho2)]
    latitude3 = latitude2[np.isfinite(rho2)]
    pressure3 = pressure2[np.isfinite(rho2)]
    rho3 = rho2[np.isfinite(rho2)]
                
    # initialise axes for binning
    lon_bin = np.arange(-47.5-0.5*dx, 12.5+0.5*dx, dx)
    lat_bin = np.arange(-3-0.5*dy, 3+0.5*dy, dy)
    pres_bin = np.arange(0, 2000+dz, dz)
    
    # binning with numpy N-D histogram
    nsamples, edges = np.histogramdd((pressure3, latitude3, longitude3), bins=(pres_bin, lat_bin, lon_bin))
    # extract bin edges out of "edges" which is a list of three arrays
    pres_edges = edges[0]
    lat_edges = edges[1]
    lon_edges = edges[2]
    # Now with weights to get the sum of all data in the bins
    rho_binned_sum, _ = np.histogramdd((pressure3, latitude3, longitude3), bins=(pres_bin, lat_bin, lon_bin), weights=rho3)
    CT_binned_sum, _ = np.histogramdd((pressure3, latitude3, longitude3), bins=(pres_bin, lat_bin, lon_bin), weights=CT3)
    SA_binned_sum, _ = np.histogramdd((pressure3, latitude3, longitude3), bins=(pres_bin, lat_bin, lon_bin), weights=SA3)
    # now calculate average density in each bin
    rho_binned_ave = rho_binned_sum / nsamples
    CT_binned_ave = CT_binned_sum / nsamples
    SA_binned_ave = SA_binned_sum / nsamples

    # take centre of bins as coordinates
    lon_centre = 0.5 * (lon_edges[1:] + lon_edges[0:-1]) 
    lat_centre = 0.5 * (lat_edges[1:] + lat_edges[0:-1]) 
    pres_centre = 0.5 * (pres_edges[1:] + pres_edges[0:-1]) 
    
    # time coordinate
    time_coord = datetime.datetime(int(file_names[0][-16:-12]), int(file_names[0][-12:-10]), 15)
    
    # calculate Nsquared for each profile
    Nsq = np.zeros((len(pres_centre)-1, len(lat_centre), len(lon_centre))) + np.nan
    for i in range(0, len(lat_centre)):
        for j in range(0, len(lon_centre)):
            Nsq[:, i, j], pmid = gsw.Nsquared(SA_binned_ave[:, i, j], CT_binned_ave[:, i, j], pres_centre, lat=lat_centre[i])
            
    
    # generate xarray dataset
    monthly_density_dataset = xr.Dataset({"binned_density": (["time", "pressure", "lat", "lon"], rho_binned_ave[None,:,:,:], {"unit": "kg/m^3"}),
                                          "binned_absolute_salinity": (["time", "pressure", "lat", "lon"], SA_binned_ave[None, :, :, :], {"unit": "g/kg"}),
                                          "binned_conservative_temperature": (["time", "pressure", "lat", "lon"], CT_binned_ave[None, :, :, :], {"unit": "deg C"}),
                                          "Nsquared": (["time", "midpressure", "lat", "lon"], Nsq[None, :, :, :], {"unit": "1/s"}),
                                          "number_measurements": (["time", "pressure", "lat", "lon"], nsamples[None,:,:,:]),
                                          "numdays_weight": (["time"], [len(file_names),])},
                                          coords={"time": (["time"], [time_coord,]),
                                                  "pressure": (["pressure"], pres_centre, {"unit": "decibars (0 at sea level)"}),
                                                  "lat": (["lat"], lat_centre),
                                                  "lon": (["lon"], lon_centre),
                                                  "midpressure": (["midpressure"], pmid)})
    
    return monthly_density_dataset
                    