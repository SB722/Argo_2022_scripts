"""Module that takes care of loading and preparation of YoMaHa'07 Argo float
displacement data, as well as binning in space and time

(March 2020, Swantje Bastin)
"""

import numpy as np
import xarray as xr
import scipy.stats as stats

def prepare_argo(argo_file, depth_sel, lat_range):
    """Loads Yomaha'07 Argo velocity data from text file, selects depth
    and latitude range
    
    Input: 
    argo_file (string): path to file
    depth_sel (scalar): parking depth to select
    lat_range (tuple): (lat_min, lat_max), latitudinal range to select
    
    Returns:
    lon, lat, time, depth, u
    """
    
    argo = np.loadtxt(argo_file)
    
    # select the right columns
    lon_deep = argo[:,0]
    lat_deep = argo[:,1]
    pressure_park = argo[:,2]   # in dbars
    time = argo[:,3]   # Julian time (days) relative to 2000-01-01 00:00 UTC
    u = argo[:,4]

    # select depth, and get rid of undefined values
    lon_sel = lon_deep[((pressure_park == depth_sel) & (lon_deep > -999) & (u > -999) & (time > -999))]
    lat_sel = lat_deep[((pressure_park == depth_sel) & (lon_deep > -999) & (u > -999) & (time > -999))]
    z_sel = pressure_park[((pressure_park == depth_sel) & (lon_deep > -999) & (u > -999) & (time > -999))]
    time_sel = time[((pressure_park == depth_sel) & (lon_deep > -999) & (u > -999) & (time > -999))]
    u_sel = u[((pressure_park == depth_sel) & (lon_deep > -999) & (u > -999) & (time > -999))]

    # select latitude range
    lon_eq = lon_sel[((lat_sel > lat_range[0]) & (lat_sel < lat_range[1]))]
    lat_eq = lat_sel[((lat_sel > lat_range[0]) & (lat_sel < lat_range[1]))]
    z_eq = z_sel[((lat_sel > lat_range[0]) & (lat_sel < lat_range[1]))]
    time_eq = time_sel[((lat_sel > lat_range[0]) & (lat_sel < lat_range[1]))]
    u_eq = u_sel[((lat_sel > lat_range[0]) & (lat_sel < lat_range[1]))]

    # convert longitudes from -180/180 to -60/300 to make slice selection possible for all three oceans
    lon_eq[(lon_eq < -60)] = lon_eq[(lon_eq < -60)] % 360
    
    return lon_eq, lat_eq, time_eq, z_eq, u_eq


def bin_argo(lon, lat, time, u, dlon, dlat, dt, lat_range):
    """Bins zonal velocity data in three dimensions (x, y, and time).
    
    Input:
    lon (array, length n): longitude
    lat (array, length n): latitude
    time (array, length n): time
    u (array, length n): zonal velocity
    dlon (scalar): bin size in x
    dlat (scalar): bin size in y
    dt (scalar): bin size in time
    lat_range (tuple): (lat_min, lat_max), latitudinal range to select
    
    Returns:
    binned_u_dataset (xarray dataset): contains central lat, lon, and time of bins, 
    binned zonal velocity, and number of values averaged in each bin
    """
    
    # define coordinate axes for binning
    time_binned = np.arange(np.min(time), np.max(time) + dt, dt)
    lon_binned = np.arange(-60, 300 + dlon, dlon)
    lat_binned = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
    
    ## binning with numpy N-D histogram
    nsamples, edges = np.histogramdd((time, lat, lon), bins=(time_binned, lat_binned, lon_binned))
    # extract bin edges out of "edges" which is a list of three arrays
    time_edges = edges[0]
    lat_edges = edges[1]
    lon_edges = edges[2]
    # Now with weights to get the sum of all u data in the bins
    u_binned_sum, _ = np.histogramdd((time, lat, lon), bins=(time_binned, lat_binned, lon_binned), weights=u)
    # now calculate average u in each bin
    u_binned_ave = u_binned_sum / nsamples
    # throw out unrealistic values
    u_binned_ave[(np.absolute(u_binned_ave) > 40)] = np.nan

    # take centre of bins as coordinates
    lon_centre = 0.5 * (lon_edges[1:] + lon_edges[0:-1]) 
    lat_centre = 0.5 * (lat_edges[1:] + lat_edges[0:-1]) 
    time_centre = 0.5 * (time_edges[1:] + time_edges[0:-1]) 
    
    # generate xarray dataset
    u_dataset = xr.Dataset({"binned_u": (["time", "lat", "lon"], u_binned_ave, {"unit": "cm/s"}),
                            "number_measurements": (["time", "lat", "lon"], nsamples)},
                           coords={"time": (["time"], time_centre, {"unit": "Julian days relative to 2000-01-01 00:00 UTC"}),
                                   "lat": (["lat"], lat_centre),
                                   "lon": (["lon"], lon_centre)})
    
    return u_dataset


def bin_argo_with_stats(lon, lat, time, u, dlon, dlat, dt, lat_range):
    """Bins zonal velocity data in three dimensions (x, y, and time).
    
    Input:
    lon (array, length n): longitude
    lat (array, length n): latitude
    time (array, length n): time
    u (array, length n): zonal velocity
    dlon (scalar): bin size in x
    dlat (scalar): bin size in y
    dt (scalar): bin size in time
    lat_range (tuple): (lat_min, lat_max), latitudinal range to select
    
    Returns:
    binned_u_dataset (xarray dataset): contains central lat, lon, and time of bins, 
        binned averaged zonal velocity, number of values averaged in each bin, sample 
        standard deviation
    """
    
    # define coordinate axes for binning
    time_binned = np.arange(np.min(time), np.max(time) + dt, dt)
    lon_binned = np.arange(-60, 300 + dlon, dlon)
    lat_binned = np.arange(lat_range[0], lat_range[1] + dlat, dlat)
    
    ## binning with generalized scipy N-D histogram
    # mean u in bins
    u_binned_ave, edges, _ = stats.binned_statistic_dd((time, lat, lon), u, statistic="mean", bins=(time_binned, lat_binned, lon_binned))
    # extract bin edges out of "edges" which is a list of three arrays
    time_edges = edges[0]
    lat_edges = edges[1]
    lon_edges = edges[2]
    # number of measurements
    num_meas, _, _ = stats.binned_statistic_dd((time, lat, lon), u, statistic="count", bins=(time_binned, lat_binned, lon_binned))
    # sample standard deviation
    def std_func(x):
        return np.std(x, ddof=1)
    sample_std, _, _ = stats.binned_statistic_dd((time, lat, lon), u, statistic=std_func, bins=(time_binned, lat_binned, lon_binned))
    
    # throw out unrealistic values
    u_binned_ave[(np.absolute(u_binned_ave) > 50)] = np.nan
    num_meas[(np.absolute(u_binned_ave) > 50)] = np.nan
    sample_std[(np.absolute(u_binned_ave) > 50)] = np.nan

    # take centre of bins as coordinates
    lon_centre = 0.5 * (lon_edges[1:] + lon_edges[0:-1]) 
    lat_centre = 0.5 * (lat_edges[1:] + lat_edges[0:-1]) 
    time_centre = 0.5 * (time_edges[1:] + time_edges[0:-1]) 
    
    # generate xarray dataset
    u_dataset = xr.Dataset({"binned_u": (["time", "lat", "lon"], u_binned_ave, {"unit": "cm/s"}),
                            "number_measurements": (["time", "lat", "lon"], num_meas),
                            "sample_standard_deviation": (["time", "lat", "lon"], sample_std)},
                           coords={"time": (["time"], time_centre, {"unit": "Julian days relative to 2000-01-01 00:00 UTC"}),
                                   "lat": (["lat"], lat_centre),
                                   "lon": (["lon"], lon_centre)})
    
    return u_dataset
    
    
def prepare_etopo(etopo_file, lon, lat):
    """Load Etopo1 dataset and interpolate it to argo grid, to plot land-sea mask 
    at chosen depth
    
    Input:
    etopo_file (string): path to etopo1 dataset
    lon (array): longitude vector of binned argo dataset
    lat (array): latitude vector of binned argo dataset
    
    Returns:
    etopo_dataset (xarray dataset): containing interpolated etopo values
    """
    
    etopo = xr.open_dataset(etopo_file)
    # lon vector for interpolation (has to be in -180 to 180)
    lon2 = np.copy(lon)
    lon2[(lon2 > 180)] = ((lon2[(lon2 > 180)] - 180) % 360) - 180
    lon2.sort()
    etopo_interp = etopo.interp(coords={"x": lon2, "y": lat}, method="linear")
    # change lon coordinate back to -60 to 300
    etopo_interp.x.values[(etopo_interp.x.values < -60)] = etopo_interp.x.values[(etopo_interp.x.values < -60)] % 360
    etopo_interp = etopo_interp.sortby("x")
    
    return etopo_interp