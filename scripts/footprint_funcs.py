

import refet
import utm
import pynldas2 as nldas
import scipy

import calc_footprint_FFP_climatology as myfootprint_s
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
#import shapefile
import rasterio
import cv2
from affine import Affine
import pyproj as proj
#import cartopy.crs as ccrs
import traceback
import os
import datetime as dt
import scipy


def date_parse(yr, doy, hr):
    '''
    Standard date parser (pd.read_csv) for flux table outputs
    '''

    if '2400' in hr:
        hr = '000'
        return pd.datetime.strptime(f'{yr}{int(doy):03}{int(hr):04}', '%Y%j%H%M')
    else:
        return pd.datetime.strptime(f'{yr}{int(doy):03}{int(hr):04}', '%Y%j%H%M')


def date_parse_sigv_17(doy, hr):
    '''
    Sigv date parser (pd.read_csv) for 2017
    '''
    yr = '2017'
    if '2400' in hr:
        hr = '000'
        return pd.datetime.strptime(f'{yr}{int(doy) + 1}{int(hr):04}', '%Y%j%H%M')
    else:
        return pd.datetime.strptime(f'{yr}{doy}{int(hr):04}', '%Y%j%H%M')


def date_parse_sigv_18(doy, hr):
    '''
    Sigv date parser (pd.read_csv) for 2018
    '''
    yr = '2018'
    if '2400' in hr:
        hr = '000'
        return pd.datetime.strptime(f'{yr}{int(doy) + 1}{int(hr):04}', '%Y%j%H%M')
    else:
        return pd.datetime.strptime(f'{yr}{doy}{int(hr):04}', '%Y%j%H%M')


def mask_fp_cutoff(f_array, cutoff=.9):
    '''
    Masks all values outside of the cutoff value

    Args:
        f_array (float) : 2D numpy array of point footprint contribution values (no units)
        cutoff (float) : Cutoff value for the cumulative sum of footprint values

    Returns:
        f_array (float) : 2D numpy array of footprint values, with nan == 0
    '''
    val_array = f_array.flatten()
    sort_df = pd.DataFrame({'f': val_array}).sort_values(by='f').iloc[::-1]
    sort_df['cumsum_f'] = sort_df['f'].cumsum()

    sort_group = sort_df.groupby('f', as_index=True).mean()
    diff = abs(sort_group['cumsum_f'] - cutoff)
    sum_cutoff = diff.idxmin()
    f_array = np.where(f_array >= sum_cutoff, f_array, np.nan)
    f_array[~np.isfinite(f_array)] = 0.00000000e+000

    return f_array


def find_transform(xs, ys):
    '''
    Returns the affine transform for 2d arrays xs and ys

    Args:
        xs (float) : 2D numpy array of x-coordinates
        ys (float) : 2D numpy array of y-coordinates

    Returns:
        aff_transform : affine.Affine object
    '''

    shape = xs.shape

    # Choose points to calculate affine transform
    y_points = [0, 0, shape[0] - 1]
    x_points = [0, shape[0] - 1, shape[1] - 1]
    in_xy = np.float32([[i, j] for i, j in zip(x_points, y_points)])
    out_xy = np.float32([[xs[i, j], ys[i, j]] for i, j in zip(y_points, x_points)])

    # Calculate affine transform
    aff_transform = Affine(*cv2.getAffineTransform(in_xy, out_xy).flatten())

    return aff_transform




def ref_et_from_nldas(date='2020-07-01 10:00', latitude=41.1, longitude=-112.1, zm=10.,
                      elevation=1025.):
    """

    This method calculates reference evapotranspiration (ETo) and reference crop evapotranspiration (ETr) using NLDAS data.

    Parameters:
    - date (str or datetime, optional): The date for which to calculate ETo and ETr. Defaults to '2020-07-01 10:00'.
    - latitude (float, optional): The latitude of the location. Defaults to 41.1.
    - longitude (float, optional): The longitude of the location. Defaults to -112.1.
    - zm (float, optional): The measurement height in meters above the ground for wind speed. Defaults to 10.0.
    - elevation (float, optional): The elevation of the location in meters. Defaults to 1025.0.

    Returns:
    - DataFrame: A pandas DataFrame containing the calculated ETo and ETr values for the given date, latitude, and longitude.

    """
    if isinstance(date, str):
        date = pd.to_datetime(date)

    df = nldas.get_bycoords([(longitude, latitude)], start_date=f'{date:%Y-%m-%d}',
                            end_date=f'{date + pd.DateOffset(days=1):%Y-%m-%d}', source='netcdf')
    df['pair'] = df['psurf'].values / 1000
    df['ea'] = refet.calcs._actual_vapor_pressure(q=df['humidity'], pair=df['pair'])
    df['wind'] = np.sqrt(df['wind_u'] ** 2 + df['wind_v'] ** 2)
    df['temp_C'] = df['temp'].values - 273.15

    unit_dict = {'rs': 'w/m2'}

    # create refet object for calculating
    refet_obj = refet.Hourly(
        tmean=df['temp_C'], ea=df['ea'], rs=df['rsds'], uz=df['wind'],
        zw=zm, elev=elevation, lat=latitude, lon=longitude,
        doy=df.index.dayofyear, time=df.index.hour, method='asce', input_units=unit_dict)  # HH must be int

    df['ETr'] = refet_obj.etr()
    df['ETo'] = refet_obj.eto()
    df.index.name = 'date'

    return df

def weight_raster(x_2d, y_2d, f_2d, flux_raster):
    """
    Calculates the weighted sum of values in a raster based on the provided x, y coordinates and footprint values.

    Parameters:
    - x_2d: 2D array-like. The x-coordinates of the raster.
    - y_2d: 2D array-like. The y-coordinates of the raster.
    - f_2d: 2D array-like. The footprint values of the raster.
    - flux_raster: 2D array-like. The raster containing the values to be weighted.

    Returns:
    - The weighted sum of values in the flux_raster.

    Note: This method uses a KD tree to find the closest points in the raster to the provided coordinates.

    """

    # Flatten arrays and create kd tress from x,y points
    footprint_df = pd.DataFrame(
        {'x_foot': x_2d.ravel(), 'y_foot': y_2d.ravel(), 'footprint': f_2d.ravel()}).dropna().reset_index()
    points = np.column_stack([footprint_df['x_foot'].values, footprint_df['y_foot'].values])
    dist, idx = kd_tree.query(list(points))
    footprint_df['x'] = combine_xy_df.loc[idx, 'x_ls'].reset_index()['x_ls']
    footprint_df['y'] = combine_xy_df.loc[idx, 'y_ls'].reset_index()['y_ls']

    # Calculate cumulative sum for the footprint weights
    weights = footprint_df.groupby(['x', 'y'], as_index=False).agg({'footprint': 'sum'})

    test_weights = []
    test_efs = []

    # Loop through weights, find closest raster points, and sum up weights for each raster point
    for p in weights.index:
        pixel_weight = weights['footprint'][p]
        x, y = weights['x'][p], weights['y'][p]
        temp_ef = combine_xy_df[(combine_xy_df['x_ls'] == x) & (combine_xy_df['y_ls'] == y)]['ef'].values
        weighted_ef = pixel_weight * temp_ef
        test_efs.append(temp_ef)
        test_weights.append(pixel_weight)

    efs = np.array(test_efs).ravel()
    weights = np.array(test_weights).ravel()

    return np.sum(efs * weights)


def footprint_cdktree(raster, ):
    # temp_nc = ls8.sel(time=t.strftime('%Y-%m-%d'))
    # Calculate ckd tree of landsat images
    ls_x = temp_nc['x'].values
    ls_y = temp_nc['y'].values
    ls_xx, ls_yy = np.meshgrid(ls_x, ls_y)
    ls_xflat = ls_xx.ravel()
    ls_yflat = ls_yy.ravel()
    dummy_mask = temp_nc['EF'].values.ravel()

    combine_xy_df = pd.DataFrame({'x_ls': ls_xflat, 'y_ls': ls_yflat, 'ef': dummy_mask})
    combine_xy_df = combine_xy_df.dropna().reset_index()

    combine_xy = np.column_stack([combine_xy_df['x_ls'].values, combine_xy_df['y_ls'].values])

    kd_tree = scipy.spatial.cKDTree(list(combine_xy))

    return kd_tree


def plot_footprint():
    # Plot out footprints
    fig, ax = plt.subplots(**{'figsize': (10, 10)})
    fprint = ax.pcolormesh(x_2d, y_2d, f_2d)
    cbar = fig.colorbar(fprint)
    cbar.set_label(label='Footprint Contribution (per point)', fontsize='xx-large', rotation=270, labelpad=40)
    time = t.strftime('%Y-%m-%d')
    ax.grid(ls='--')
    ax.set_xlim(station_x - origin_d, station_x + origin_d)
    ax.set_ylim(station_y - origin_d, station_y + origin_d)
    ax.set_title(t)
    plt.savefig(f'{time}.png', transparent=True)
