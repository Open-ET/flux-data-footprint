import calc_footprint_FFP_climatology as myfootprint_s
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
import shapefile
import rasterio
import cv2
from affine import Affine
import pyproj as proj
import cartopy.crs as ccrs
import traceback
import os
import datetime as dt
import scipy

def date_parse(yr,doy,hr):
    """

    This method `date_parse` is used to parse a given date, represented by year (yr), day of year (doy), and hour (hr) into a datetime object.

    Parameters:
    - yr (str): The year of the date in string format.
    - doy (str): The day of year of the date in string format.
    - hr (str): The hour of the date in string format.

    Returns:
    - pd.datetime: The parsed datetime object representing the given date.

    Example usage:
    ```python
    yr = '2022'
    doy = '123'
    hr = '100'

    parsed_date = date_parse(yr, doy, hr)
    ```

    Note:
    - If the hour (hr) parameter is equal to '2400', it will be converted to '000' before parsing the date.

    """
    
    if '2400' in hr:
        hr = '000'
        return pd.datetime.strptime(f'{yr}{int(doy):03}{int(hr):04}', '%Y%j%H%M')
    else:
        return pd.datetime.strptime(f'{yr}{int(doy):03}{int(hr):04}', '%Y%j%H%M')
    
def date_parse_sigv_17(doy,hr):
    """
    Parses a date and time in the format specified by SIGV-17.

    Parameters:
    - doy (str): The day of the year in three digits (e.g., '001' for January 1st).
    - hr (str): The hour and minute in four digits (e.g., '1530' for 3:30 PM).

    Returns:
    - datetime.datetime: A datetime object representing the parsed date and time.

    Note:
    - If '2400' is provided as the value for hr, it will be treated as '000' for the next day.
    - The date is set to the year 2017 for every parsed datetime object.
    - The format string used for parsing is '%Y%j%H%M', which represents the year (4 digits), day of the year (3 digits), hour (2 digits), and minute (2 digits).
    """
    yr='2017'
    if '2400' in hr:
        hr = '000'
        return pd.datetime.strptime(f'{yr}{int(doy)+1}{int(hr):04}', '%Y%j%H%M')
    else:
        return pd.datetime.strptime(f'{yr}{doy}{int(hr):04}', '%Y%j%H%M')


def date_parse_sigv_18(doy,hr):
    """
    Parses a date in the format 'doy' (day of year) + 'hr' (hour) into a `datetime` object.

    Parameters:
    - doy (str or int): The day of the year as a string or integer.
    - hr (str): The hour in the format 'hh'.

    Returns:
    - datetime: The parsed datetime object.

    Example Usage:
    >>> date_parse_sigv_18(365, '23')
    datetime.datetime(2018, 12, 31, 23, 0)

    >>> date_parse_sigv_18('001', '2400')
    datetime.datetime(2019, 1, 2, 0, 0)
    """
    yr='2018'
    if '2400' in hr:
        hr = '000'
        return pd.datetime.strptime(f'{yr}{int(doy)+1}{int(hr):04}', '%Y%j%H%M')
    else:
        return pd.datetime.strptime(f'{yr}{doy}{int(hr):04}', '%Y%j%H%M')
    
def mask_fp_cutoff(f_array,cutoff=.9):
    """
    Masks the elements of the input array based on the cumulative sum of its sorted values.

    Parameters:
    f_array (np.ndarray): Input array of floating-point numbers.
    cutoff (float, optional): Cutoff value for the cumulative sum. Defaults to 0.9.

    Returns:
    np.ndarray: Array with masked values.

    Example:
    >>> f_array = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    >>> mask_fp_cutoff(f_array)
    array([0., 0., 0., 0., 0.])
    """
    val_array = f_array.flatten()
    sort_df = pd.DataFrame({'f':val_array}).sort_values(by='f').iloc[::-1]
    sort_df['cumsum_f'] = sort_df['f'].cumsum()
    
    sort_group = sort_df.groupby('f',as_index=True).mean()
    diff = abs(sort_group['cumsum_f']-cutoff)
    sum_cutoff = diff.idxmin()
    f_array = np.where(f_array>=sum_cutoff,f_array,np.nan)
    f_array[~np.isfinite(f_array)] = 0.00000000e+000
    
    return f_array

def find_transform(xs,ys):
    """

    Find the affine transform between two sets of points.

    Parameters:
    - xs (ndarray): The x-coordinates of the input points.
    - ys (ndarray): The y-coordinates of the input points.

    Returns:
    - aff_transform (Affine): The calculated affine transform between the input points.

    """
    
    shape = xs.shape

    #Choose points to calculate affine transform
    y_points = [0, 0, shape[0] - 1]
    x_points = [0, shape[0] - 1, shape[1] - 1]
    in_xy = np.float32([[i, j] for i, j in zip(x_points, y_points)])
    out_xy = np.float32([[xs[i, j], ys[i, j]] for i, j in zip(y_points, x_points)])
    

    #Calculate affine transform
    aff_transform = Affine(*cv2.getAffineTransform(in_xy,out_xy).flatten())

    return aff_transform
        

    
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

    
    #Flatten arrays and create kd tress from x,y points
    footprint_df = pd.DataFrame({'x_foot':x_2d.ravel(),'y_foot':y_2d.ravel(),'footprint':f_2d.ravel()}).dropna().reset_index()
    points = np.column_stack([footprint_df['x_foot'].values,footprint_df['y_foot'].values])
    dist,idx = kd_tree.query(list(points))
    footprint_df['x'] = combine_xy_df.loc[idx,'x_ls'].reset_index()['x_ls']
    footprint_df['y'] = combine_xy_df.loc[idx,'y_ls'].reset_index()['y_ls']
    
    #Calculate cumulative sum for the footprint weights
    weights = footprint_df.groupby(['x','y'],as_index=False).agg({'footprint':'sum'})
    
    test_weights = []
    test_efs = []
    
    #Loop through weights, find closest raster points, and sum up weights for each raster point
    for p in weights.index:
        pixel_weight = weights['footprint'][p]
        x,y = weights['x'][p],weights['y'][p]
        temp_ef = combine_xy_df[(combine_xy_df['x_ls'] == x) & (combine_xy_df['y_ls'] == y)]['ef'].values
        weighted_ef = pixel_weight*temp_ef
        test_efs.append(temp_ef)
        test_weights.append(pixel_weight)
        
    efs = np.array(test_efs).ravel()
    weights = np.array(test_weights).ravel()
    
    return np.sum(efs*weights)
    
def footprint_cdktree(raster,):
    #temp_nc = ls8.sel(time=t.strftime('%Y-%m-%d'))
    #Calculate ckd tree of landsat images
    ls_x = temp_nc['x'].values
    ls_y = temp_nc['y'].values
    ls_xx,ls_yy = np.meshgrid(ls_x,ls_y)
    ls_xflat = ls_xx.ravel()
    ls_yflat = ls_yy.ravel()
    dummy_mask = temp_nc['EF'].values.ravel()

    combine_xy_df = pd.DataFrame({'x_ls':ls_xflat,'y_ls':ls_yflat,'ef':dummy_mask})
    combine_xy_df = combine_xy_df.dropna().reset_index()

    combine_xy = np.column_stack([combine_xy_df['x_ls'].values,combine_xy_df['y_ls'].values])

    kd_tree = scipy.spatial.cKDTree(list(combine_xy))
    
    return kd_tree

def plot_footprint():
    #Plot out footprints
    fig,ax = plt.subplots(**{'figsize':(10,10)})
    fprint = ax.pcolormesh(x_2d,y_2d,f_2d)
    cbar = fig.colorbar(fprint)
    cbar.set_label(label='Footprint Contribution (per point)',fontsize='xx-large',rotation=270,labelpad=40)
    time = t.strftime('%Y-%m-%d')
    ax.grid(ls='--')
    ax.set_xlim(station_x-origin_d,station_x+origin_d)
    ax.set_ylim(station_y-origin_d,station_y+origin_d)
    ax.set_title(t)
    plt.savefig(f'{time}.png',transparent=True)
