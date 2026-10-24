# -*- coding: utf-8 -*-
"""
Helper functions to mask footprints and georeference them as rasters.
"""

import numpy as np
import pandas as pd
from affine import Affine


def mask_fp_cutoff(f_array,cutoff=.9):
    """
    Mask footprint values outside of a cumulative source area cutoff.

    Footprint values are sorted from largest to smallest and summed. The
    value where the cumulative sum is closest to ``cutoff`` becomes the
    threshold, and all smaller values are set to zero. The input should
    be footprint weights per grid cell that sum to about 1 over the
    domain, e.g. ``fclim_2d * dx**2`` from :func:`~fluxfootprint.ffp.ffp_climatology`.

    Arguments:
        f_array (:obj:`numpy.ndarray`): 2D array of footprint weights per
            grid cell [-].

    Keyword Arguments:
        cutoff (float): default 0.9. Fraction of the cumulative footprint
            to keep.

    Returns:
        f_array (:obj:`numpy.ndarray`): 2D footprint weights with values
        outside of the cutoff set to 0.

    Example:
        >>> f_2d = mask_fp_cutoff(f_2d, cutoff=0.8)
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
    Get the affine transform used to write a footprint grid to a GeoTIFF.

    ``xs`` and ``ys`` are the projected coordinates of the footprint grid
    nodes, e.g. ``x_2d`` and ``y_2d`` from :func:`~fluxfootprint.ffp.ffp_climatology`
    shifted to the tower location. The nodes become pixel centres, and
    because a GeoTIFF transform gives the upper left corner of a pixel the
    transform is moved by half a pixel.

    Arguments:
        xs (:obj:`numpy.ndarray`): 2D array of x-coordinates [m].
        ys (:obj:`numpy.ndarray`): 2D array of y-coordinates [m].

    Returns:
        aff_transform (:obj:`affine.Affine`): transform to pass to
        :func:`rasterio.open` when writing the raster.

    Example:
        >>> transform = find_transform(x_2d, y_2d)

    Note:
        The grid must be regular and aligned with the x and y axes, as
        produced by :func:`numpy.meshgrid` and the FFP model.
    """
    n_rows, n_cols = xs.shape

    # pixel size from the full extent of the grid nodes [m]
    pixel_width = (xs[0, -1] - xs[0, 0]) / (n_cols - 1)
    pixel_height = (ys[-1, 0] - ys[0, 0]) / (n_rows - 1)

    # node coordinates are pixel centres, move the origin to the pixel corner
    x_origin = xs[0, 0] - pixel_width / 2
    y_origin = ys[0, 0] - pixel_height / 2

    aff_transform = Affine(
        pixel_width, 0, x_origin,
        0, pixel_height, y_origin
    )

    return aff_transform
