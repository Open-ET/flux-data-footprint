import numpy as np
# import sys
#import numbers
# from numpy import ma
# import matplotlib._cntr as cntr
from matplotlib.pyplot import contour
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

def FFP(zm=None, z0=None, umean=None, h=None, ol=None, sigmav=None, ustar=None,
        wind_dir=None, rs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8), rslayer=0,
        nx=1000, crop=False, fig=False):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015: 
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: n.kljun@swansea.ac.uk

    FFP Input
    zm     = Measurement height above displacement height (i.e. z-d) [m]
    z0     = Roughness length [m]; enter None if not known 
    umean  = Mean wind speed at zm [m/s]; enter None if not known 
             Either z0 or umean is required. If both are given,
             z0 is selected to calculate the footprint
    h      = Boundary layer height [m]
    ol     = Obukhov length [m]
    sigmav = standard deviation of lateral velocity fluctuations [ms-1]
    ustar  = friction velocity [ms-1]

    optional inputs:
    wind_dir = wind direction in degrees (of 360) for rotation of the footprint    
    rs       = Percentage of source area for which to provide contours, must be between 10% and 90%. 
               Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
               Expressed either in percentages ("80") or as fractions of 1 ("0.8"). 
               Default is [10:10:80]. Set to "None" for no output of percentages
    nx       = Integer scalar defining the number of grid elements of the scaled footprint.
               Large nx results in higher spatial resolution and higher computing time.
               Default is 1000, nx must be >=600.
    rslayer  = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
               Note that this only gives a rough estimate of the footprint as the model is not 
               valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
               z0 is needed for estimation of the RS.
    crop     = Crop output area to size of the 80% footprint or the largest r given if crop=1
    fig      = Plot an example figure of the resulting footprint (on the screen): set fig = 1. 
               Default is 0 (i.e. no figure). 
 
    FFP output
    x_ci_max = x location of footprint peak (distance from measurement) [m]
    x_ci	 = x array of crosswind integrated footprint [m]
    f_ci	 = array with footprint function values of crosswind integrated footprint [m-1] 
    x_2d	 = x-grid of 2-dimensional footprint [m], rotated if wind_dir is provided
    y_2d	 = y-grid of 2-dimensional footprint [m], rotated if wind_dir is provided
    f_2d	 = footprint function values of 2-dimensional footprint [m-2]
    rs       = percentage of footprint as in input, if provided
    fr       = footprint value at r, if r is provided
    xr       = x-array for contour line of r, if r is provided
    yr       = y-array for contour line of r, if r is provided
    flag_err = 0 if no error, 1 in case of error

    created: 15 April 2015 natascha kljun
    translated to python, December 2015 Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.3
    last change: 08/12/2017 natascha kljun
    Copyright (C) 2015,2016,2017,2018 Natascha Kljun
    """
    # ===========================================================================
    # Input check
    flag_err = 0

    # Check existence of required input pars
    if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1)

    # Define rslayer if not passed
    if rslayer is None:
        rslayer = 0

    # Define crop if not passed
    if crop is None:
        crop = 0

    # Define fig if not passed
    if fig is None:
        fig = 0

    # Check passed values
    if zm <= 0.:
        raise_ffp_exception(2)
    if z0 is not None and umean is None and z0 <= 0.:
        raise_ffp_exception(3)
    if h <= 10.:
        raise_ffp_exception(4)
    if zm > h:
        raise_ffp_exception(5)
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        if rslayer == 1:
            raise_ffp_exception(6)
        else:
            raise_ffp_exception(12)
    if float(zm) / ol <= -15.5:
        raise_ffp_exception(7)
    if sigmav <= 0:
        raise_ffp_exception(8)
    if ustar <= 0.1:
        raise_ffp_exception(9)
    if wind_dir is not None:
        if wind_dir > 360 or wind_dir < 0:
            raise_ffp_exception(10)
    if nx < 600:
        raise_ffp_exception(11)

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    if None not in [z0, umean]:
        raise_ffp_exception(13)

    # ===========================================================================
    # Handle rs
    if rs is not None:

        # Check that rs is a list, otherwise make it a list
        if isinstance(rs,(float,int)):
            if 0.9 < rs <= 1 or 90 < rs <= 100:
                rs = 0.9
            rs = [rs]
        if not isinstance(rs, list):
            raise_ffp_exception(14)

        # If rs is passed as percentages, normalize to fractions of one
        if np.max(rs) >= 1:
            rs = [x / 100. for x in rs]

        # Eliminate any values beyond 0.9 (90%) and inform user
        if np.max(rs) > 0.9:
            raise_ffp_exception(15)
            rs = [item for item in rs if item <= 0.9]

        # Sort levels in ascending order
        rs = list(np.sort(rs))

    # ===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    xstar_end = 30
    oln = 5000  # limit to L for neutral scaling
    k = 0.4  # von Karman

    # ===========================================================================
    # Scaled X* for crosswind integrated footprint
    xstar_ci_param = np.linspace(d, xstar_end, nx + 2)
    xstar_ci_param = xstar_ci_param[1:]

    # Crosswind integrated scaled F* 
    fstar_ci_param = a * (xstar_ci_param - d) ** b * np.exp(-c / (xstar_ci_param - d))
    ind_notnan = ~np.isnan(fstar_ci_param)
    fstar_ci_param = fstar_ci_param[ind_notnan]
    xstar_ci_param = xstar_ci_param[ind_notnan]

    # Scaled sig_y*
    sigystar_param = ac * np.sqrt(bc * xstar_ci_param ** 2 / (1 + cc * xstar_ci_param))

    # ===========================================================================
    # Real scale x and f_ci
    if z0 is not None:
        # Use z0
        if ol <= 0 or ol >= oln:
            xx = (1 - 19.0 * zm / ol) ** 0.25
            psi_f = np.log((1 + xx ** 2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi / 2
        else: #if 0 < ol < oln:
            psi_f = -5.3 * zm / ol

        x = xstar_ci_param * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
        if np.log(zm / z0) - psi_f > 0:
            x_ci = x
            f_ci = fstar_ci_param / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f)
        else:
            x_ci_max, x_ci, f_ci, x_2d, y_2d, f_2d = [None]*6
            flag_err = 1
    else:
        # Use umean if z0 not available
        x = xstar_ci_param * zm / (1. - zm / h) * (umean / ustar * k)
        if 0 < (umean / ustar):
            x_ci = x
            f_ci = fstar_ci_param / zm * (1. - zm / h) / (umean / ustar * k)
        else:
            x_ci_max, x_ci, f_ci, x_2d, y_2d, f_2d = [None]*6
            flag_err = 1

    # Maximum location of influence (peak location)
    xstarmax = -c / b + d
    if z0 is not None:
        x_ci_max = xstarmax * zm / (1. - (zm / h)) * (np.log(zm / z0) - psi_f)
    else:
        x_ci_max = xstarmax * zm / (1. - (zm / h)) * (umean / ustar * k)

    # Real scale sig_y
    if abs(ol) > oln:
        ol = -1E6
    if ol <= 0:  # convective
        scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.80
    else:  # stable ol > 0
        scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.55
    if scale_const > 1:
        scale_const = 1.0
    sigy = sigystar_param / scale_const * zm * sigmav / ustar
    sigy[sigy < 0] = np.nan

    # Real scale f(x,y)
    dx = x_ci[2] - x_ci[1]
    y_pos = np.arange(0, (len(x_ci) / 2.) * dx * 1.5, dx)
    # f_pos = np.full((len(f_ci), len(y_pos)), np.nan)
    f_pos = np.empty((len(f_ci), len(y_pos)))
    f_pos[:] = np.nan
    for ix in range(len(f_ci)):
        f_pos[ix, :] = f_ci[ix] * 1 / (np.sqrt(2 * np.pi) * sigy[ix]) * np.exp(-y_pos ** 2 / (2 * sigy[ix] ** 2))

    # Complete footprint for negative y (symmetrical)
    y_neg = - np.fliplr(y_pos[None, :])[0]
    f_neg = np.fliplr(f_pos)
    y = np.concatenate((y_neg[0:-1], y_pos))
    f = np.concatenate((f_neg[:, :-1].T, f_pos.T)).T

    # Matrices for output
    x_2d = np.tile(x[:, None], (1, len(y)))
    y_2d = np.tile(y.T, (len(x), 1))
    f_2d = f

    # ===========================================================================
    # Derive footprint ellipsoid incorporating R% of the flux, if requested,
    # starting at peak value.
    dy = dx
    xrs = []
    yrs = []
    frs = None
    if rs is not None:
        clevs = get_contour_levels(f_2d, dx, dy, rs)
        frs = [item[2] for item in clevs]

        for ix, fr in enumerate(frs):
            xr, yr = get_contour_vertices(x_2d, y_2d, f_2d, fr)
            if xr is None:
                frs[ix] = None
            xrs.append(xr)
            yrs.append(yr)
    else:
        if crop:
            rs_dummy = 0.8  # crop to 80%
            clevs = get_contour_levels(f_2d, dx, dy, rs_dummy)
            xrs, yrs = get_contour_vertices(x_2d, y_2d, f_2d, clevs[0][2])

    # ===========================================================================
    # Crop domain and footprint to the largest rs value
    if crop:
        xrs_crop = [x for x in xrs if x is not None]
        yrs_crop = [x for x in yrs if x is not None]
        if rs is not None:
            dminx = np.floor(min(xrs_crop[-1]))
            dmaxx = np.ceil(max(xrs_crop[-1]))
            dminy = np.floor(min(yrs_crop[-1]))
            dmaxy = np.ceil(max(yrs_crop[-1]))
        else:
            dminx = np.floor(min(xrs_crop))
            dmaxx = np.ceil(max(xrs_crop))
            dminy = np.floor(min(yrs_crop))
            dmaxy = np.ceil(max(yrs_crop))
        jrange = np.where((y_2d[0] >= dminy) & (y_2d[0] <= dmaxy))[0]
        jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
        jrange = jrange[np.where((jrange >= 0) & (jrange <= y_2d.shape[0] - 1))[0]]
        irange = np.where((x_2d[:, 0] >= dminx) & (x_2d[:, 0] <= dmaxx))[0]
        irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
        irange = irange[np.where((irange >= 0) & (irange <= x_2d.shape[1] - 1))[0]]
        jrange = [[it] for it in jrange]
        x_2d = x_2d[irange, jrange]
        y_2d = y_2d[irange, jrange]
        f_2d = f_2d[irange, jrange]

    # ===========================================================================
    # Rotate 3d footprint if requested
    if wind_dir is not None:
        wind_dir = wind_dir * np.pi / 180.
        dist = np.sqrt(x_2d ** 2 + y_2d ** 2)
        angle = np.arctan2(y_2d, x_2d)
        x_2d = dist * np.sin(wind_dir - angle)
        y_2d = dist * np.cos(wind_dir - angle)

        if rs is not None:
            for ix, r in enumerate(rs):
                xr_lev = np.array([x for x in xrs[ix] if x is not None])
                yr_lev = np.array([x for x in yrs[ix] if x is not None])
                dist = np.sqrt(xr_lev ** 2 + yr_lev ** 2)
                angle = np.arctan2(yr_lev, xr_lev)
                xr = dist * np.sin(wind_dir - angle)
                yr = dist * np.cos(wind_dir - angle)
                xrs[ix] = list(xr)
                yrs[ix] = list(yr)

                # ===========================================================================
    # Plot footprint
    if fig:
        fig_out, ax = plot_footprint(x_2d=x_2d, y_2d=y_2d, fs=f_2d,
                                     show_footprint=True, clevs=frs)

    # ===========================================================================
    # Fill output structure
    if rs is not None:
        return {'x_ci_max': x_ci_max, 'x_ci': x_ci, 'f_ci': f_ci,
                'x_2d': x_2d, 'y_2d': y_2d, 'f_2d': f_2d,
                'rs': rs, 'fr': frs, 'xr': xrs, 'yr': yrs, 'flag_err': flag_err}
    else:
        return {'x_ci_max': x_ci_max, 'x_ci': x_ci, 'f_ci': f_ci,
                'x_2d': x_2d, 'y_2d': y_2d, 'f_2d': f_2d, 'flag_err': flag_err}


# ===============================================================================
# ===============================================================================
def get_contour_levels(f, dx, dy, rs=None):
    """
    Get contour levels based on input parameters.

    Parameters:
        f (array-like): Input array of values.
        dx (float): Grid spacing along x-axis.
        dy (float): Grid spacing along y-axis.
        rs (int, float, list, optional): Contour levels to be computed. If not provided, default levels will be used.

    Returns:
        list: A list of tuples containing contour level, accumulated area, and percentile of each contour level.

    Example:
        f = [[0.5, 0.2, 0.7],
             [0.9, 0.1, 0.4],
             [0.3, 0.6, 0.8]]
        dx = 1.0
        dy = 1.0
        get_contour_levels(f, dx, dy, rs=[0.2, 0.5])

        Output:
        [(0.2, 0.2, 0.2), (0.5, 0.8, 0.4)]
    """

    # Check input and resolve to default levels in needed
    if not isinstance(rs, (int, float, list)):
        rs = list(np.linspace(0.10, 0.90, 9))
    if isinstance(rs, (int, float)):
        rs = [rs]

    # Levels
    pclevs = np.empty(len(rs))
    pclevs[:] = np.nan
    ars = np.empty(len(rs))
    ars[:] = np.nan

    sf = np.sort(f, axis=None)[::-1]
    msf = np.ma.masked_array(sf, mask=(np.isnan(sf) | np.isinf(sf)))  # Masked array for handling potential nan

    csf = msf.cumsum().filled(np.nan) * dx * dy
    for ix, r in enumerate(rs):
        dcsf = np.abs(csf - r)
        pclevs[ix] = sf[np.nanargmin(dcsf)]
        ars[ix] = csf[np.nanargmin(dcsf)]

    return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]


# ===============================================================================
def get_contour_vertices(x, y, f, lev):
    """
    Get Contour Vertices

    This function calculates the contour vertices based on the provided parameters.

    Parameters:
    - x (numeric): X coordinates of the grid points.
    - y (numeric): Y coordinates of the grid points.
    - f (numeric): Values of the grid points.
    - lev (numeric): Contour level value.

    Returns:
    - List[List[numeric]]: List containing the X and Y coordinates of the contour points.

    Example Usage:
        x = [0, 1, 2, 3]
        y = [0, 1, 2, 3]
        f = [[1, 2, 3, 4],
             [5, 6, 7, 8],
             [9, 10, 11, 12],
             [13, 14, 15, 16]]
        lev = 10

        vertices = get_contour_vertices(x, y, f, lev)
        print(vertices)  # Output: [[0, 1, 2, 3], [2, 2, 2, 2]]
    """
    c = contour(x, y, f)
    nlist = c.trace(lev, lev, 0)
    segs = nlist[:len(nlist) // 2]
    N = len(segs[0][:, 0])
    xr = [segs[0][ix, 0] for ix in range(N)]
    yr = [segs[0][ix, 1] for ix in range(N)]

    return [xr, yr]  # x,y coords of contour points.


# ===============================================================================
def plot_footprint(x_2d, y_2d, fs, clevs=None, show_footprint=True, normalize=None,
                   colormap=None, line_width=0.5, iso_labels=None):
    """
    Plots the footprint of a given set of 2D footprints.

    Parameters:
    - x_2d (numpy.ndarray): 2D array of x-coordinates for grid points
    - y_2d (numpy.ndarray): 2D array of y-coordinates for grid points
    - fs (List[numpy.ndarray]): List of 2D footprints to be plotted
    - clevs (List[float], optional): List of contour levels for the footprints. Default is None.
    - show_footprint (bool, optional): Determines whether to show the footprint as a heatmap. Default is True.
    - normalize (str, optional): Determines the normalization method for the heatmap. Default is None.
    - colormap (matplotlib.colors.LinearSegmentedColormap, optional): Colormap for the contours and heatmap. Default is None.
    - line_width (float, optional): Width of contour lines. Default is 0.5.
    - iso_labels (List[int], optional): List of labels for isopleth levels. Default is None.

    Returns:
    - Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: Tuple containing the figure and axes objects for the plot.
    """

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_footprint = False
    else:
        fs = [fs]

    if colormap is None:
        colormap = cm.jet
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 10))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs is not None:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = clevs[::-1]

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev is not None]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c] * len(levs)
            if show_footprint:
                cp = ax.contour(x_2d, y_2d, f, levs, colors='r', linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors=cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels is not None:
                pers = [str(int(clev[0] * 100)) + '%' for clev in clevs]
                fmt = {}
                for l, s in zip(cp.levels, pers):
                    fmt[l] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_footprint:
        if normalize == 'log':
            norm = LogNorm()
        else:
            norm = None

        for f in fs:
            pcol = plt.pcolormesh(x_2d, y_2d, f, cmap=colormap, norm=norm)
            cbar = fig.colorbar(pcol, shrink=1.0, format='%.3e')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.gca().set_aspect('equal', 'box')

        # cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax


# ===============================================================================
# ===============================================================================
exTypes = {'message': 'Message',
           'alert': 'Alert',
           'error': 'Error',
           'fatal': 'Fatal error'}

exceptions = [
    {'code': 1,
     'type': exTypes['fatal'],
     'msg': 'At least one required parameter is missing. Please enter all '
            'required inputs. Check documentation for details.'},
    {'code': 2,
     'type': exTypes['fatal'],
     'msg': 'zm (measurement height) must be larger than zero.'},
    {'code': 3,
     'type': exTypes['fatal'],
     'msg': 'z0 (roughness length) must be larger than zero.'},
    {'code': 4,
     'type': exTypes['fatal'],
     'msg': 'h (BPL height) must be larger than 10m.'},
    {'code': 5,
     'type': exTypes['fatal'],
     'msg': 'zm (measurement height) must be smaller than h (PBL height).'},
    {'code': 6,
     'type': exTypes['alert'],
     'msg': 'zm (measurement height) should be above the roughness sub-layer (12.5*z0).'},
    {'code': 7,
     'type': exTypes['fatal'],
     'msg': 'zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5'},
    {'code': 8,
     'type': exTypes['fatal'],
     'msg': 'sigmav (standard deviation of crosswind) must be larger than zero'},
    {'code': 9,
     'type': exTypes['error'],
     'msg': 'ustar (friction velocity) must be >=0.1.'},
    {'code': 10,
     'type': exTypes['fatal'],
     'msg': 'wind_dir (wind direction) must be >=0 and <=360.'},
    {'code': 11,
     'type': exTypes['error'],
     'msg': 'nx must be >=600.'},
    {'code': 12,
     'type': exTypes['alert'],
     'msg': 'Using z0, ignoring umean.'},
    {'code': 13,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be above roughness sub-layer (12.5*z0).'},
    {'code': 14,
     'type': exTypes['fatal'],
     'msg': 'if provided, rs must be in the form of a number or a list of numbers.'},
    {'code': 15,
     'type': exTypes['alert'],
     'msg': 'rs value(s) larger than 90% were found and eliminated.'},
]


def raise_ffp_exception(code):
    """Raise exception or prints message according to specified code"""

    ex = [it for it in exceptions if it['code'] == code][0]
    string = ex['type'] + '(' + str(ex['code']).zfill(4) + '):\n ' + ex['msg']

    print('')
    if ex['type'] == exTypes['fatal']:
        string = string + '\n FFP_fixed_domain execution aborted.'
        raise Exception(string)
    else:
        print(string)
