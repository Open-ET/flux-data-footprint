from __future__ import annotations
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm



def ffp_climatology(zm=None, z0=None, umean=None, h=None, ol=None, sigmav=None, ustar=None,
                    wind_dir=None, domain=None, dx=None, dy=None, nx=None, ny=None,
                    rs=None, rslayer=0,
                    smooth_data=1, crop=False, pulse=None, verbosity=2, fig=False, **kwargs):
    """
    Derive a flux footprint estimate based on the simple parameterisation FFP
    See Kljun, N., P. Calanca, M.W. Rotach, H.P. Schmid, 2015:
    The simple two-dimensional parameterisation for Flux Footprint Predictions FFP.
    Geosci. Model Dev. 8, 3695-3713, doi:10.5194/gmd-8-3695-2015, for details.
    contact: natascha.kljun@cec.lu.se

    This function calculates footprints within a fixed physical domain for a series of
    time steps, rotates footprints into the corresponding wind direction and aggregates
    all footprints to a footprint climatology. The percentage of source area is
    calculated for the footprint climatology.
    For determining the optimal extent of the domain (large enough to include footprints)
    use calc_footprint_FFP.py.

    FFP Input
        All vectors need to be of equal length (one value for each time step)
        zm       = Measurement height above displacement height (i.e. z-d) [m]
                   usually a scalar, but can also be a vector
        z0       = Roughness length [m] - enter [None] if not known
                   usually a scalar, but can also be a vector
        umean    = Vector of mean wind speed at zm [ms-1] - enter [None] if not known
                   Either z0 or umean is required. If both are given,
                   z0 is selected to calculate the footprint
        h        = Vector of boundary layer height [m]
        ol       = Vector of Obukhov length [m]
        sigmav   = Vector of standard deviation of lateral velocity fluctuations [ms-1]
        ustar    = Vector of friction velocity [ms-1]
        wind_dir = Vector of wind direction in degrees (of 360) for rotation of the footprint

        Optional input:
        domain       = Domain size as an array of [xmin xmax ymin ymax] [m].
                       Footprint will be calculated for a measurement at [0 0 zm] m
                       Default is smallest area including the r% footprint or [-1000 1000 -1000 1000]m,
                       whichever smallest (80% footprint if r not given).
        dx, dy       = Cell size of domain [m]
                       Small dx, dy results in higher spatial resolution and higher computing time
                       Default is dx = dy = 2 m. If only dx is given, dx=dy.
        nx, ny       = Two integer scalars defining the number of grid elements in x and y
                       Large nx/ny result in higher spatial resolution and higher computing time
                       Default is nx = ny = 1000. If only nx is given, nx=ny.
                       If both dx/dy and nx/ny are given, dx/dy is given priority if the domain is also specified.
        rs           = Percentage of source area for which to provide contours, must be between 10% and 90%.
                       Can be either a single value (e.g., "80") or a list of values (e.g., "[10, 20, 30]")
                       Expressed either in percentages ("80") or as fractions of 1 ("0.8").
                       Default is [10:10:80]. Set to "None" for no output of percentages
        rslayer      = Calculate footprint even if zm within roughness sublayer: set rslayer = 1
                       Note that this only gives a rough estimate of the footprint as the model is not
                       valid within the roughness sublayer. Default is 0 (i.e. no footprint for within RS).
                       z0 is needed for estimation of the RS.
        smooth_data  = Apply convolution filter to smooth footprint climatology if smooth_data=1 (default)
        crop         = Crop output area to size of the 80% footprint or the largest r given if crop=1
        pulse        = Display progress of footprint calculations every pulse-th footprint (e.g., "100")
        verbosity    = Level of verbosity at run time: 0 = completely silent, 1 = notify only of fatal errors,
                       2 = all notifications
        fig          = Plot an example figure of the resulting footprint (on the screen): set fig = 1.
                       Default is 0 (i.e. no figure).

    FFP output
        FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
        x_2d	    = x-grid of 2-dimensional footprint [m]
        y_2d	    = y-grid of 2-dimensional footprint [m]
        fclim_2d = Normalised footprint function values of footprint climatology [m-2]
        rs       = Percentage of footprint as in input, if provided
        fr       = Footprint value at r, if r is provided
        xr       = x-array for contour line of r, if r is provided
        yr       = y-array for contour line of r, if r is provided
        n        = Number of footprints calculated and included in footprint climatology
        flag_err = 0 if no error, 1 in case of error, 2 if not all contour plots (rs%) within specified domain,
                   3 if single data points had to be removed (outside validity)

    Created: 19 May 2016 natascha kljun
    Converted from matlab to python, together with Gerardo Fratini, LI-COR Biosciences Inc.
    version: 1.42
    last change: 11/12/2019 Gerardo Fratini, ported to Python 3.x
    Copyright (C) 2015 - 2023 Natascha Kljun
    """

    if rs is None:
        rs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    output = {}
    # ===========================================================================
    # Get kwargs
    show_heatmap = kwargs.get('show_heatmap', True)

    # ===========================================================================
    # Input check
    flag_err = 0

    # Check existence of required input pars
    if None in [zm, h, ol, sigmav, ustar] or (z0 is None and umean is None):
        raise_ffp_exception(1, verbosity)

    # Convert all input items to lists
    if not isinstance(zm, list):
        zm = [zm]
    if not isinstance(h, list):
        h = [h]
    if not isinstance(ol, list):
        ol = [ol]
    if not isinstance(sigmav, list):
        sigmav = [sigmav]
    if not isinstance(ustar, list):
        ustar = [ustar]
    if not isinstance(wind_dir, list):
        wind_dir = [wind_dir]
    if not isinstance(z0, list):
        z0 = [z0]
    if not isinstance(umean, list):
        umean = [umean]

    # Check that all lists have same length, if not raise an error and exit
    ts_len = len(ustar)
    if any(len(lst) != ts_len for lst in [sigmav, wind_dir, h, ol]):
        # at least one list has a different length, exit with error message
        raise_ffp_exception(11, verbosity)

    # Special treatment for zm, which is allowed to have length 1 for any
    # length >= 1 of all other parameters
    if all(val is None for val in zm):
        raise_ffp_exception(12, verbosity)
    if len(zm) == 1:
        raise_ffp_exception(17, verbosity)
        zm = [zm[0] for i in range(ts_len)]

    # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
    # If at least one value of z0 is passed, use z0 (by setting umean to None)
    if not all(val is None for val in z0):
        raise_ffp_exception(13, verbosity)
        umean = [None for i in range(ts_len)]
        # If only one value of z0 was passed, use that value for all footprints
        if len(z0) == 1:
            z0 = [z0[0] for i in range(ts_len)]
    elif len(umean) == ts_len and not all(val is None for val in umean):
        raise_ffp_exception(14, verbosity)
        z0 = [None for i in range(ts_len)]
    else:
        raise_ffp_exception(15, verbosity)

    # Rename lists as now the function expects time series of inputs
    ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans = \
        ustar, sigmav, h, ol, wind_dir, zm, z0, umean

    # ===========================================================================
    # Handle rs
    if rs is not None:

        # Check that rs is a list, otherwise make it a list
        if isinstance(rs, float) or isinstance(rs, int):
            if 0.9 < rs <= 1 or 90 < rs <= 100:
                rs = 0.9
            rs = [rs]
        if not isinstance(rs, list):
            raise_ffp_exception(18, verbosity)

        # If rs is passed as percentages, normalize to fractions of one
        if np.max(rs) >= 1:
            rs = [x / 100. for x in rs]

        # Eliminate any values beyond 0.9 (90%) and inform user
        if np.max(rs) > 0.9:
            raise_ffp_exception(19, verbosity)
            rs = [item for item in rs if item <= 0.9]

        # Sort levels in ascending order
        rs = list(np.sort(rs))

    # Define computational domain
    # Check passed values and make some smart assumptions
    if isinstance(dx, float) or isinstance(dx, int) and dy is None:
        dy = dx
    if isinstance(dy, float) or isinstance(dy, int) and dx is None:
        dx = dy
    if not all(isinstance(item, float) for item in [dx, dy]) or not all(isinstance(item, int) for item in [dx, dy]):
        dx = dy = None
    if isinstance(nx, int) and ny is None:
        ny = nx
    if not (not isinstance(ny, int) or not (nx is None)):
        nx = ny
    if not all(isinstance(item, int) for item in [nx, ny]):
        nx = ny = None
    if not isinstance(domain, list) or len(domain) != 4:
        domain = None

    if all(item is None for item in [dx, nx, domain]):
        # If nothing is passed, default domain is a square of 2 Km size centered
        # at the tower with pizel size of 2 meters (hence a 1000x1000 grid)
        domain = [-1000., 1000., -1000., 1000.]
        dx = dy = 2.
        nx = ny = 1000
    elif domain is not None:
        # If domain is passed, it takes the precendence over anything else
        if dx is not None:
            # If dx/dy is passed, takes precendence over nx/ny
            nx = int((domain[1] - domain[0]) / dx)
            ny = int((domain[3] - domain[2]) / dy)
        else:
            # If dx/dy is not passed, use nx/ny (set to 1000 if not passed)
            if nx is None:
                nx = ny = 1000
            # If dx/dy is not passed, use nx/ny
            dx = (domain[1] - domain[0]) / float(nx)
            dy = (domain[3] - domain[2]) / float(ny)
    elif dx is not None and nx is not None:
        # If domain is not passed but dx/dy and nx/ny are, define domain
        domain = [-nx * dx / 2, nx * dx / 2, -ny * dy / 2, ny * dy / 2]
    elif dx is not None:
        # If domain is not passed but dx/dy is, define domain and nx/ny
        domain = [-1000, 1000, -1000, 1000]
        nx = int((domain[1] - domain[0]) / dx)
        ny = int((domain[3] - domain[2]) / dy)
    elif nx is not None:
        # If domain and dx/dy are not passed but nx/ny is, define domain and dx/dy
        domain = [-1000, 1000, -1000, 1000]
        dx = (domain[1] - domain[0]) / float(nx)
        dy = (domain[3] - domain[2]) / float(nx)

    # Put domain into more convenient vars
    xmin, xmax, ymin, ymax = domain

    # Define rslayer if not passed
    if rslayer is None:
        rslayer = 0

    # Define smooth_data if not passed
    if smooth_data is None:
        smooth_data = 1

    # Define crop if not passed
    if crop is None:
        crop = 0

    # Define pulse if not passed
    if pulse is None:
        if ts_len <= 20:
            pulse = 1
        else:
            pulse = int(ts_len / 20)

    # Define fig if not passed
    if fig is None:
        fig = 0

    # ===========================================================================
    # Model parameters
    a = 1.4524
    b = -1.9914
    c = 1.4622
    d = 0.1359
    ac = 2.17
    bc = 1.66
    cc = 20.0

    oln = 5000  # limit to L for neutral scaling
    k = 0.4  # von Karman

    # ===========================================================================
    # Define physical domain in cartesian and polar coordinates
    # Cartesian coordinates
    x = np.linspace(xmin, xmax, nx + 1)
    y = np.linspace(ymin, ymax, ny + 1)
    x_2d, y_2d = np.meshgrid(x, y)

    # Polar coordinates
    # Set theta such that North is pointing upwards and angles increase clockwise
    rho = np.sqrt(x_2d ** 2 + y_2d ** 2)
    theta = np.arctan2(x_2d, y_2d)

    # initialize raster for footprint climatology
    fclim_2d = np.zeros(x_2d.shape)

    # ===========================================================================
    # Loop on time series

    # Initialize logic array valids to those 'timestamps' for which all inputs are
    # at least present (but not necessarily phisically plausible)
    valids = [True if not any([val is None for val in vals]) else False \
              for vals in zip(ustars, sigmavs, hs, ols, wind_dirs, zms)]

    if verbosity > 1:
        print('')

    for ix, (ustar, sigmav, h, ol, wind_dir, zm, z0, umean) \
            in enumerate(zip(ustars, sigmavs, hs, ols, wind_dirs, zms, z0s, umeans)):

        # Counter
        if verbosity > 1 and ix % pulse == 0:
            print('Calculating footprint ', ix + 1, ' of ', ts_len)

        valids[ix] = check_ffp_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbosity)

        # If inputs are not valid, skip current footprint
        if not valids[ix]:
            raise_ffp_exception(16, verbosity)
        else:
            # ===========================================================================
            # Rotate coordinates into wind direction
            if wind_dir is not None:
                rotated_theta = theta - wind_dir * np.pi / 180.
            else:
                rotated_theta = theta

            # ===========================================================================
            # Create real scale crosswind integrated footprint and dummy for
            # rotated scaled footprint
            fstar_ci_dummy = np.zeros(x_2d.shape)
            f_ci_dummy = np.zeros(x_2d.shape)
            xstar_ci_dummy = np.zeros(x_2d.shape)
            px = np.ones(x_2d.shape)
            if z0 is not None:
                if ol > 0 and ol < oln:  # ol > 0 and ol < oln:
                    psi_f = -5.3 * zm / ol
                # Use z0
                else:
                    xx = (1 - 19.0 * zm / ol) ** 0.25
                    psi_f = (np.log((1 + xx ** 2) / 2.) + 2. * np.log((1 + xx) / 2.) - 2. * np.arctan(xx) + np.pi / 2)

                if (np.log(zm / z0) - psi_f) > 0:
                    xstar_ci_dummy = (rho * np.cos(rotated_theta) / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f))
                    px = np.where(xstar_ci_dummy > d)
                    fstar_ci_dummy[px] = a * (xstar_ci_dummy[px] - d) ** b * np.exp(-c / (xstar_ci_dummy[px] - d))
                    f_ci_dummy[px] = (fstar_ci_dummy[px] / zm * (1. - (zm / h)) / (np.log(zm / z0) - psi_f))
                else:
                    flag_err = 3
                    valids[ix] = 0
            else:
                # Use umean if z0 not available
                xstar_ci_dummy = (rho * np.cos(rotated_theta) / zm * (1. - (zm / h)) / (umean / ustar * k))
                px = np.where(xstar_ci_dummy > d)
                fstar_ci_dummy[px] = a * (xstar_ci_dummy[px] - d) ** b * np.exp(-c / (xstar_ci_dummy[px] - d))
                f_ci_dummy[px] = (fstar_ci_dummy[px] / zm * (1. - (zm / h)) / (umean / ustar * k))

            # ===========================================================================
            # Calculate dummy for scaled sig_y* and real scale sig_y
            sigystar_dummy = np.zeros(x_2d.shape)
            sigystar_dummy[px] = (ac * np.sqrt(bc * np.abs(xstar_ci_dummy[px]) ** 2 / (1 +
                                                                                       cc * np.abs(
                        xstar_ci_dummy[px]))))

            if abs(ol) > oln:
                ol = -1E6
            if ol <= 0:  # convective
                scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.80
            elif ol > 0:  # stable
                scale_const = 1E-5 * abs(zm / ol) ** (-1) + 0.55
            if scale_const > 1:
                scale_const = 1.0

            sigy_dummy = np.zeros(x_2d.shape)
            sigy_dummy[px] = (sigystar_dummy[px] / scale_const * zm * sigmav / ustar)
            sigy_dummy[sigy_dummy < 0] = np.nan

            # ===========================================================================
            # Calculate real scale f(x,y)
            f_2d = np.zeros(x_2d.shape)
            f_2d[px] = (f_ci_dummy[px] / (np.sqrt(2 * np.pi) * sigy_dummy[px]) *
                        np.exp(-(rho[px] * np.sin(rotated_theta[px])) ** 2 / (2. * sigy_dummy[px] ** 2)))

            # ===========================================================================
            # Add to footprint climatology raster
            fclim_2d = fclim_2d + f_2d

    # ===========================================================================
    # Continue if at least one valid footprint was calculated
    n = sum(valids)
    vs = None
    clevs = None
    xrs = []
    yrs = []
    frs = []

    if n == 0:
        print("No footprint calculated")
        flag_err = 1
    else:

        # ===========================================================================
        # Normalize and smooth footprint climatology
        fclim_2d = fclim_2d / n

        if smooth_data is not None:
            skernel = np.matrix('0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05')
            fclim_2d = sg.convolve2d(fclim_2d, skernel, mode='same')
            fclim_2d = sg.convolve2d(fclim_2d, skernel, mode='same')

        # ===========================================================================
        # Derive footprint ellipsoid incorporating R% of the flux, if requested,
        # starting at peak value.

        if rs is not None:
            clevs = get_contour_levels(fclim_2d, dx, dy, rs)
            frs = [item[2] for item in clevs]
            for ix, fr in enumerate(frs):
                xr, yr = get_contour_vertices(x_2d, y_2d, fclim_2d, fr)
                if xr is None:
                    frs[ix] = None
                    flag_err = 2
                xrs.append(xr)
                yrs.append(yr)
        else:
            if crop:
                rs_dummy = 0.8  # crop to 80%
                clevs = get_contour_levels(fclim_2d, dx, dy, rs_dummy)

                xrs, yrs = get_contour_vertices(x_2d, y_2d, fclim_2d, clevs[0][2])

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

            if dminy >= ymin and dmaxy <= ymax:
                jrange = np.where((y_2d[:, 0] >= dminy) & (y_2d[:, 0] <= dmaxy))[0]
                jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
                jrange = jrange[np.where((jrange >= 0) & (jrange <= y_2d.shape[0]))[0]]
            else:
                jrange = np.linspace(0, 1, y_2d.shape[0] - 1)

            if dminx >= xmin and dmaxx <= xmax:
                irange = np.where((x_2d[0, :] >= dminx) & (x_2d[0, :] <= dmaxx))[0]
                irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
                irange = irange[np.where((irange >= 0) & (irange <= x_2d.shape[1]))[0]]
            else:
                irange = np.linspace(0, 1, x_2d.shape[1] - 1)

            jrange = [[it] for it in jrange]
            x_2d = x_2d[jrange, irange]
            y_2d = y_2d[jrange, irange]
            fclim_2d = fclim_2d[jrange, irange]

        # ===========================================================================
        # Plot footprint
        if fig:
            fig_out, ax = plot_footprint(x_2d=x_2d, y_2d=y_2d, fs=fclim_2d,
                                         show_heatmap=show_heatmap, clevs=frs)
    output['x_2d'] = x_2d
    output['y_2d'] = y_2d
    output['fclim_2d'] = fclim_2d
    output['rs'] = rs
    output['frs'] = frs
    output['xr'] = xrs
    output['yr'] = yrs
    output['n'] = n
    output['flag_err'] = flag_err
    # ===========================================================================
    # Fill output structure
    return output


# ===============================================================================
# ===============================================================================
def check_ffp_inputs(ustar, sigmav, h, ol, wind_dir, zm, z0, umean, rslayer, verbosity):
    """
    Check the passed values for physical plausibility and consistency.

    Parameters:
    - ustar: float - friction velocity
    - sigmav: float - standard deviation of lateral wind velocity fluctuations
    - h: float - height of measurement
    - ol: float - Obukhov length
    - wind_dir: float - wind direction
    - zm: float - measurement height for wind speed and temperature
    - z0: float or None - roughness length for momentum
    - umean: float or None - horizontal mean wind speed at measurement height
    - rslayer: int - flag indicating if Richardson number is calculated in the
      surface layer (=1) or not (=0)
    - verbosity: int - verbosity level for logging output

    Returns:
    - True if all input values are valid, False otherwise

    Raises:
    - ffp_exception: Exception with specific error code and verbosity level for logging
      output

    """
    # Check passed values for physical plausibility and consistency
    if zm <= 0.:
        raise_ffp_exception(2, verbosity)
        return False
    if z0 is not None and umean is None and z0 <= 0.:
        raise_ffp_exception(3, verbosity)
        return False
    if h <= 10.:
        raise_ffp_exception(4, verbosity)
        return False
    if zm > h:
        raise_ffp_exception(5, verbosity)
        return False
    if z0 is not None and umean is None and zm <= 12.5 * z0:
        if rslayer == 1:
            raise_ffp_exception(6, verbosity)
        else:
            raise_ffp_exception(20, verbosity)
            return False
    if float(zm) / ol <= -15.5:
        raise_ffp_exception(7, verbosity)
        return False
    if sigmav <= 0:
        raise_ffp_exception(8, verbosity)
        return False
    if ustar <= 0.1:
        raise_ffp_exception(9, verbosity)
        return False
    if wind_dir > 360:
        raise_ffp_exception(10, verbosity)
        return False
    if wind_dir < 0:
        raise_ffp_exception(10, verbosity)
        return False
    return True


# ===============================================================================
# ===============================================================================
def get_contour_levels(f, dx, dy, rs=None):
    """
    This method, get_contour_levels, calculates the contour levels of a given array, f, based on specified parameters.

    Parameters:
        - f: 2D numpy array.
        - dx: float, representing the x-axis spacing.
        - dy: float, representing the y-axis spacing.
        - rs: Optional[int | float | List[float]], representing the desired contour ratios. If not provided or invalid, default levels are used.

    Returns:
        - List[Tuple[float, float, float]]: A list of tuples, where each tuple contains three values:
            - The contour ratio, rounded to 3 decimal places.
            - The accumulated area up to that contour ratio.
            - The corresponding contour level value.

    Example usage:
        f = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        dx = 1
        dy = 1
        rs = [0.1, 0.5, 0.9]
        levels = get_contour_levels(f, dx, dy, rs)
        print(levels)

    Output:
        [(0.1, 0.09999999999999998, 9), (0.5, 5.000000000000001, 5), (0.9, 8.899999999999999, 1)]
    """

    import sys

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


def get_contour_vertices(x, y, f, lev):
    """
    Get the x and y coordinates of a contour plot at a specific contour level.

    Parameters:
    - x: Array-like object containing the x values of the contour plot.
    - y: Array-like object containing the y values of the contour plot.
    - f: Array-like object containing the values of the contour plot.
    - lev: Float value representing the contour level.

    Returns:
    - A list containing two arrays: xr and yr. The array xr contains the x coordinates of the contour points, and the array
      yr contains the y coordinates of the contour points.

    Example usage:
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    f = [[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7], [4, 5, 6, 7, 8], [5, 6, 7, 8, 9]]
    lev = 4.5

    result = get_contour_vertices(x, y, f, lev)
    print(result)
    # Output: [[2.5, 3.5, 4.5], [2.5, 3.5, 4.5]]

    Note: This method utilizes the `contour` function from the `matplotlib.pyplot` module.
    """
    # https://stackoverflow.com/questions/5666056/matplotlib-extracting-data-from-contour-lines?noredirect=1&lq=1
    # https://discourse.matplotlib.org/t/collections-attribute-deprecation-in-version-3-8/24164
    # import matplotlib._contour as cntr

    cs = plt.contour(x, y, f, [lev])
    p = cs.collections[0].get_paths()[0]
    v = p.vertices
    xr = v[:, 0]
    yr = v[:, 1]
    plt.close()

    # Set contour to None if it's found to reach the physical domain
    # if x.min() >= min(segs[:, 0]) or max(segs[:, 0]) >= x.max() or \
    #        y.min() >= min(segs[:, 1]) or max(segs[:, 1]) >= y.max():
    #    return [None, None]

    return [xr, yr]  # x,y coords of contour points.


# ===============================================================================
def plot_footprint(x_2d, y_2d, fs, clevs=None, show_heatmap=True, normalize=None,
                   colormap=None, line_width=0.5, iso_labels=None):
    """
    Plot the footprint or contours of a given set of footprints on a 2D grid.

    Parameters:
    - x_2d (numpy.ndarray): 2D grid of x-coordinates.
    - y_2d (numpy.ndarray): 2D grid of y-coordinates.
    - fs (list or numpy.ndarray): The footprints to plot. If fs is a list, only the contour lines will be plotted with different colors. If fs is a single array, the heatmap of the footprint
    * will also be plotted.
    - clevs (list[int] or list[float], optional): The contour levels to plot.
    - show_heatmap (bool, optional): Whether to display the heatmap of the footprint. Default is True.
    - normalize (str, optional): The normalization method for the heatmap. Valid options are 'log' or None. Default is None.
    - colormap (matplotlib.colors.Colormap, optional): The colormap to use for plotting the footprints. Default is jet colormap.
    - line_width (float, optional): The line width for the contour lines. Default is 0.5.
    - iso_labels (list of str, optional): The labels to display for each contour level. The labels should correspond to the contour levels in clevs list.

    Returns:
    - fig (matplotlib.figure.Figure): The generated figure containing the plot.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes of the plot.
    """

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    if colormap is None:
        colormap = cm.get_cmap('jet')
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 8))
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
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors='w', linewidths=line_width)
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
    if show_heatmap:
        if normalize == 'log':
            norm = LogNorm()
        else:
            norm = None

        xmin = np.nanmin(x_2d)
        xmax = np.nanmax(x_2d)
        ymin = np.nanmin(y_2d)
        ymax = np.nanmax(y_2d)
        for f in fs:
            im = ax.imshow(f[:, :], cmap=colormap, extent=(xmin, xmax, ymin, ymax),
                           norm=norm, origin='lower', aspect=1)
            # Colorbar
            cbar = fig.colorbar(im, shrink=1.0, format='%.3e')
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

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
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be larger than zero.'},
    {'code': 3,
     'type': exTypes['error'],
     'msg': 'z0 (roughness length) must be larger than zero.'},
    {'code': 4,
     'type': exTypes['error'],
     'msg': 'h (BPL height) must be larger than 10 m.'},
    {'code': 5,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be smaller than h (PBL height).'},
    {'code': 6,
     'type': exTypes['alert'],
     'msg': 'zm (measurement height) should be above roughness sub-layer (12.5*z0).'},
    {'code': 7,
     'type': exTypes['error'],
     'msg': 'zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.'},
    {'code': 8,
     'type': exTypes['error'],
     'msg': 'sigmav (standard deviation of crosswind) must be larger than zero.'},
    {'code': 9,
     'type': exTypes['error'],
     'msg': 'ustar (friction velocity) must be >=0.1.'},
    {'code': 10,
     'type': exTypes['error'],
     'msg': 'wind_dir (wind direction) must be >=0 and <=360.'},
    {'code': 11,
     'type': exTypes['fatal'],
     'msg': 'Passed data arrays (ustar, zm, h, ol) don\'t all have the same length.'},
    {'code': 12,
     'type': exTypes['fatal'],
     'msg': 'No valid zm (measurement height above displacement height) passed.'},
    {'code': 13,
     'type': exTypes['alert'],
     'msg': 'Using z0, ignoring umean if passed.'},
    {'code': 14,
     'type': exTypes['alert'],
     'msg': 'No valid z0 passed, using umean.'},
    {'code': 15,
     'type': exTypes['fatal'],
     'msg': 'No valid z0 or umean array passed.'},
    {'code': 16,
     'type': exTypes['error'],
     'msg': 'At least one required input is invalid. Skipping current footprint.'},
    {'code': 17,
     'type': exTypes['alert'],
     'msg': 'Only one value of zm passed. Using it for all footprints.'},
    {'code': 18,
     'type': exTypes['fatal'],
     'msg': 'if provided, rs must be in the form of a number or a list of numbers.'},
    {'code': 19,
     'type': exTypes['alert'],
     'msg': 'rs value(s) larger than 90% were found and eliminated.'},
    {'code': 20,
     'type': exTypes['error'],
     'msg': 'zm (measurement height) must be above roughness sub-layer (12.5*z0).'},
]


def raise_ffp_exception(code, verbosity):
    """
    Raise FFP Exception

    Raise a custom exception based on the provided error code.

    Parameters:
        code (int): The error code for the exception.
        verbosity (int): The verbosity level for printing additional information.

    Raises:
        Exception: If the error code matches a fatal exception type, the method raises an exception with the formatted error message.
                   If the error code matches an alert or error exception type, the method prints the formatted error message.

    """

    ex = [it for it in exceptions if it['code'] == code][0]
    string = ex['type'] + '(' + str(ex['code']).zfill(4) + '):\n ' + ex['msg']

    if verbosity > 0:
        print('')

    if ex['type'] == exTypes['fatal']:
        if verbosity > 0:
            string = string + '\n FFP_fixed_domain execution aborted.'
        else:
            string = ''
        raise Exception(string)
    elif ex['type'] == exTypes['alert']:
        string = string + '\n Execution continues.'
        if verbosity > 1:
            print(string)
    elif ex['type'] == exTypes['error']:
        string = string + '\n Execution continues.'
        if verbosity > 1:
            print(string)
    else:
        if verbosity > 1:
            print(string)
