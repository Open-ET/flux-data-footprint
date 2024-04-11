import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from scipy import signal



class FFPCalc:
    def __init__(
        self,
        zms=None,
        z0s=None,
        umeans=None,
        hs=None,
        ols=None,
        sigmavs=None,
        ustars=None,
        wind_dirs=None,
        rs=None,
        rslayer=None,
        domain=None,
        dx=None,
        dy=None,
        nx=None,
        ny=None,
        crop=None,
        smooth_data=None,
        fig=None,
        **kwargs,
    ):
        """
        Initializes the FootprintCalculator with atmospheric parameters.

        Parameters:
            zm (float): Measurement height.
            z0 (float): Roughness length.
            umean (float): Mean wind speed.
            h (float): Boundary layer height.
            ol (float): Obukhov length.
            sigmav (float): Standard deviation of crosswind.
            ustar (float): Friction velocity.
            wind_dir (float): Wind direction.
            rs (list, optional): Source strength values. Defaults to a range from 0.1 to 0.8.
        """
        self.zms = zms
        self.z0s = z0s
        self.umeans = umeans
        self.hs = hs
        self.ols = ols
        self.sigmavs = sigmavs
        self.ustars = ustars
        self.wind_dirs = wind_dirs
        self.rs = rs if rs is not None else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.frs = None
        self.rslayer = rslayer if rslayer is not None else 0
        self.smooth_data = smooth_data if smooth_data is not None else 1
        self.crop = crop if crop is not None else 0
        self.pulse = 1
        self.fig = fig if fig is not None else 0

        self.x, self.y = None, None
        self.x_2d, self.y_2d = None, None
        self.rho, self.theta = None, None
        self.fclim_2d = None

        # Get kwargs
        self.show_heatmap = kwargs.get("show_heatmap", True)
        self.verbosity = 1

        self.output = {}

        # Model parameters with explanations
        self.model_params = {
            "a": 1.4524,  # Model parameter a explanation
            "b": -1.9914,  # Model parameter b explanation
            "c": 1.4622,
            "d": 0.1359,
            "ac": 2.17,
            "bc": 1.66,
            "cc": 20.0,
            "xstar_end": 30,
            "oln": 5000,  # limit to L for neutral scaling
            "k": 0.4,
        }  # von Karman

        # Initialize other attributes
        self.validate_inputs()
        self.handle_rs()
        self.initialize_domains(domain, dx, dy, nx, ny)
        self.define_physical_domain()

        self.loop_on_time_series()

    def raise_ffp_exception(self, code):
        """

        Raises a custom FFP exception based on the given code.

        Parameters:
            - code: (int) The code of the exception to raise.

        Example:
            raise_ffp_exception(1001)

        """
        self.exTypes = {
            "message": "Message",
            "alert": "Alert",
            "error": "Error",
            "fatal": "Fatal error",
        }

        self.exceptions = [
            {
                "code": 1,
                "type": self.exTypes["fatal"],
                "msg": "At least one required parameter is missing. Please enter all "
                "required inputs. Check documentation for details.",
            },
            {
                "code": 2,
                "type": self.exTypes["error"],
                "msg": "zm (measurement height) must be larger than zero.",
            },
            {
                "code": 3,
                "type": self.exTypes["error"],
                "msg": "z0 (roughness length) must be larger than zero.",
            },
            {
                "code": 4,
                "type": self.exTypes["error"],
                "msg": "h (BPL height) must be larger than 10 m.",
            },
            {
                "code": 5,
                "type": self.exTypes["error"],
                "msg": "zm (measurement height) must be smaller than h (PBL height).",
            },
            {
                "code": 6,
                "type": self.exTypes["alert"],
                "msg": "zm (measurement height) should be above roughness sub-layer (12.5*z0).",
            },
            {
                "code": 7,
                "type": self.exTypes["error"],
                "msg": "zm/ol (measurement height to Obukhov length ratio) must be equal or larger than -15.5.",
            },
            {
                "code": 8,
                "type": self.exTypes["error"],
                "msg": "sigmav (standard deviation of crosswind) must be larger than zero.",
            },
            {
                "code": 9,
                "type": self.exTypes["error"],
                "msg": "ustar (friction velocity) must be >=0.1.",
            },
            {
                "code": 10,
                "type": self.exTypes["error"],
                "msg": "wind_dir (wind direction) must be >=0 and <=360.",
            },
            {
                "code": 11,
                "type": self.exTypes["fatal"],
                "msg": "Passed data arrays (ustar, zm, h, ol) don't all have the same length.",
            },
            {
                "code": 12,
                "type": self.exTypes["fatal"],
                "msg": "No valid zm (measurement height above displacement height) passed.",
            },
            {
                "code": 13,
                "type": self.exTypes["alert"],
                "msg": "Using z0, ignoring umean if passed.",
            },
            {
                "code": 14,
                "type": self.exTypes["alert"],
                "msg": "No valid z0 passed, using umean.",
            },
            {
                "code": 15,
                "type": self.exTypes["fatal"],
                "msg": "No valid z0 or umean array passed.",
            },
            {
                "code": 16,
                "type": self.exTypes["error"],
                "msg": "At least one required input is invalid. Skipping current footprint.",
            },
            {
                "code": 17,
                "type": self.exTypes["alert"],
                "msg": "Only one value of zm passed. Using it for all footprints.",
            },
            {
                "code": 18,
                "type": self.exTypes["fatal"],
                "msg": "if provided, rs must be in the form of a number or a list of numbers.",
            },
            {
                "code": 19,
                "type": self.exTypes["alert"],
                "msg": "rs value(s) larger than 90% were found and eliminated.",
            },
            {
                "code": 20,
                "type": self.exTypes["error"],
                "msg": "zm (measurement height) must be above roughness sub-layer (12.5*z0).",
            },
        ]

        ex = [it for it in self.exceptions if it["code"] == code][0]
        string = ex["type"] + "(" + str(ex["code"]).zfill(4) + "):\n " + ex["msg"]

        print("")

        if self.verbosity > 0:
            print("")

        if ex["type"] == self.exTypes["fatal"]:
            if self.verbosity > 0:
                string = string + "\n FFP_fixed_domain execution aborted."
            else:
                string = ""
            raise Exception(string)
        elif ex["type"] == self.exTypes["alert"]:
            string = string + "\n Execution continues."
            if self.verbosity > 1:
                print(string)
        elif ex["type"] == self.exTypes["error"]:
            string = string + "\n Execution continues."
            if self.verbosity > 1:
                print(string)
        else:
            if self.verbosity > 1:
                print(string)

    def initialize_domains(self, domain, dx, dy, nx, ny):
        """
        Initializes computational and physical domains based on the provided parameters.
        """
        # Domain initialization logic...
        # domain variables
        self.domain = domain
        self.dx = dx
        self.dy = dy
        self.nx = nx
        self.ny = ny

        self.xrs = []
        self.yrs = []

        # Define computational domain
        # Check passed values and make some smart assumptions
        if isinstance(self.dx, (float, int)) and self.dy is None:
            self.dy = self.dx
        if isinstance(self.dy, (float, int)) and self.dx is None:
            self.dx = self.dy
        if not all(isinstance(item, float) for item in [self.dx, self.dy]) or not all(
            isinstance(item, int) for item in [self.dx, self.dy]
        ):
            self.dx = self.dy = None
        if isinstance(self.nx, int) and self.ny is None:
            self.ny = self.nx
        if not (not isinstance(self.ny, int) or not (self.nx is None)):
            self.nx = self.ny
        if not all(isinstance(item, int) for item in [self.nx, self.ny]):
            self.nx = self.ny = None
        if not isinstance(self.domain, list) or len(self.domain) != 4:
            self.domain = None

        if all(item is None for item in [self.dx, self.nx, self.domain]):
            # If nothing is passed, default domain is a square of 2 Km size centered
            # at the tower with pixel size of 2 meters (hence a 1000x1000 grid)
            self.domain = [-1000.0, 1000.0, -1000.0, 1000.0]
            self.dx = self.dy = 2.0
            self.nx = self.ny = 1000
        elif self.domain is not None:
            # If domain is passed, it takes the precedence over anything else
            if self.dx is not None:
                # If dx/dy is passed, takes precedence over nx/ny
                self.nx = int((self.domain[1] - self.domain[0]) / self.dx)
                self.ny = int((self.domain[3] - self.domain[2]) / self.dy)
            else:
                # If dx/dy is not passed, use nx/ny (set to 1000 if not passed)
                if self.nx is None:
                    self.nx = self.ny = 1000
                # If dx/dy is not passed, use nx/ny
                self.dx = (self.domain[1] - self.domain[0]) / float(self.nx)
                self.dy = (self.domain[3] - self.domain[2]) / float(self.ny)
        elif self.dx is not None and self.nx is not None:
            # If domain is not passed but dx/dy and nx/ny are, define domain
            self.domain = [
                -self.nx * self.dx / 2,
                self.nx * self.dx / 2,
                -self.ny * self.dy / 2,
                self.ny * self.dy / 2,
            ]
        elif self.dx is not None:
            # If domain is not passed but dx/dy is, define domain and nx/ny
            self.domain = [-1000, 1000, -1000, 1000]
            self.nx = int((self.domain[1] - self.domain[0]) / self.dx)
            self.ny = int((self.domain[3] - self.domain[2]) / self.dy)
        elif self.nx is not None:
            # If domain and dx/dy are not passed but nx/ny is, define domain and dx/dy
            self.domain = [-1000, 1000, -1000, 1000]
            self.dx = (self.domain[1] - self.domain[0]) / float(self.nx)
            self.dy = (self.domain[3] - self.domain[2]) / float(self.nx)

        # Put domain into more convenient vars
        self.xmin, self.xmax, self.ymin, self.ymax = self.domain

    def validate_inputs(self):
        """
        Validates the input parameters to ensure they meet the expected criteria.
        """

        # Input check
        flag_err = 0

        # Check existence of required input pars
        if None in [self.zms, self.hs, self.ols, self.sigmavs, self.ustars] or (
            self.z0s is None and self.umeans is None):
            self.raise_ffp_exception(1)

        # Convert all input items to lists
        if not isinstance(self.zms, list):
            self.zms = [self.zms]
        if not isinstance(self.hs, list):
            self.hs = [self.hs]
        if not isinstance(self.ols, list):
            self.ols = [self.ols]
        if not isinstance(self.sigmavs, list):
            self.sigmavs = [self.sigmavs]
        if not isinstance(self.ustars, list):
            self.ustars = [self.ustars]
        if not isinstance(self.wind_dirs, list):
            self.wind_dirs = [self.wind_dirs]
        if not isinstance(self.z0s, list):
            self.z0s = [self.z0s]
        if not isinstance(self.umeans, list):
            self.umeans = [self.umeans]

        # Check that all lists have same length, if not raise an error and exit
        self.ts_len = len(self.ustars)
        if self.ts_len > 20:
            self.pulse = int(self.ts_len / 20)

        if any(len(lst) != self.ts_len for lst in [self.sigmavs, self.wind_dirs, self.hs, self.ols]):
            # at least one list has a different length, exit with error message
            self.raise_ffp_exception(11)

        # Special treatment for zm, which is allowed to have length 1 for any
        # length >= 1 of all other parameters
        if all(val is None for val in self.zms):
            self.raise_ffp_exception(12)
        if len(self.zms) == 1:
            self.raise_ffp_exception(17)
            self.zms = self.zms * self.ts_len

        # Resolve ambiguity if both z0 and umean are passed (defaults to using z0)
        # If at least one value of z0 is passed, use z0 (by setting umean to None)
        if not all(val is None for val in self.z0s):
            self.raise_ffp_exception(13)
            self.umeans = [None] * self.ts_len
            # If only one value of z0 was passed, use that value for all footprints
            if len(self.z0s) == 1:
                self.z0s = [self.z0s[0]] * self.ts_len
        elif len(self.umeans) == self.ts_len and not all(
            val is None for val in self.umeans
        ):
            self.raise_ffp_exception(14)
            self.z0s = [None] * self.ts_len
        else:
            self.raise_ffp_exception(15)


    def handle_rs(self):
        # ===========================================================================
        # Handle rs
        if self.rs is not None:

            # Check that rs is a list, otherwise make it a list
            if isinstance(self.rs, float) or isinstance(self.rs, int):
                if 0.9 < self.rs <= 1 or 90 < self.rs <= 100:
                    self.rs = 0.9
                self.rs = [self.rs]
            if not isinstance(self.rs, list):
                self.raise_ffp_exception(18)

            # If rs is passed as percentages, normalize to fractions of one
            if np.max(self.rs) >= 1:
                self.rs = [x / 100.0 for x in self.rs]

            # Eliminate any values beyond 0.9 (90%) and inform user
            if np.max(self.rs) > 0.9:
                self.raise_ffp_exception(19)
                self.rs = [item for item in self.rs if item <= 0.9]

            # Sort levels in ascending order
            self.rs = list(np.sort(self.rs))

    def define_physical_domain(self):
        # ===========================================================================
        # Define physical domain in cartesian and polar coordinates
        # Cartesian coordinates
        self.x = np.linspace(self.xmin, self.xmax, self.nx + 1)
        self.y = np.linspace(self.ymin, self.ymax, self.ny + 1)
        self.x_2d, self.y_2d = np.meshgrid(self.x, self.y)

        # Polar coordinates
        # Set theta such that North is pointing upwards and angles increase clockwise
        self.rho = np.sqrt(self.x_2d**2 + self.y_2d**2)
        self.theta = np.arctan2(self.x_2d, self.y_2d)
        self.rotated_theta = self.theta #added for if valid is none before real_scaledxst
        # initialize raster for footprint climatology
        self.fclim_2d = np.zeros(self.x_2d.shape)

    def real_scaledxst(self, ix, z0, ol, zm, h, umean, ustar):
        # ===========================================================================
        # Create real scale crosswind integrated footprint and dummy for
        # rotated scaled footprint
        self.fstar_ci_dummy = np.zeros(self.x_2d.shape)
        self.f_ci_dummy = np.zeros(self.x_2d.shape)
        self.xstar_ci_dummy = np.zeros(self.x_2d.shape)
        self.px = np.ones(self.x_2d.shape)
        if z0 is not None:
            if ol > 0 and ol < self.model_params["oln"]:  # ol > 0 and ol < oln:
                self.psi_f = -5.3 * zm / ol
            # Use z0
            else:
                xx = (1 - 19.0 * zm / ol) ** 0.25
                self.psi_f = (np.log((1 + xx**2) / 2.0) + 2.0 * np.log((1 + xx) / 2.0) - 2.0 * np.arctan(xx) + np.pi / 2)

            if (np.log(zm / z0) - self.psi_f) > 0:
                self.xstar_ci_dummy = (
                    self.rho * np.cos(self.rotated_theta)
                    / zm * (1.0 - (zm / h)) / (np.log(zm / z0) - self.psi_f))
                self.px = np.where(self.xstar_ci_dummy > self.model_params["d"])
                self.fstar_ci_dummy[self.px] = (
                    self.model_params["a"]
                    * (self.xstar_ci_dummy[self.px] - self.model_params["d"])
                    ** self.model_params["b"]
                    * np.exp(
                        -self.model_params["c"]
                        / (
                            self.xstar_ci_dummy[self.px]
                            - self.model_params["d"]
                        )
                    )
                )
                self.f_ci_dummy[self.px] = (
                    self.fstar_ci_dummy[self.px]
                    / zm
                    * (1.0 - (zm / h))
                    / (np.log(zm / z0) - self.psi_f)
                )
            else:
                flag_err = 3
                self.valids[ix] = 0
        else:
            # Use umean if z0 not available
            self.xstar_ci_dummy = (
                self.rho
                * np.cos(self.rotated_theta)
                / zm
                * (1.0 - (zm / h))
                / (umean / ustar * self.model_params["k"])
            )
            self.px = np.where(self.xstar_ci_dummy > self.model_params["d"])
            self.fstar_ci_dummy[self.px] = (
                self.model_params["a"]
                * (self.xstar_ci_dummy[self.px] - self.model_params["d"])
                ** self.model_params["b"]
                * np.exp(
                    -self.model_params["c"]
                    / (self.xstar_ci_dummy[self.px] - self.model_params["d"])
                )
            )
            self.f_ci_dummy[self.px] = (
                self.fstar_ci_dummy[self.px]
                / zm
                * (1.0 - (zm / h))
                / (umean / ustar * self.model_params["k"])
            )

    def check_ffp_inputs(self, zm, z0, h, ol, ustar, sigmav, umean, wind_dir):
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
        if zm <= 0.0:
            self.raise_ffp_exception(2)
            return False
        if z0 is not None and umean is None and z0 <= 0.0:
            self.raise_ffp_exception(3)
            return False
        if h <= 10.0:
            self.raise_ffp_exception(4)
            return False
        if zm > h:
            self.raise_ffp_exception(5)
            return False
        if z0 is not None and umean is None and zm <= 12.5 * z0:
            if self.rslayer == 1:
                self.raise_ffp_exception(6)
            else:
                self.raise_ffp_exception(20)
                return False
        if float(zm) / ol <= -15.5:
            self.raise_ffp_exception(7)
            return False
        if sigmav <= 0:
            self.raise_ffp_exception(8)
            return False
        if ustar <= 0.1:
            self.raise_ffp_exception(9)
            return False
        if wind_dir > 360:
            self.raise_ffp_exception(10)
            return False
        if wind_dir < 0:
            self.raise_ffp_exception(10)
            return False
        return True

    @staticmethod
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
        msf = np.ma.masked_array(
            sf, mask=(np.isnan(sf) | np.isinf(sf))
        )  # Masked array for handling potential nan
        csf = msf.cumsum().filled(np.nan) * dx * dy
        for ix, r in enumerate(rs):
            dcsf = np.abs(csf - r)
            pclevs[ix] = sf[np.nanargmin(dcsf)]
            ars[ix] = csf[np.nanargmin(dcsf)]

        return [(round(r, 3), ar, pclev) for r, ar, pclev in zip(rs, ars, pclevs)]

    @staticmethod
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

    def derive_footprint_ellipsoid(self):
        """
        Derive footprint ellipsoid incorporating R% of the flux, if requested, starting at peak value.

        Parameters:
        - self: The current object instance

        Returns: None

        This method calculates the contour levels for the given 2D flux data and stores the corresponding vertices of the contours in the `xrs` and `yrs` lists. If the `rs` parameter is provided
        *, the contour levels are derived using the specified percentage of flux. Otherwise, if the `crop` parameter is True, the contour levels are derived using a default percentage of 80
        *%.

        Example usage:
        obj = ClassName()
        obj.derive_footprint_ellipsoid()
        """
        # ===========================================================================
        # Derive footprint ellipsoid incorporating R% of the flux, if requested,
        # starting at peak value.
        #self.dy = self.dx
        if self.rs is not None:
            clevs = self.get_contour_levels(self.f_2d, self.dx, self.dy, self.rs)
            self.frs = [item[2] for item in clevs]

            for ix, fr in enumerate(self.frs):
                xr, yr = self.get_contour_vertices(self.x_2d, self.y_2d, self.f_2d, fr)
                if xr is None:
                    self.frs[ix] = None
                self.xrs.append(xr)
                self.yrs.append(yr)
        else:
            if self.crop:
                rs_dummy = 0.8  # crop to 80%
                clevs = self.get_contour_levels(self.fclim_2d, self.dx, self.dy, rs_dummy)
                #self.xrs = []
                #self.yrs = []
                self.xrs, self.yrs = self.get_contour_vertices(
                    self.x_2d, self.y_2d, self.f_2d, clevs[0][2]
                )

    def crop_footprint_ellipsoid(self):
        # ===========================================================================
        # Crop domain and footprint to the largest rs value
        if self.crop:
            xrs_crop = [x for x in self.xrs if x is not None]
            yrs_crop = [x for x in self.yrs if x is not None]
            if self.rs is not None:
                dminx = np.floor(min(xrs_crop[-1]))
                dmaxx = np.ceil(max(xrs_crop[-1]))
                dminy = np.floor(min(yrs_crop[-1]))
                dmaxy = np.ceil(max(yrs_crop[-1]))

            else:
                dminx = np.floor(min(xrs_crop))
                dmaxx = np.ceil(max(xrs_crop))
                dminy = np.floor(min(yrs_crop))
                dmaxy = np.ceil(max(yrs_crop))

            if dminy >= self.ymin and dmaxy <= self.ymax:
                jrange = np.where((self.y_2d[:, 0] >= dminy) & (self.y_2d[:, 0] <= dmaxy))[0]
                jrange = np.concatenate(([jrange[0] - 1], jrange, [jrange[-1] + 1]))
                jrange = jrange[np.where((jrange >= 0) & (jrange <= self.y_2d.shape[0]))[0]]
            else:
                jrange = np.linspace(0, 1, self.y_2d.shape[0] - 1)

            if dminx >= self.xmin and dmaxx <= self.xmax:
                irange = np.where((self.x_2d[0, :] >= dminx) & (self.x_2d[0, :] <= dmaxx))[0]
                irange = np.concatenate(([irange[0] - 1], irange, [irange[-1] + 1]))
                irange = irange[np.where((irange >= 0) & (irange <= self.x_2d.shape[1]))[0]]
            else:
                irange = np.linspace(0, 1, self.x_2d.shape[1] - 1)

            jrange = [[it] for it in jrange]
            self.x_2d = self.x_2d[jrange, irange]
            self.y_2d = self.y_2d[jrange, irange]
            self.f_2d = self.f_2d[jrange, irange]
            self.fclim_2d = self.fclim_2d[jrange, irange]

    def plot_footprint(
        self,
        x_2d,
        y_2d,
        fs,
        clevs=None,
        show_heatmap=True,
        normalize=None,
        colormap=None,
        line_width=0.5,
        iso_labels=None,
    ):
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
            colormap = cm.get_cmap("jet")
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
                    cp = ax.contour(
                        x_2d, y_2d, f, levs, colors="w", linewidths=line_width
                    )
                else:
                    cp = ax.contour(
                        x_2d, y_2d, f, levs, colors=cc, linewidths=line_width
                    )
                # Isopleth Labels
                if iso_labels is not None:
                    pers = [str(int(clev[0] * 100)) + "%" for clev in clevs]
                    fmt = {}
                    for l, s in zip(cp.levels, pers):
                        fmt[l] = s
                    plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

        # plot footprint heatmap if requested and if only one footprint is passed
        if show_heatmap:
            if normalize == "log":
                norm = LogNorm()
            else:
                norm = None

            xmin = np.nanmin(x_2d)
            xmax = np.nanmax(x_2d)
            ymin = np.nanmin(y_2d)
            ymax = np.nanmax(y_2d)
            for f in fs:
                im = ax.imshow(
                    f[:, :],
                    cmap=colormap,
                    extent=(xmin, xmax, ymin, ymax),
                    norm=norm,
                    origin="lower",
                    aspect=1,
                )
                # Colorbar
                cbar = fig.colorbar(im, shrink=1.0, format="%.3e")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")

            # cbar.set_label('Flux contribution', color = 'k')
        plt.show()

        return fig, ax

    def loop_on_time_series(self):
        # ===========================================================================
        # Loop on time series

        # Initialize logic array valids to those 'timestamps' for which all inputs are
        # at least present (but not necessarily phisically plausible)
        self.valids = [
            True if not any([val is None for val in vals]) else False
            for vals in zip(
                self.ustars, self.sigmavs, self.hs, self.ols, self.wind_dirs, self.zms
            )
        ]

        for ix, (ustar, sigmav, h, ol, wind_dir, zm, z0, umean,) in enumerate(
            zip(
                self.ustars,
                self.sigmavs,
                self.hs,
                self.ols,
                self.wind_dirs,
                self.zms,
                self.z0s,
                self.umeans,
            )
        ):

            # Counter
            if ix % self.pulse == 0:
                print("Calculating footprint ", ix + 1, " of ", self.ts_len)

            self.valids[ix] = self.check_ffp_inputs(zm, z0, h, ol, ustar, sigmav, umean, wind_dir)

            # If inputs are not valid, skip current footprint
            if not self.valids[ix]:
                self.raise_ffp_exception(16)
            else:
                # ===========================================================================
                # Rotate coordinates into wind direction
                if wind_dir is not None:
                    self.rotated_theta = self.theta - wind_dir * np.pi / 180.0
                else:
                    self.rotated_theta = self.theta

            self.real_scaledxst(ix, z0, ol, zm, h, umean, ustar)

            # ===========================================================================
            # Calculate dummy for scaled sig_y* and real scale sig_y
            sigystar_dummy = np.zeros(self.x_2d.shape)
            sigystar_dummy[self.px] = self.model_params.get("ac", 1) * np.sqrt(
                self.model_params.get("bc", 1)
                * np.abs(self.xstar_ci_dummy[self.px]) ** 2
                / (
                    1
                    + self.model_params.get("cc", 1)
                    * np.abs(self.xstar_ci_dummy[self.px])
                )
            )

            if abs(ol) > self.model_params.get("oln", 1):
                ol = -1e6
            if ol <= 0:  # convective
                scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.80
            else: #if ol > 0:  # stable
                scale_const = 1e-5 * abs(zm / ol) ** (-1) + 0.55
            if scale_const > 1:
                scale_const = 1.0

            sigy_dummy = np.zeros(self.x_2d.shape)
            sigy_dummy[self.px] = (
                sigystar_dummy[self.px]
                / scale_const
                * zm
                * sigmav
                / ustar
            )
            sigy_dummy[sigy_dummy < 0] = np.nan

            # ===========================================================================
            # Calculate real scale f(x,y)
            self.f_2d = np.zeros(self.x_2d.shape)
            self.f_2d[self.px] = (
                self.f_ci_dummy[self.px]
                / (np.sqrt(2 * np.pi) * sigy_dummy[self.px])
                * np.exp(
                    -((self.rho[self.px] * np.sin(self.rotated_theta[self.px])) ** 2)
                    / (2.0 * sigy_dummy[self.px] ** 2)
                )
            )

            # ===========================================================================
            # Add to footprint climatology raster
            self.fclim_2d = self.fclim_2d + self.f_2d

        # ===========================================================================
        # Continue if at least one valid footprint was calculated
        n = np.sum(self.valids)
        vs = None
        clevs = None

        # frs = []
        flag_err = 0
        if n == 0:
            print("No footprint calculated")
            flag_err = 1
        else:
            self.fclim_2d = self.fclim_2d / n
            self.normalize_and_smooth_footprint()
            self.derive_footprint_ellipsoid()
            self.crop_footprint_ellipsoid()

            # ===========================================================================
            # Plot footprint
            if self.fig:
                fig_out, ax = self.plot_footprint(
                    x_2d=self.x_2d,
                    y_2d=self.y_2d,
                    fs=self.fclim_2d,
                    show_heatmap=self.show_heatmap,
                    clevs=self.frs,
                )
        self.output["x_2d"] = self.x_2d
        self.output["y_2d"] = self.y_2d
        self.output["fclim_2d"] = self.fclim_2d
        self.output["rs"] = self.rs
        self.output["frs"] = self.frs
        self.output["xr"] = self.xrs
        self.output["yr"] = self.yrs
        self.output["n"] = n
        self.output["flag_err"] = flag_err
        # ===========================================================================
        # Fill output structure

    def normalize_and_smooth_footprint(self):
        # ===========================================================================
        # Normalize and smooth footprint climatology

        if self.smooth_data is not None:
            skernel = np.matrix("0.05 0.1 0.05; 0.1 0.4 0.1; 0.05 0.1 0.05")
            self.fclim_2d = signal.convolve2d(self.fclim_2d, skernel, mode="same")
            self.fclim_2d = signal.convolve2d(self.fclim_2d, skernel, mode="same")
