Changelog
=========

0.1.1 
-----

* Fixed the input checks in ``check_ffp_inputs`` (``fluxfootprint/ffp.py``) that never triggered. Since August 2024 they were written with ``.any()``, e.g. ``ustar.any() <= 0.1``, which compares True/False to the threshold. Hours with u* <= 0.1 m/s, zm/L <= -15.5, sigma_v <= 0, or wind direction outside 0-360 degrees now get no footprint, as in the Kljun et al. (2015) FFP code. Daily and monthly ETo weighted footprints that include such hours change.
* ``ffp_climatology`` now also accepts plain Python numbers, and lists with one value per time step, as inputs. Before, inputs had to be numpy values, such as single-valued numpy arrays.

0.1.0
-----

First named version, as an installable Python package ``fluxfootprint``.

* The Kljun et al. (2015) FFP model (``ffp.py``), NLDAS-2 download from NASA Giovanni (``nldas.py``), and helper functions to mask footprints and write georeferenced rasters (``utils.py``), moved from ``scripts/``.
* Install with conda (``environment.yml``) or pip (``pyproject.toml``). 
* The example notebooks and the daily and monthly ETo weighted workflow notebooks (``scripts/Run_AMF_Daily.ipynb``, ``scripts/Run_AMF_Monthly.ipynb``) import from the package. 
* ``find_transform`` computes the raster transform from the pixel size instead of with OpenCV, which rounded UTM coordinates and could shift the raster origin by up to 0.25 m.
* Installation and getting started instructions in the README, and documentation on Read the Docs.
