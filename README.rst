
flux-data-footprint
===================

This package uses the `Kljun et al., 2015 <https://gmd.copernicus.org/articles/8/3695/2015/>`_, 2-dimensional, physically-based, flux footprint model to create daily and monthly weighted footprints from hourly time series inputs from eddy flux towers. The package also downloads hourly reference evapotranspiration (ET) drivers (temperature, radiation, vapor pressure, and wind speed) from NLDAS version 2 and computes the ASCE Standardized Penman Monteith reference ET which is used for weighting hourly footprint predictions which are subsequently averaged over daily or monthly periods.

Documentation
-------------

`ReadTheDocs <https://flux-data-footprint.readthedocs.io/>`__

Installation
------------

The package is not on PyPI yet, so it is installed from a copy of this repository. Conda is recommended because rasterio and pyproj install most easily from conda-forge:

.. code-block:: bash

    git clone https://github.com/Open-ET/flux-data-footprint.git
    cd flux-data-footprint
    conda env create -f environment.yml
    conda activate ffp

``environment.yml`` installs the dependencies from conda-forge and then installs ``fluxfootprint`` from the cloned folder in editable mode, so changes to the code take effect without reinstalling. To check the install:

.. code-block:: bash

    python -c "import fluxfootprint; print(fluxfootprint.__version__)"

To run the example notebooks also install Jupyter, and for the AmeriFlux example the `flux-data-qaqc <https://github.com/Open-ET/flux-data-qaqc>`__ package:

.. code-block:: bash

    conda install -c conda-forge jupyterlab
    pip install fluxdataqaqc "numpy<2.4" "refet<0.5"

Without conda, install with pip (Python 3.9 or newer) from the cloned folder:

.. code-block:: bash

    pip install -e .

**Note**, numpy is limited to versions below 2.4 and refet to versions below 0.5 because the footprint and reference ET code is not yet compatible with newer versions.

Getting started
---------------

The main functions are imported from the package:

* ``ffp_climatology``, the Kljun et al. (2015) footprint model
* ``mask_fp_cutoff``, keeps the part of a footprint that makes up a set fraction (default 90%) of the source area
* ``find_transform``, the affine transform used to write a footprint grid to a georeferenced GeoTIFF with rasterio
* ``get_nldas2_data``, downloads hourly NLDAS-2 forcing data for a point from NASA Giovanni

A footprint for one half hour at the Celery example site:

.. code-block:: python

    import numpy as np
    from fluxfootprint import ffp_climatology, mask_fp_cutoff

    dx = 3.0  # grid resolution [m]
    fp = ffp_climatology(
        zm=1.24,         # measurement height minus displacement height [m]
        z0=0.015,        # roughness length [m]
        umean=None,
        h=2000.0,        # boundary layer height [m]
        ol=213.0,        # Obukhov length [m]
        sigmav=0.60,     # standard deviation of lateral wind speed [m s-1]
        ustar=0.35,      # friction velocity [m s-1]
        wind_dir=19.8,   # wind direction [degrees from north]
        domain=[-75, 75, -75, 75], dx=dx, dy=dx, rs=None, fig=False, verbosity=0,
    )
    f_2d = np.array(fp['fclim_2d']) * dx**2  # footprint weight per grid cell
    f_2d = mask_fp_cutoff(f_2d, cutoff=0.9)  # keep the 90% source area

See the `documentation <https://flux-data-footprint.readthedocs.io/en/latest/getting_started.html>`__ for plotting, writing a georeferenced raster, and footprints over several time steps.

Examples
~~~~~~~~

The example Jupyter notebooks are in the ``example notebooks`` folder:

* ``celery_site/celery.ipynb`` is the simplest place to start. It shows the input data the footprint model needs and makes half-hourly footprints centered on the tower.
* ``using_fluxdataqaqc_with_AMF/example with flux-data-qaqc.ipynb`` prepares AmeriFlux data with flux-data-qaqc and writes a georeferenced (UTM) GeoTIFF with one band per hour for a day.

Neither example uses NLDAS reference ET to weight the footprints.

Acknowledgements and contributions
----------------------------------
* Martin Schroeder (conversion of footprint data to georeferenced raster and basic development)
* Dr. Richard Allen and David Ekhardt (for reference ET weighting approach)
* The OpenET science team (general guidance)
* Dr. Natascha Kljun (development of main footprint prediction code)

How to cite
-----------

Volk, J. M., Huntington, J., Melton, F. S., Allen, R., Anderson, M. C., Fisher, J. B., ... & Kustas, W. (2023). Development of a Benchmark Eddy Flux Evapotranspiration Dataset for Evaluation of Satellite-Driven Evapotranspiration Models Over the CONUS. Agricultural and Forest Meteorology (331), https://doi.org/10.1016/j.agrformet.2023.109307.

