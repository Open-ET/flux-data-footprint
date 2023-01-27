
flux-data-footprint
===================

This package uses the `Kljun et al., 2015 <https://gmd.copernicus.org/articles/8/3695/2015/>`_, 2-dimensional, physically-based, flux footprint model to create daily and monthly weighted footprints from hourly time series inputs from eddy flux towers. The package also downloads hourly reference evapotranspiration (ET) drivers (temperature, radiation, vapor pressure, and wind speed) from NLDAS version 2 and computes the ASCE Standadrized Penman Monteith reference ET which is used for weighting hourly footprint predictions which are subsequently averaged over daily or monthly periods.

Documentation
-------------

Under development, in the meantime please view the example Jupyter Notebooks, starting with the Celery flux tower example which explains the required input data clearly, the other example shows how to use the `flux-data-qaqc <https://github.com/Open-ET/flux-data-qaqc>`__ Python package to aid in preparation of the input data. **Note**, neither of these examples employ NLDAS refernce ET for weighting the results. For those workflows please see the submodules under the "scripts" folder until further documentation is complte.


Acknowledgemnts and contributions
---------------------------------
* Martin Schroeder (conversion of footprint data to georeferenced raster and basic development)
* Dr. Richard Allen and David Ekhardt (for reference ET weighting approach)
* The OpenET science team (general guidance)
* Dr. Natascha Kljun (development of main footprint prediction code)

How to cite
-----------

Volk, J. M., Huntington, J., Melton, F. S., Allen, R., Anderson, M. C., Fisher, J. B., ... & Kustas, W. (2023). Development of a Benchmark Eddy Flux Evapotranspiration Dataset for Evaluation of Satellite-Driven Evapotranspiration Models Over the CONUS. Agricultural and Forest Meteorology (331), https://doi.org/10.1016/j.agrformet.2023.109307.

