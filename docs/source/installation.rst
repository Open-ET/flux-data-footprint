Installation
============

flux-data-footprint is not on PyPI yet, so it is installed from a copy of the GitHub repository. It needs Python 3.9 or newer. Conda is recommended because rasterio and pyproj install most easily from conda-forge:

.. code-block:: bash

    git clone https://github.com/Open-ET/flux-data-footprint.git
    cd flux-data-footprint
    conda env create -f environment.yml
    conda activate ffp

``environment.yml`` installs the dependencies from conda-forge and then installs ``fluxfootprint`` from the cloned folder in editable mode, so changes to the code take effect without reinstalling. Without conda, install with pip from the cloned folder:

.. code-block:: bash

    pip install -e .

To check the installation:

.. code-block:: bash

    python -c "import fluxfootprint; print(fluxfootprint.__version__)"

To run the example notebooks also install Jupyter, and for the AmeriFlux example the `flux-data-qaqc <https://github.com/Open-ET/flux-data-qaqc>`__ package:

.. code-block:: bash

    conda install -c conda-forge jupyterlab
    pip install fluxdataqaqc "numpy<2.4" "refet<0.5"

numpy is limited to versions below 2.4 and refet to versions below 0.5 because the footprint and reference ET code is not yet compatible with newer versions. The other dependencies are affine, matplotlib, pandas, pyproj, rasterio, requests and scipy.
