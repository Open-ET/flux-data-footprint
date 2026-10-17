# -*- coding: utf-8 -*-
"""
Flux footprint rasters for eddy covariance towers.

Uses the Kljun et al. (2015) two-dimensional flux footprint prediction
(FFP) model to create georeferenced footprint rasters from hourly tower
data, with optional weighting by hourly NLDAS-2 reference ET.

Example:
    >>> from fluxfootprint import ffp_climatology, mask_fp_cutoff
"""

from .ffp import ffp_climatology
from .nldas import get_nldas2_data
from .utils import find_transform, mask_fp_cutoff

__version__ = '0.1.0'
