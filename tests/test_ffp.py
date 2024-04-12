import unittest
import sys
import os

# Add the ffp folder path to the sys.path list
sys.path.append(os.path.abspath("/scripts/"))

from ffp import FFPCalc


class TestFFPCalc(unittest.TestCase):

    def setUp(self):
        self.instance = FFPCalc(zm=1.2441298,z0=0.01854008,umean=6.083972,h=2000,
                                ol=-54.00268,sigmav=1.360829,ustar=0.5258447,wind_dir=58.98489,
                                fig=True,dx=100,dy=100,crop=True)

    def test_init(self):
        self.assertIsNotNone(self.instance)

    def test_raise_ffp_exception(self):
        with self.assertRaises(Exception):
            self.instance.raise_ffp_exception(1)

    def test_initialize_domains(self):
        domain = (0, 0, 10, 10)
        dx = 0.1
        dy = 0.1
        nx = 10
        ny = 10
        self.instance.initialize_domains(domain, dx, dy, nx, ny)
        self.assertIsNotNone(self.instance)

    def test_handle_rs(self):
        self.instance.handle_rs()
        self.assertIsNotNone(self.instance)

    def test_define_physical_domain(self):
        self.instance.define_physical_domain()
        self.assertIsNotNone(self.instance)

    def test_real_scaledxst(self):
        with self.assertRaises(Exception):
            self.instance.real_scaledxst(1, None, None, None, 1, 1, 1)

    def test_check_ffp_inputs(self):
        with self.assertRaises(Exception):
            self.instance.check_ffp_inputs(None, 1, 1, 1, 1, 1, 1, 1)

    def test_get_contour_vertices(self):
        with self.assertRaises(Exception):
            self.instance.get_contour_vertices(1, 1, 10, 1)

    def test_derive_footprint_ellipsoid(self):
        self.instance.derive_footprint_ellipsoid()
        self.assertIsNotNone(self.instance)

    def test_crop_footprint_ellipsoid(self):
        self.instance.crop_footprint_ellipsoid()
        self.assertIsNotNone(self.instance)

    def test_normalize_and_smooth_footprint(self):
        self.instance.normalize_and_smooth_footprint()
        self.assertIsNotNone(self.instance)
