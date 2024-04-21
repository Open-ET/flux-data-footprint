import unittest
from scripts import ffp
import numpy as np


class TestDefinePhysicalDomain(unittest.TestCase):

    def setUp(self):
        # Sample data setup
        data = {'zm': np.array([1.2441298]), 'z0': np.array([0.01854]), 'umean': np.array([6.083]), 'h': 2000,
                'ol': np.array([-54]), 'sigmav': np.array([1.36]), 'ustar': np.array([0.53]),'fig':True,
                'wind_dir': np.array([59.]), 'dx': 100, 'dy': 100, 'crop': True}
        self.ffptest = ffp.ffp_climatology(**data)
    def test_initialization(self):
        # Test that the instance is initialized correctly
        self.assertEqual(self.ffptest.a, 1.4524)
    def test_define_physical_domain(self):
        # Define test inputs
        xmin, xmax = -100, 100
        ymin, ymax = -100, 100
        nx, ny = 200, 200
        
        # Run the method with the test inputs
        results = self.ffptest.define_physical_domain(xmin, xmax, ymin, ymax, nx, ny)

        # Check the shapes of the outputs
        self.assertEqual(results[0].shape, (ny + 1, nx + 1))  # x_2d
        self.assertEqual(results[2].shape, (ny + 1, nx + 1))  # rho
        self.assertEqual(results[3].shape, (ny + 1, nx + 1))  # theta
        self.assertEqual(results[4].shape, (ny + 1, nx + 1))  # fclim_2d

        # Validate the values in the x and y arrays
        #np.testing.assert_array_almost_equal(results[0][:, 0], np.linspace(xmin, xmax, nx + 1))  # x coordinates
        #np.testing.assert_array_almost_equal(results[1][0, :], np.linspace(ymin, ymax, ny + 1))  # y coordinates

    def test_negative_dimensions(self):
        with self.assertRaises(ValueError):
            self.ffptest.define_physical_domain(-100, 100, -50, 50, -200, 100)

        with self.assertRaises(ValueError):
            self.ffptest.define_physical_domain(-100, 100, -50, 50, 200, -100)



if __name__ == '__main__':
    unittest.main()