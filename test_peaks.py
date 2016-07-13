import unittest
import numpy as np

import pandas as pd

from cases import ScimapTestCase
from peakfitting import Peak, discrete_fwhm

# flake8: noqa

"""Module for testing generic peak fitting. Some techniques have more
specific tests (eg X-ray diffraction, X-ray microscopy."""

class GuessParameters(ScimapTestCase):

    def test_single_peak(self):
        peak = Peak(num_peaks=1, method="Gaussian")
        y = np.array([0, 3, 9, 3, 0])
        x = np.arange(0, len(y))
        guess = peak.guess_parameters(x=x, y=y)
        self.assertEqual(len(guess), 1)
        self.assertEqual(guess[0].height, 9)  # Height
        self.assertEqual(guess[0].center, 2)  # Center
        self.assertApproximatelyEqual(guess[0].width, 0.85)  # Width

    def test_uneven_peak(self):
        peak = Peak(num_peaks=1, method="Gaussian")
        y = np.array([0, 3, 9, 8, 6])
        x = np.arange(0, len(y))
        guess = peak.guess_parameters(x=x, y=y)
        self.assertApproximatelyEqual(guess[0].width, 0.85)

    def test_two_peaks(self):
        pass

class FullWidthHalfMax(ScimapTestCase):

    def test_calculate_fwhm(self):
        y = np.array([0, 5.9, 6.1, 12, 6.1, 5.9, 0])
        x = np.arange(0, len(y))
        self.assertEqual(discrete_fwhm(x, y), 4)


if __name__ == "__main__":
    unittest.main()
