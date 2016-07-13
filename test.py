# -*- coding: utf-8 -*-
#
# Copyright © 2016 Mark Wolf
#
# This file is part of scimap.
#
# Scimap is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Scimap is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Scimap.  If not, see <http://www.gnu.org/licenses/>.

import unittest

# Set backend so matplotlib doesn't try and show plots
import matplotlib
matplotlib.use('Agg')

if __name__ == '__main__':

    # Look for tests in files in subdirectories
    runner = unittest.TextTestRunner()
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='.')
    # suite = loader.discover(start_dir='tests')
    runner.run(suite)

    # Uncomment this line if this file contains actual tests
    # unittest.main()
