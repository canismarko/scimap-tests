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

# flake8: noqa

import math
import datetime as dt
import unittest
import os

from units import unit, predefined
predefined.define_units()

from electrochem.electrode import CathodeLaminate, CoinCellElectrode
from electrochem.galvanostatrun import GalvanostatRun
from electrochem import electrochem_units
from electrochem import biologic
from tests.cases import ScimapTestCase

testdir = os.path.join(os.path.dirname(__file__), 'testdata')
mptfile = os.path.join(testdir, 'NEI-C10_4cycles_C14.mpt')
# eclab-test-data.mpt
mprfile = os.path.join(testdir, 'NEI-C10_4cycles_C14.mpr')

class ElectrochemUnitsTest(ScimapTestCase):
    """Check that various values for current, capacity, etc are compatible."""
    def setUp(self):
        self.mAh = electrochem_units.mAh
        self.hour = electrochem_units.hour
        self.mA = unit('mA')
        self.uA = unit('µA')

    def test_milli_micro_amps(self):
        time = self.mAh(10) / self.uA(1000)
        self.assertApproximatelyEqual(self.hour(time), self.hour(10),
                                      tolerance=10**-10)


class ElectrodeTest(ScimapTestCase):
    def setUp(self):
        self.laminate = CathodeLaminate(mass_active_material=0.9,
                                        mass_carbon=0.05,
                                        mass_binder=0.05,
                                        name="LMO-NEI")
        self.electrode = CoinCellElectrode(total_mass=unit('mg')(15),
                                           substrate_mass=unit('mg')(5),
                                           laminate=self.laminate,
                                           name="DummyElectrode",
                                           diameter=unit('mm')(12.7))

    def test_area(self):
        area_unit = unit('cm') * unit('cm')
        expected_area = area_unit(math.pi * (1.27/2)**2)
        self.assertEqual(self.electrode.area(), expected_area)

    def test_mass_loading(self):
        """Ensure the electrode can calculate the loading in mg/cm^2."""
        loading_units = unit('mg')/(unit('cm')*unit('cm'))
        area = math.pi * (1.27/2)**2
        expected = loading_units((15-5)*0.9 / area)
        self.assertEqual(self.electrode.mass_loading(), expected)


class CycleTest(unittest.TestCase):
    def setUp(self):
        self.run = GalvanostatRun(mptfile, mass=0.022563)
        self.cycle = self.run.cycles[0]

    @unittest.expectedFailure
    def test_discharge_capacity(self):
        self.assertEqual(
            round(self.cycle.discharge_capacity(), 3),
            99.736
        )

class GalvanostatRunTest(ScimapTestCase):
    # Currently just tests import statement
    def test_import(self):
        GalvanostatRun(mptfile)

    def test_read_mass(self):
        run = GalvanostatRun(mptfile)
        self.assertEqual(run.mass, unit('g')(0.02265))

    def test_read_capacity(self):
        run = GalvanostatRun(mptfile)
        self.assertEqual(run.theoretical_capacity, unit('mAh')(3.357))

    def test_read_date(self):
        run = GalvanostatRun(mptfile)
        self.assertEqual(
            run.start_time,
            dt.datetime(2015, 7, 5, 15, 1, 23)
        )

    def test_read_current(self):
        run = GalvanostatRun(mptfile)
        # These are practically equal but assertEqual fails due to rounding in units package
        self.assertApproximatelyEqual(
            run.charge_current,
            unit('mA')(0.33570),
            tolerance=10**-15
        )
        self.assertApproximatelyEqual(
            run.discharge_current,
            unit('mA')(-335.70),
            tolerance=10**-15
        )

    def test_capacity_from_time(self):
        run = GalvanostatRun(mptfile)
        series = run.closest_datum(value=77090, label="time/s")
        self.assertApproximatelyEqual(series.capacity, 196.964615)


# class MPRTestCase(unittest.TestCase):

#     def setUp(self):
#         self.gvrun = GalvanostatRun(mprfile)

#     def test_output(self):
#         print(self.gvrun)
#         # self.assertEqual(
#         #     self.)

if __name__ == '__main__':
    unittest.main()
