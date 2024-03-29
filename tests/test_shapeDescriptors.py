##
## Copyright (c) of Mikołaj Szafraniec
##
## This file is part of the ShapeDTW package.
##
## ShapeDTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ShapeDTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ShapeDTW.  If not, see <http://www.gnu.org/licenses/>.

import unittest
from shapedtw.shapeDescriptors import *


class TestShapeDescriptorAbstract(unittest.TestCase):
    def test_subsequence_shorter_than_window_true(self):
        subsequence_len = 2
        window_size = 4
        self.assertTrue(
            ShapeDescriptor._subsequence_is_shorter_than_window_size(subsequence_len, window_size)
        )

    def test_subsequence_shorter_than_window_false(self):
        subsequence_len = 6
        window_size = 4
        self.assertFalse(
            ShapeDescriptor._subsequence_is_shorter_than_window_size(subsequence_len, window_size)
        )

        subsequence_len_2 = 4
        window_size_2 = 4
        self.assertFalse(
            ShapeDescriptor._subsequence_is_shorter_than_window_size(subsequence_len_2, window_size_2)
        )

    def test_subsequence_shorter_than_window_exception(self):
        subsequence_to_test = np.array([1, 2, 3])
        window_size = 4
        with self.assertRaises(SubsequenceShorterThanWindow):
            ShapeDescriptor._split_into_windows(
                ts_subsequence=subsequence_to_test,
                window_size=window_size
            )

    @staticmethod
    def _compare_lists_of_arryas(list_a: List[ndarray], list_b: List[ndarray]) -> bool:
        return all([all(x == y) for (x, y) in zip(list_a, list_b)])

    def test_equal_length_windows(self):
        subsequence_test = np.array([1, 2, 3, 4])
        window_size = 2
        expected_res = [np.array([1, 2]), np.array([3, 4])]

        test_res = ShapeDescriptor._split_into_windows(
            subsequence_test, window_size
        )
        self.assertTrue(
            self._compare_lists_of_arryas(test_res, expected_res)
        )

    def test_non_equal_length_windows(self):
        subsequence_test = np.array([1, 2, 3])
        window_size = 2
        expected_res = [np.array([1, 2]), np.array([3])]

        test_res = ShapeDescriptor._split_into_windows(
            subsequence_test, window_size
        )
        self.assertTrue(
            self._compare_lists_of_arryas(test_res, expected_res)
        )

    def test_one_unit_subsequence_window(self):
        subsequence_test = np.array([1])
        window_size = 1
        expected_res = [np.array([1])]

        test_res = ShapeDescriptor._split_into_windows(
            subsequence_test, window_size
        )
        self.assertTrue(
            self._compare_lists_of_arryas(test_res, expected_res)
        )

class TestRawSubsequenceDescriptor(unittest.TestCase):

    def test_one_unit_subsequence(self):
        input_subsequence = np.array([1])
        expected_subsequence = np.copy(input_subsequence)
        raw_descriptor = RawSubsequenceDescriptor()
        self.assertTrue(
            all(
                raw_descriptor.get_shape_descriptor(input_subsequence) ==
                expected_subsequence
            )
        )

    def test_multiple_units_subsequence(self):
        input_subsequence = np.array([1, 2, 3, 0.9])
        expected_subsequence = np.copy(input_subsequence)
        raw_descriptor = RawSubsequenceDescriptor()
        self.assertTrue(
            all(
                raw_descriptor.get_shape_descriptor(input_subsequence) ==
                expected_subsequence
            )
        )

class TestPAADescriptor(unittest.TestCase):

    def test_subsequence_even_length(self):
        input_subsequence = np.array([1, 2, 3, 4, 5, 6])
        expected_output = np.array([1.5, 3.5, 5.5])
        paa_descriptor = PAADescriptor(piecewise_aggregation_window=2)
        self.assertTrue(
            all(
                paa_descriptor.get_shape_descriptor(input_subsequence) ==
                expected_output
            )
        )

    def test_subsequence_odd_length(self):
        input_subsequence = np.array([1, 2, 3, 4, 5])
        expected_output = np.array([1.5, 3.5, 5])
        paa_descriptor = PAADescriptor(piecewise_aggregation_window=2)
        self.assertTrue(
            all(
                paa_descriptor.get_shape_descriptor(input_subsequence) ==
                expected_output
            )
        )

class TestDWTDescriptor(unittest.TestCase):

    def test_subsequence_even_length(self):
        input_subsequence = np.array([1, 2, 3, 4])
        expected_output = np.array([7.07106781,  0.0, -2.0, -0.70710678, -0.70710678])
        dwt_desc = DWTDescriptor()
        self.assertTrue(
            np.allclose(
                dwt_desc.get_shape_descriptor(input_subsequence),
                expected_output
            )
        )

    def test_subsequence_odd_length(self):
        input_subsequence = np.array([1, 2, 3, 4, 5])
        expected_output = np.array(
            [10.60660172, -3.53553391, -2.0,  0.0, -0.70710678, -0.70710678,  0.0]
        )
        dwt_desc = DWTDescriptor()
        self.assertTrue(
            np.allclose(
                dwt_desc.get_shape_descriptor(input_subsequence),
                expected_output
            )
        )

class TestSlopeDescriptor(unittest.TestCase):

    def test_subsequence_even_length(self):
        input_subsequence = np.array([1, 1, 1, 2, 1, 0, 0, 10])
        expected_output = np.array(
            [0., 1., -1., 10.]
        )

        slope_desc = SlopeDescriptor(slope_window=2)
        self.assertTrue(
            np.allclose(
                slope_desc.get_shape_descriptor(input_subsequence),
                expected_output
            )
        )

    def test_subsequence_odd_length(self):
        input_subsequence = np.array([1, 1, 1, 2, 1, 0, 5.])
        expected_output = np.array(
            [0., 1., -1., 0.]
        )

        slope_desc = SlopeDescriptor(slope_window=2)
        self.assertTrue(
            np.allclose(
                slope_desc.get_shape_descriptor(input_subsequence),
                expected_output
            )
        )

    def test_slope_window_smaller_than_2_error(self):
        with self.assertRaises(WrongSlopeWindow):
            slope_desc = SlopeDescriptor(slope_window=1)

    def test_slope_window_not_int_error(self):
        with self.assertRaises(WrongSlopeWindow):
            slope_desc = SlopeDescriptor(slope_window=2.5)


class TestDerivativeDescriptor(unittest.TestCase):

    deriv_desc = DerivativeShapeDescriptor()

    def test_subsequence_too_short(self):
        with self.assertRaises(SubsequenceTooShort):
            input_subsequence = np.array([1., 3.5])
            self.deriv_desc.get_shape_descriptor(input_subsequence)

    def test_subsequence_minimum_length(self):
        input_subsequence = np.array([1,2,3])
        expected_output = np.array([1.])
        self.assertTrue(
            np.allclose(
                self.deriv_desc.get_shape_descriptor(
                    input_subsequence
                ),
                expected_output
            )
        )

    def test_subsequence_even_length(self):
        input_subsequence = np.array([1, 0, 5, 3, 6, 2])
        expected_output = np.array([0.5,  3.25, -0.75,  1.25])
        self.assertTrue(
            np.allclose(
                self.deriv_desc.get_shape_descriptor(
                    input_subsequence
                ),
                expected_output
            )
        )

    def test_subsequence_odd_length(self):
        input_subsequence = np.array([1, 0, 5, 3, 6])
        expected_output = np.array([0.5, 3.25, -0.75])
        self.assertTrue(
            np.allclose(
                self.deriv_desc.get_shape_descriptor(
                    input_subsequence
                ),
                expected_output
            )
        )

class TestCompoundDescriptor(unittest.TestCase):

    input_subsequence = np.array([3.6, 9.1, 5.7, 8.9, 4.5, 10.1])
    raw_subsequence_desc = RawSubsequenceDescriptor()
    slope_desc = SlopeDescriptor(slope_window=2)
    derivative_desc = DerivativeShapeDescriptor()

    raw_desc_results = raw_subsequence_desc.get_shape_descriptor(input_subsequence)
    slope_desc_results = slope_desc.get_shape_descriptor(input_subsequence)
    derivative_desc_results = derivative_desc.get_shape_descriptor(input_subsequence)

    def test_compound_desc_no_weigths(self):
        compound_desc = CompoundDescriptor(
            shape_descriptors=[
                self.raw_subsequence_desc, self.slope_desc, self.derivative_desc
            ]
        )

        compound_desc_results = compound_desc.get_shape_descriptor(self.input_subsequence)
        expected_res = np.concatenate(
            [self.raw_desc_results, self.slope_desc_results, self.derivative_desc_results]
        )

        self.assertTrue(
            np.allclose(
                compound_desc_results, expected_res
            )
        )

    def test_compound_desc_weights(self):

        weight_1, weight_2, weight_3 = 1.5, 2.0, 0.5

        compound_desc = CompoundDescriptor(
            shape_descriptors=[
                self.raw_subsequence_desc, self.slope_desc, self.derivative_desc
            ],
            descriptors_weights=[weight_1, weight_2, weight_3]
        )

        compound_desc_results = compound_desc.get_shape_descriptor(self.input_subsequence)
        expected_res = np.concatenate(
            [
                self.raw_desc_results*weight_1,
                self.slope_desc_results*weight_2,
                self.derivative_desc_results*weight_3
            ]
        )

        self.assertTrue(
            np.allclose(
                compound_desc_results, expected_res
            )
        )

if __name__ == '__main__':
    unittest.main()
