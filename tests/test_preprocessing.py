import unittest

import numpy as np

from shapedtw.preprocessing import *
from shapedtw.shapeDescriptors import *


class TestPadder(unittest.TestCase):

    time_series_test = np.array([1, 2, 3, 4, 5])

    def test_left_padding(self):
        test_padder = Padder(time_series=self.time_series_test, subsequence_width=1)
        expected_val = np.array([1, 1, 2])
        padded_subsequence = test_padder.pad_left(0)

        self.assertTrue(
            np.array_equal(
                expected_val, padded_subsequence
            )
        )

    def test_left_padding_zero_width(self):
        test_padder = Padder(time_series=self.time_series_test, subsequence_width=0)
        expected_val = np.array([1])
        padded_subsequence = test_padder.pad_left(0)

        self.assertTrue(
            np.array_equal(
                expected_val, padded_subsequence
            )
        )

    def test_right_padding(self):
        test_padder = Padder(time_series=self.time_series_test, subsequence_width=1)
        expected_val = np.array([4, 5, 5])
        padded_subsequence = test_padder.pad_right(4)

        self.assertTrue(
            np.array_equal(
                expected_val, padded_subsequence
            )
        )

    def test_right_padding_zero_width(self):
        test_padder = Padder(time_series=self.time_series_test, subsequence_width=0)
        expected_val = np.array([5])
        padded_subsequence = test_padder.pad_right(4)

        self.assertTrue(
            np.array_equal(
                expected_val, padded_subsequence
            )
        )

    def test_both_side_padding(self):
        test_padder = Padder(time_series=self.time_series_test, subsequence_width=5)
        expected_val = np.array(
            [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5]
        )
        padded_subsequence = test_padder.pad_both_side(0)

        self.assertTrue(
            np.array_equal(
                expected_val, padded_subsequence
            )
        )


class TestUnivariateSubsequenceBuilder(unittest.TestCase):

    time_series_test = np.array([1, 2, 3, 4, 5])

    def test_zero_subsequence_width(self):
        subsequence_builder = UnivariateSubsequenceBuilder(self.time_series_test, subsequence_width=0)
        expected_res = np.array([
            [1], [2], [3], [4], [5]
        ])
        subsequences_returned = subsequence_builder.\
            transform_time_series_to_subsequences().subsequences

        self.assertTrue(
            np.array_equal(
                expected_res, subsequences_returned
            )
        )

    def test_without_both_side_padding(self):
        subsequence_builder = UnivariateSubsequenceBuilder(
            self.time_series_test, subsequence_width=2
        )

        expected_res = np.array([
            [1, 1, 1, 2, 3],
            [1, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 5],
            [3, 4, 5, 5, 5]
        ])

        subsequences_returned = subsequence_builder. \
            transform_time_series_to_subsequences().subsequences

        self.assertTrue(
            np.array_equal(
                expected_res, subsequences_returned
            )
        )

    def test_both_side_padding_necessary(self):
        subsequence_builder = UnivariateSubsequenceBuilder(
            self.time_series_test, subsequence_width=5
        )

        expected_res = np.array([
            [1, 1, 1, 1, 1, 1, 2, 3, 4, 5, 5],
            [1, 1, 1, 1, 1, 2, 3, 4, 5, 5, 5],
            [1, 1, 1, 1, 2, 3, 4, 5, 5, 5, 5],
            [1, 1, 1, 2, 3, 4, 5, 5, 5, 5, 5],
            [1, 1, 2, 3, 4, 5, 5, 5, 5, 5, 5]
        ])

        subsequences_returned = subsequence_builder. \
            transform_time_series_to_subsequences().subsequences

        self.assertTrue(
            np.array_equal(
                expected_res, subsequences_returned
            )
        )


class TestUnivariateSeriesSubsequences(unittest.TestCase):

    origin_ts = np.array([1, 2, 3, 4, 5])
    subsequences_array = np.array([
            [1, 1, 1, 2, 3],
            [1, 1, 2, 3, 4],
            [1, 2, 3, 4, 5],
            [2, 3, 4, 5, 5],
            [3, 4, 5, 5, 5]
    ])

    univariate_series_subsequences = UnivariateSeriesSubsequences(
        subsequences_array=subsequences_array, origin_ts=origin_ts
    )

    def test_raw_subsequence_descriptor(self):
        raw_subsequences_desc = RawSubsequenceDescriptor()
        expected_res = self.subsequences_array.copy()
        shape_descriptors_returned = self.univariate_series_subsequences.\
            get_shape_descriptors(shape_descriptor=raw_subsequences_desc). \
            shape_descriptors_array

        self.assertTrue(
            np.array_equal(
                expected_res, shape_descriptors_returned
            )
        )

    def test_slope_descriptor(self):
        slope_descriptor = SlopeDescriptor(slope_window=2)
        expected_res = np.array([
            [0., 1., 0.],
            [0., 1., 0.],
            [1., 1., 0.],
            [1., 1., 0.],
            [1., 0., 0.]
        ])

        shape_descriptors_returned = self.univariate_series_subsequences. \
            get_shape_descriptors(shape_descriptor=slope_descriptor). \
            shape_descriptors_array

        self.assertTrue(
            np.array_equal(
                expected_res, shape_descriptors_returned
            )
        )


class TestMultivariateSubsequenceBuilder(unittest.TestCase):

    time_series_test = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])

    subsequences_builder = MultivariateSubsequenceBuilder(
        time_series=time_series_test,
        subsequence_width=2
    )

    expected_res_dim_1 = np.array([
        [1, 1, 1, 2, 3],
        [1, 1, 2, 3, 4],
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 5],
        [3, 4, 5, 5, 5]
    ])

    expected_res_dim_2 = np.array([
        [10, 10, 10, 20, 30],
        [10, 10, 20, 30, 40],
        [10, 20, 30, 40, 50],
        [20, 30, 40, 50, 50],
        [30, 40, 50, 50, 50]
    ])

    def test_multivariate_subsequence_builder(self):
        time_series_transformed = self.subsequences_builder.\
            transform_time_series_to_subsequences()

        self.assertTrue(
            np.array_equal(
                self.expected_res_dim_1,
                time_series_transformed.subsequences_list[0].subsequences
            ) &
            np.array_equal(
                self.expected_res_dim_2,
                time_series_transformed.subsequences_list[1].subsequences
            )
        )

class TestMultivariateSeriesSubsequences(unittest.TestCase):
    origin_ts = np.array([
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40],
        [5, 50]
    ])

    multivariate_series_subsequences = MultivariateSubsequenceBuilder(
        origin_ts, subsequence_width=2
    ).transform_time_series_to_subsequences()

    def test_raw_descriptor(self):
        raw_shape_descriptor = RawSubsequenceDescriptor()
        shape_descriptors = self.multivariate_series_subsequences.get_shape_descriptors(
            raw_shape_descriptor
        )

        arrays_equal = [
            np.array_equal(
                uni_subsequence.subsequences,
                uni_shape_desc.shape_descriptors_array)
            for (uni_subsequence, uni_shape_desc) in
            zip(
                self.multivariate_series_subsequences.subsequences_list,
                shape_descriptors.descriptors_list
            )
        ]

        self.assertTrue(
            all(arrays_equal)
        )

    def test_slope_descriptor(self):
        slope_descriptor = SlopeDescriptor(slope_window=2)
        expected_res_1 = np.array([
            [0., 1., 0.],
            [0., 1., 0.],
            [1., 1., 0.],
            [1., 1., 0.],
            [1., 0., 0.]
        ])
        expected_res_2 = expected_res_1*10
        expected_res_list = [expected_res_1, expected_res_2]

        shape_descriptors = self.multivariate_series_subsequences.get_shape_descriptors(
            slope_descriptor
        )

        arrays_equal = [
            np.array_equal(
                exp_res,
                uni_shape_desc.shape_descriptors_array
            )
            for (exp_res, uni_shape_desc) in
            zip(expected_res_list, shape_descriptors.descriptors_list)
        ]

        self.assertTrue(
            all(arrays_equal)
        )

class TestUnivariateSeriesShapeDescriptors(unittest.TestCase):

    dim_check_array_1 = np.array([1, 2, 3])
    dim_check_array_2 = np.array(
        [[1, 2,3],
        [4, 5, 6]]
    )
    dim_check_array_3 = np.array(
        [[[1, 2, 3],
         [4, 5, 6]],
        [[7, 8, 9],
         [10, 11, 12]]]
    )

    def test_number_of_dimensions_check(self):
        one_true = UnivariateSeriesShapeDescriptors._check_dimensions_number(
            self.dim_check_array_1, 1
        )
        two_true = UnivariateSeriesShapeDescriptors._check_dimensions_number(
            self.dim_check_array_2, 2
        )
        three_true = UnivariateSeriesShapeDescriptors._check_dimensions_number(
            self.dim_check_array_3, 3
        )
        one_false = not UnivariateSeriesShapeDescriptors._check_dimensions_number(
            self.dim_check_array_1, 2
        )
        two_false = not UnivariateSeriesShapeDescriptors._check_dimensions_number(
            self.dim_check_array_2, 3
        )
        three_false = not UnivariateSeriesShapeDescriptors._check_dimensions_number(
            self.dim_check_array_3, 1
        )

        self.assertTrue(
            all([one_false, two_true, three_true, one_false, two_false, three_false])
        )

if __name__ == '__main__':
    unittest.main()
