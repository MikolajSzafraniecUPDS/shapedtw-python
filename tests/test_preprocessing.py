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

    def test_empty_array_1_dim_error(self):
        empty_array_1 = np.array([])
        empty_origin_ts = np.array([])

        with self.assertRaises(EmptyShapeDescriptorsArray):
            UnivariateSeriesShapeDescriptors(
                empty_array_1,
                empty_origin_ts
            )

    def test_empty_array_2_dim_error(self):
        empty_array_2 = np.array([[], []])
        origin_ts_len_2 = np.array([1, 2])

        with self.assertRaises(EmptyShapeDescriptorsArray):
            UnivariateSeriesShapeDescriptors(
                empty_array_2,
                origin_ts_len_2
            )

    def test_3_dim_array_error(self):
        array_3_dim = np.array([
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]]
        ])
        origin_ts_len_2 = np.array([1, 2])

        with self.assertRaises(TooManyDimensionsArray):
            UnivariateSeriesShapeDescriptors(
                array_3_dim,
                origin_ts_len_2
            )

    def test_1_dim_array_transposition(self):
        array_1_dim = np.array([1.0, 2.3, 4.5])
        origin_ts_len_3 = np.array([1.0, 2.3, 4.5])
        array_1_dim_transponed = np.array([[1.0], [2.3], [4.5]])

        univariate_series_shape_desc = UnivariateSeriesShapeDescriptors(
            array_1_dim,
            origin_ts_len_3
        )

        self.assertTrue(
            np.array_equal(
                univariate_series_shape_desc.shape_descriptors_array,
                array_1_dim_transponed
            )
        )

    def test_array_ts_incompatibility(self):
        array_2_rows = np.array(
            [[1, 2, 3], [4, 5, 6]]
        )
        origin_ts_len_3 = np.array([1.0, 2.3, 4.5])

        with self.assertRaises(UnivariateOriginTSShapeDescriptorsIncompatibility):
            UnivariateSeriesShapeDescriptors(
                array_2_rows,
                origin_ts_len_3
            )

    def test_classes_incompatibility(self):
        array_2_rows = np.array(
            [[1, 2, 3],
             [4, 5, 6]]
        )
        origin_ts_len_2 = np.array(
            [[1, 2],
             [3, 4]]
        )

        ussd = UnivariateSeriesShapeDescriptors(
            array_2_rows,
            origin_ts_len_2
        )
        multivariate_shape_desc = MultivariateSeriesShapeDescriptors(
            [ussd, ussd],
            origin_ts_len_2
        )

        with self.assertRaises(ObjectOfWrongClass):
            ussd.calc_distance_matrix(multivariate_shape_desc)

    def test_distance_results(self):
        origin_ts_len_2 = np.array([1, 2])

        test_dist_array_1 = np.array([[1, 2, 3], [4, 5, 6]])
        test_dist_array_2 = np.array([[1.1, 2, 3.5], [4.4, 5, 6.7]])

        dist_results = np.array([
            [0.50990195, 5.85234996],
            [4.8641546, 0.80622577]
        ])

        ussd_1 = UnivariateSeriesShapeDescriptors(
            test_dist_array_1,
            origin_ts_len_2
        )
        ussd_2 = UnivariateSeriesShapeDescriptors(
            test_dist_array_2,
            origin_ts_len_2
        )
        res = ussd_1.calc_distance_matrix(ussd_2)

        self.assertTrue(
            np.allclose(
                res.dist_matrix,
                dist_results
            )
        )

class MultivariateSeriesShapeDescriptor(unittest.TestCase):

    def test_dimension_incompatibility_exception(self):
        origin_ts_multidim = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        desc_array_1 = np.array(
            [[1, 1, 4],
             [1, 4, 7],
             [4, 7, 7]]
        )
        desc_array_2 = np.array(
            [[2, 2, 5],
             [2, 5, 8],
             [5, 8, 8]]
        )
        origin_ts_univariate_1 = np.array([1, 4, 7])
        origin_ts_univariate_2 = np.array([2, 5, 8])
        usd_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array = desc_array_1,
            origin_ts= origin_ts_univariate_1
        )
        usd_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1,
            origin_ts=origin_ts_univariate_1
        )

        with self.assertRaises(MultivariateOriginTSShapeDescriptorsDimIncompatibility):
            msd = MultivariateSeriesShapeDescriptors(
                descriptors_list=[usd_1, usd_2],
                origin_ts=origin_ts_multidim
            )


if __name__ == '__main__':
    unittest.main()
