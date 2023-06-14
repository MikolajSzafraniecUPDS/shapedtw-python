import unittest

from shapedtw.preprocessing import *
from shapedtw.shapeDescriptors import *



class TestUnivariateSubsequenceBuilder(unittest.TestCase):
    time_series_test = np.array([1, 2, 3, 4, 5])

    def test_zero_subsequence_width(self):
        subsequence_builder = UnivariateSubsequenceBuilder(self.time_series_test, subsequence_width=0)
        expected_res = np.array([
            [1], [2], [3], [4], [5]
        ])
        subsequences_returned = subsequence_builder. \
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
        shape_descriptors_returned = self.univariate_series_subsequences. \
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
        time_series_transformed = self.subsequences_builder. \
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
        expected_res_2 = expected_res_1 * 10
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


class TestMultivariateSeriesShapeDescriptor(unittest.TestCase):

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
            descriptors_array=desc_array_1,
            origin_ts=origin_ts_univariate_1
        )
        usd_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2,
            origin_ts=origin_ts_univariate_2
        )

        with self.assertRaises(MultivariateOriginTSShapeDescriptorsDimIncompatibility):
            msd = MultivariateSeriesShapeDescriptors(
                descriptors_list=[usd_1, usd_2],
                origin_ts=origin_ts_multidim
            )

    def test_length_incompatibility_exception(self):
        origin_ts_multidim = np.array(
            [[1, 2],
             [4, 5],
             [7, 8]]
        )
        desc_array_1 = np.array(
            [[1, 1, 4],
             [1, 4, 7],
             [4, 7, 7],
             [7, 7, 7]]
        )
        desc_array_2 = np.array(
            [[2, 2, 5],
             [2, 5, 8],
             [5, 8, 8]]
        )
        origin_ts_univariate_1 = np.array([1, 4, 7, 7])
        origin_ts_univariate_2 = np.array([2, 5, 8])
        usd_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1,
            origin_ts=origin_ts_univariate_1
        )
        usd_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2,
            origin_ts=origin_ts_univariate_2
        )

        with self.assertRaises(MultivariateOriginTSShapeDescriptorsLengthIncompatibility):
            msd = MultivariateSeriesShapeDescriptors(
                descriptors_list=[usd_1, usd_2],
                origin_ts=origin_ts_multidim
            )

    def test_base_exceptions(self):
        origin_ts_multidim_1 = np.array(
            [[1, 2],
             [4, 5],
             [7, 8]]
        )
        desc_array_1_1 = np.array(
            [[1, 1, 4],
             [1, 4, 7],
             [4, 7, 7]]
        )
        desc_array_1_2 = np.array(
            [[2, 2, 5],
             [2, 5, 8],
             [5, 8, 8]]
        )

        origin_ts_multidim_2 = np.array(
            [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        )
        desc_array_2_1 = np.array(
            [[1, 1, 4],
             [1, 4, 7],
             [4, 7, 7]]
        )
        desc_array_2_2 = np.array(
            [[2, 2, 5],
             [2, 5, 8],
             [5, 8, 8]]
        )
        desc_array_2_3 = np.array(
            [[3, 3, 6],
             [3, 6, 9],
             [6, 9, 9]]
        )

        origin_ts_univariate_1 = np.array([1, 4, 7])
        origin_ts_univariate_2 = np.array([2, 5, 8])
        origin_ts_univariate_3 = np.array([3, 6, 9])

        usd_1_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1_1,
            origin_ts=origin_ts_univariate_1
        )
        usd_1_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1_2,
            origin_ts=origin_ts_univariate_2
        )

        usd_2_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_1,
            origin_ts=origin_ts_univariate_1
        )
        usd_2_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_2,
            origin_ts=origin_ts_univariate_2
        )
        usd_2_3 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_3,
            origin_ts=origin_ts_univariate_3
        )

        msd_1 = MultivariateSeriesShapeDescriptors(
            descriptors_list=[usd_1_1, usd_1_2],
            origin_ts=origin_ts_multidim_1
        )
        msd_2 = MultivariateSeriesShapeDescriptors(
            descriptors_list=[usd_2_1, usd_2_2, usd_2_3],
            origin_ts=origin_ts_multidim_2
        )

        with self.assertRaises(ObjectOfWrongClass):
            msd_1.calc_distance_matrices(
                usd_1_1
            )

        with self.assertRaises(MultivariateSeriesShapeDescriptorsIncompatibility):
            msd_1.calc_distance_matrices(msd_2)

    def test_distance_results_independent_type(self):
        origin_ts_multidim_1 = np.array(
            [[1., 2.],
             [4., 5.],
             [7., 8.]]
        )
        origin_ts_multidim_2 = np.array(
            [[2.5, 4.5],
             [4.0, 7.0],
             [5.0, 4.5]]
        )

        origin_ts_univariate_1_1 = np.array([1., 4., 7.])
        origin_ts_univariate_1_2 = np.array([2., 5., 8.])

        origin_ts_univariate_2_1 = np.array([2.5, 4.0, 5.0])
        origin_ts_univariate_2_2 = np.array([4.5, 7.0, 4.5])

        desc_array_1_1 = np.array(
            [[1., 1., 4.],
             [1., 4., 7.],
             [4., 7., 7.]]
        )
        desc_array_1_2 = np.array(
            [[2., 2., 5.],
             [2., 5., 8.],
             [5., 8., 8.]]
        )
        desc_array_2_1 = np.array(
            [[2.5, 2.5, 4.0],
             [2.5, 4.0, 5.0],
             [4.0, 5.0, 5.0]]
        )
        desc_array_2_2 = np.array(
            [[4.5, 4.5, 7.0],
             [4.5, 7.0, 4.5],
             [4.5, 7.0, 7.0]]
        )

        usd_1_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1_1,
            origin_ts=origin_ts_univariate_1_1
        )
        usd_1_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1_2,
            origin_ts=origin_ts_univariate_1_2
        )
        usd_2_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_1,
            origin_ts=origin_ts_univariate_2_1
        )
        usd_2_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_2,
            origin_ts=origin_ts_univariate_2_2
        )

        msd_1 = MultivariateSeriesShapeDescriptors(
            descriptors_list=[usd_1_1, usd_1_2],
            origin_ts=origin_ts_multidim_1
        )
        msd_2 = MultivariateSeriesShapeDescriptors(
            descriptors_list=[usd_2_1, usd_2_2],
            origin_ts=origin_ts_multidim_2
        )

        expected_dist_1 = np.array(
            [[2.12132034, 3.5, 5.09901951],
             [3.67423461, 2.5, 3.74165739],
             [5.61248608, 3.90512484, 2.82842712]]
        )
        expected_dist_2 = np.array(
            [[4.0620192, 5.61248608, 5.93717104],
             [2.73861279, 4.74341649, 3.35410197],
             [3.67423461, 3.67423461, 1.5]]
        )

        results = msd_1.calc_distance_matrices(msd_2, dist_method="euclidean")

        self.assertIsInstance(
            results, MultivariateDistanceMatrixIndependent
        )

        self.assertTrue(
            np.array_equal(
                results.ts_x,
                origin_ts_multidim_1
            )
        )

        self.assertTrue(
            np.array_equal(
                results.ts_y,
                origin_ts_multidim_2
            )
        )

        self.assertTrue(
            len(results.distance_matrices_list) == 2
        )

        self.assertTrue(
            np.allclose(
                results.distance_matrices_list[0].dist_matrix,
                expected_dist_1
            )
        )

        self.assertTrue(
            np.allclose(
                results.distance_matrices_list[1].dist_matrix,
                expected_dist_2
            )
        )

    def test_distance_results_dependent_type(self):
        origin_ts_multidim_1 = np.array(
            [[1., 2.],
             [4., 5.],
             [7., 8.]]
        )
        origin_ts_multidim_2 = np.array(
            [[2.5, 4.5],
             [4.0, 7.0],
             [5.0, 4.5]]
        )

        origin_ts_univariate_1_1 = np.array([1., 4., 7.])
        origin_ts_univariate_1_2 = np.array([2., 5., 8.])

        origin_ts_univariate_2_1 = np.array([2.5, 4.0, 5.0])
        origin_ts_univariate_2_2 = np.array([4.5, 7.0, 4.5])

        desc_array_1_1 = np.array(
            [[1., 1., 4.],
             [1., 4., 7.],
             [4., 7., 7.]]
        )
        desc_array_1_2 = np.array(
            [[2., 2., 5.],
             [2., 5., 8.],
             [5., 8., 8.]]
        )
        desc_array_2_1 = np.array(
            [[2.5, 2.5, 4.0],
             [2.5, 4.0, 5.0],
             [4.0, 5.0, 5.0]]
        )
        desc_array_2_2 = np.array(
            [[4.5, 4.5, 7.0],
             [4.5, 7.0, 4.5],
             [4.5, 7.0, 7.0]]
        )

        usd_1_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1_1,
            origin_ts=origin_ts_univariate_1_1
        )
        usd_1_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_1_2,
            origin_ts=origin_ts_univariate_1_2
        )
        usd_2_1 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_1,
            origin_ts=origin_ts_univariate_2_1
        )
        usd_2_2 = UnivariateSeriesShapeDescriptors(
            descriptors_array=desc_array_2_2,
            origin_ts=origin_ts_univariate_2_2
        )

        msd_1 = MultivariateSeriesShapeDescriptors(
            descriptors_list=[usd_1_1, usd_1_2],
            origin_ts=origin_ts_multidim_1
        )
        msd_2 = MultivariateSeriesShapeDescriptors(
            descriptors_list=[usd_2_1, usd_2_2],
            origin_ts=origin_ts_multidim_2
        )

        expected_euclid_dist = np.array(
            [[4.58257569, 6.61437828, 7.82623792],
             [4.58257569, 5.36190265, 5.02493781],
             [6.70820393, 5.36190265, 3.20156212]]
        )
        expected_cityblock_dist = np.array(
            [[10., 13.5, 17.5],
             [10., 11.5, 11.5],
             [14., 11.5, 6.5]]
        )

        results_euclid = msd_1.calc_summed_distance_matrix(msd_2, dist_method="euclidean")
        results_cityblock = msd_1.calc_summed_distance_matrix(msd_2, dist_method="cityblock")

        self.assertIsInstance(
            results_euclid, MultivariateDistanceMatrixDependent
        )

        self.assertIsInstance(
            results_cityblock, MultivariateDistanceMatrixDependent
        )

        self.assertTrue(
            np.array_equal(
                results_euclid.ts_x,
                origin_ts_multidim_1
            )
        )

        self.assertTrue(
            np.array_equal(
                results_euclid.ts_y,
                origin_ts_multidim_2
            )
        )

        self.assertTrue(
            np.array_equal(
                results_cityblock.ts_x,
                origin_ts_multidim_1
            )
        )

        self.assertTrue(
            np.array_equal(
                results_cityblock.ts_y,
                origin_ts_multidim_2
            )
        )

        self.assertTrue(
            np.allclose(
                results_euclid.distance_matrix,
                expected_euclid_dist
            )
        )

        self.assertTrue(
            np.allclose(
                results_cityblock.distance_matrix,
                expected_cityblock_dist
            )
        )


class TestDistanceMatrixCalculator(unittest.TestCase):

    def test_empty_arrays_exception(self):
        ts_1 = np.array([])
        ts_2 = np.array([])
        ts_3 = np.array([1., 2.4])

        with self.assertRaises(DimensionError):
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()

        with self.assertRaises(DimensionError):
            DistanceMatrixCalculator(
                ts_1, ts_3, "euclidean"
            ).calc_distance_matrix()

    def test_3d_arrays_exception(self):
        ts_1 = np.array([
            [[1., 2.],
             [3., 4.]],
            [[5., 6.],
             [7., 8.]]]
        )
        ts_2 = np.array([
            [[2., 3.],
             [3.5, 4.1]],
            [[5.3, 6.6],
             [7.4, 8.1]]]
        )

        with self.assertRaises(DimensionError):
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()

        try:
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()
        except DimensionError as e:
            self.assertTrue(
                str(e) == "Only arrays of 1 and 2 dimensions are supported"
            )

    def test_different_dim_number_exception(self):
        ts_1 = np.array([1., 2., 3.])
        ts_2 = np.array([[1., 2., 3.], [4., 5., 6.]])

        with self.assertRaises(DimensionError):
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()

        try:
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()
        except DimensionError as e:
            self.assertTrue(
                str(e) == "Shapes of time series are different"
            )

    def test_number_of_cols_doesnt_match(self):
        ts_1 = np.array(
            [
                [1., 2., 3.],
                [4., 5., 6.]
            ]
        )
        ts_2 = np.array(
            [
                [1.1, 2.6],
                [4.3, 5.5]
            ]
        )

        with self.assertRaises(DimensionError):
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()

        try:
            DistanceMatrixCalculator(
                ts_1, ts_2, "euclidean"
            ).calc_distance_matrix()
        except DimensionError as e:
            self.assertTrue(
                str(e) == "Number of time series columns doesn't match"
            )

    def test_univariate_series_transposition(self):
        ts_1 = np.array([1., 2.])
        ts_2 = np.array([2., 3.])

        res = DistanceMatrixCalculator(
            ts_1, ts_2, "euclidean"
        ).calc_distance_matrix()

        self.assertTrue(
            res.shape == (2, 2)
        )

    def test_1d_series_dist_matrix_results(self):
        ts_1 = np.array([1., 2.])
        ts_2 = np.array([2., 3.])

        expected_res = np.array(
            [[1., 2.],
             [0., 1.]]
        )

        res = DistanceMatrixCalculator(
            ts_1, ts_2
        ).calc_distance_matrix()

        self.assertTrue(
            np.array_equal(
                expected_res,
                res
            )
        )

    def test_2d_series_dist_matrix_results(self):
        ts_1 = np.array(
            [[1., 2.5, 4.5, 7.6],
             [2.5, 6., 3.4, 8.1]]
        )
        ts_2= np.array(
            [[2., 4.5, 4.5, 2.1],
             [6.5, 5., 6.7, 4.3]]
        )

        expected_res = np.array(
            [[5.93717104, 7.22703259],
             [6.3015871, 6.50615094]]
        )

        res = DistanceMatrixCalculator(
            ts_1, ts_2
        ).calc_distance_matrix()

        self.assertTrue(
            np.allclose(
                expected_res,
                res
            )
        )

if __name__ == '__main__':
    unittest.main()
