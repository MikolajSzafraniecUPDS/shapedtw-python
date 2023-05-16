import unittest
import math

from shapedtw.shapedtw import *
from shapedtw.exceptions import *

class TestStepPatternMatrixTransformator(unittest.TestCase):

    def test_get_step_pattern_matrix(self):
        expected_array = np.array(
            [[1., 1., 0., -1.],
             [1., 0., 0., 1.],
             [2., 1., 1., -1.],
             [2., 0., 0., 1.],
             [3., 1., 2., -1.],
             [3., 0., 0., 1.]]
        )

        res_array = StepPatternMatrixTransformator(
            "asymmetric"
        )._get_step_pattern_matrix()

        self.assertTrue(
            np.array_equal(
                expected_array, res_array
            )
        )

    def test_step_pattern_doesnt_exists(self):
        with self.assertRaises(ProvidedStepPatternDoesNotExists):
            spmt = StepPatternMatrixTransformator(
                "asymetric"
            )

    def test_get_segments_number(self):
        self.assertEqual(
            StepPatternMatrixTransformator("symmetric1")._get_segments_number(),
            3
        )

        self.assertEqual(
            StepPatternMatrixTransformator("symmetricP05")._get_segments_number(),
            5
        )

    def test_get_matrix_segment(self):
        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetricP05")._get_matrix_segment(6)

        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetric1")._get_matrix_segment(0)

        expected_res = np.array(
            [[ 1.,  0.,  1., -1.],
             [ 1.,  0.,  0.,  0.]]
        )

        self.assertTrue(
            np.array_equal(
                expected_res,
                StepPatternMatrixTransformator("asymmetricP0")._get_matrix_segment(1)
            )
        )

    def test_get_segment_length(self):
        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetricP05")._get_segment_length(6)

        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetric1")._get_segment_length(0)

        self.assertEqual(
            StepPatternMatrixTransformator("symmetricP05")._get_segment_length(5),
            3
        )

        self.assertEqual(
            StepPatternMatrixTransformator("symmetricP0")._get_segment_length(2),
            1
        )

    def test_get_segment_pattern(self):
        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetricP05")._get_matrix_segment(6)

        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetric1")._get_matrix_segment(0)

        self.assertEqual(
            (1, 3),
            StepPatternMatrixTransformator("asymmetricP05")._get_segment_pattern(1)
        )

        self.assertEqual(
            (1, 0),
            StepPatternMatrixTransformator("symmetricP0")._get_segment_pattern(3)
        )

    def test_segment_to_dict(self):
        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetricP05")._get_matrix_segment(6)

        with self.assertRaises(SegmentIndexOutOfRange):
            StepPatternMatrixTransformator("symmetric1")._get_matrix_segment(0)

        expected_res_symmetricP0_1 = {0: {'x_index': 0, 'y_index': 0, 'weight': 2.0}}
        expected_res_symmetricP05_2 = {
            0: {'x_index': 0, 'y_index': 1, 'weight': 2.0},
            1: {'x_index': 0, 'y_index': 0, 'weight': 1.0}
        }

        self.assertEqual(
            StepPatternMatrixTransformator("symmetricP0")._segment_to_dict(1),
            expected_res_symmetricP0_1
        )

        self.assertEqual(
            StepPatternMatrixTransformator("symmetricP05")._segment_to_dict(2),
            expected_res_symmetricP05_2
        )

    def test_step_pattern_matrix_to_dict(self):
        expected_res_symmetric1 = {
            (1, 1): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}},
            (0, 1): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}},
            (1, 0): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}}
        }

        expected_res_typeIIc = {
            (1, 1): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}},
            (1, 2): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}},
            (2, 1): {0: {'x_index': 0, 'y_index': 0, 'weight': 2.0}}
        }

        expected_res_asymmetricP05 = {
            (1, 3): {
                0: {'x_index': 0, 'y_index': 2, 'weight': 0.3333333333333333},
                1: {'x_index': 0, 'y_index': 1, 'weight': 0.3333333333333333},
                2: {'x_index': 0, 'y_index': 0, 'weight': 0.3333333333333333}
            },
            (1, 2): {
                0: {'x_index': 0, 'y_index': 1, 'weight': 0.5},
                1: {'x_index': 0, 'y_index': 0, 'weight': 0.5}
            },
            (1, 1): {
                0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}
            },
            (2, 1): {
                0: {'x_index': 1, 'y_index': 0, 'weight': 1.0},
                1: {'x_index': 0, 'y_index': 0, 'weight': 1.0}
            },
            (3, 1): {
                0: {'x_index': 2, 'y_index': 0, 'weight': 1.0},
                1: {'x_index': 1, 'y_index': 0, 'weight': 1.0},
                2: {'x_index': 0, 'y_index': 0, 'weight': 1.0}
            }
        }

        self.assertEqual(
            expected_res_symmetric1,
            StepPatternMatrixTransformator("symmetric1").step_pattern_matrix_to_dict()
        )

        self.assertEqual(
            expected_res_typeIIc,
            StepPatternMatrixTransformator("typeIIc").step_pattern_matrix_to_dict()
        )

        self.assertEqual(
            expected_res_asymmetricP05,
            StepPatternMatrixTransformator("asymmetricP05").step_pattern_matrix_to_dict()
        )


class TestDistanceReconstructor(unittest.TestCase):

    ts_x = np.array([1.2, 4.5, 1.3, 2.4, 2.6])
    ts_y = np.array([2.4, 3.4, 5.6, 1.3, 5.7])

    distance_matrix = np.array(
            [[1.2, 2.2, 4.4, 0.1, 4.5],
             [2.1, 1.1, 1.1, 3.2, 1.2],
             [1.1, 2.1, 4.3, 0., 4.4],
             [0., 1., 3.2, 1.1, 3.3],
             [0.2, 0.8, 3., 1.3, 3.1]]
        )

    ts_x_warping_path_symmetric2 = dtw(x=ts_x, y=ts_y, dist_method="euclidean", step_pattern="symmetric2").index1s
    ts_y_warping_path_symmetric2 = dtw(x=ts_x, y=ts_y, dist_method="euclidean", step_pattern="symmetric2").index2s

    ts_x_warping_path_symmetricP05 = dtw(x=ts_x, y=ts_y, dist_method="euclidean", step_pattern="symmetricP05").index1s
    ts_y_warping_path_symmetricP05 = dtw(x=ts_x, y=ts_y, dist_method="euclidean", step_pattern="symmetricP05").index2s

    def test_calc_distance_matrix(self):
        res = DistanceReconstructor(
            "symmetric2", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetric2,
            self.ts_y_warping_path_symmetric2
        )._calc_distance_matrix()

        self.assertTrue(
            np.allclose(
                self.distance_matrix,
                res
            )
        )

    def test_calc_single_distance_simple(self):
        distance_reconstructor = DistanceReconstructor(
            "symmetric2", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetric2, self.ts_y_warping_path_symmetric2
        )

        x_ind = 1
        y_ind = 1
        pattern = (1, 1)
        pattern_dict = distance_reconstructor.step_pattern_dictionary

        expected_res = distance_reconstructor.distance_matrix[1, 1]*2

        res = distance_reconstructor._calc_single_distance(
            x_index=x_ind, y_index=y_ind, single_pattern_dict=pattern_dict[pattern][0]
        )

        self.assertEqual(expected_res, res)

    def test_calc_single_distance_compound(self):
        distance_reconstructor = DistanceReconstructor(
            "symmetricP05", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetricP05,
            self.ts_y_warping_path_symmetricP05
        )

        x_ind = 3
        y_ind = 3
        pattern = (2,1)
        pattern_dict = distance_reconstructor.step_pattern_dictionary

        expected_res = distance_reconstructor.distance_matrix[2, 3]*2.0
        res = distance_reconstructor._calc_single_distance(
            x_index=x_ind, y_index=y_ind, single_pattern_dict=pattern_dict[pattern][0]
        )

        self.assertEqual(expected_res, res)

    def test_calc_distance_for_pattern_simple(self):
        distance_reconstructor = DistanceReconstructor(
            "symmetric2", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetric2, self.ts_y_warping_path_symmetric2
        )

        x_ind = 1
        y_ind = 1
        pattern = (1, 1)

        expected_res = distance_reconstructor.distance_matrix[1, 1] * 2.0
        res = distance_reconstructor._calc_distance_for_given_pattern(
            x_index=x_ind, y_index=y_ind, pattern=pattern
        )

        self.assertEqual(expected_res, res)

    def test_calc_distance_for_pattern_compound(self):
        distance_reconstructor = DistanceReconstructor(
            "symmetricP05", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetricP05,
            self.ts_y_warping_path_symmetricP05
        )

        x_ind = 3
        y_ind = 3
        pattern = (2, 1)

        expected_res = (
                distance_reconstructor.distance_matrix[2, 3] * 2.0
        ) + (
                distance_reconstructor.distance_matrix[3, 3] * 1.0
        )
        res = distance_reconstructor._calc_distance_for_given_pattern(
            x_index=x_ind, y_index=y_ind,
            pattern=pattern
        )

        self.assertEqual(expected_res, res)

    def test_get_indices_pairs(self):
        distance_reconstructor = DistanceReconstructor(
            "symmetric2", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetric2,
            self.ts_y_warping_path_symmetric2
        )

        # Warping paths:
        # ts_x = [0, 1, 1, 2, 3, 4, 4]
        # ts_y = [0, 1, 2, 3, 3, 3, 4]
        expected_res = [
            (0, 0), (1, 1), (1, 2), (2, 3), (3, 3), (4, 3), (4, 4)
        ]
        res = distance_reconstructor._get_indices_pairs()
        self.assertEqual(expected_res, res)

    def test_get_indices_patterns(self):
        distance_reconstructor = DistanceReconstructor(
            "symmetric2", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetric2,
            self.ts_y_warping_path_symmetric2
        )

        # Warping paths:
        # ts_x = [0, 1, 1, 2, 3, 4, 4]
        # ts_y = [0, 1, 2, 3, 3, 3, 4]
        expected_res = [
            (1, 1), (0, 1), (1, 1), (1, 0), (1, 0), (0, 1)
        ]
        res = distance_reconstructor._get_indices_patterns()
        self.assertEqual(expected_res, res)

    def test_calc_raw_ts_distance_simple_pattern(self):
        # Warping path = [(0, 0), (1, 1), (1, 2), (2, 3), (3, 3), (4, 3), (4, 4)]
        # Warping steps = [(1, 1), (0, 1), (1, 1), (1, 0), (1, 0), (0, 1)]
        expected_res = self.distance_matrix[0,0] + \
            self.distance_matrix[1,1]*2.0 +\
            self.distance_matrix[1,2] + \
            self.distance_matrix[2,3]*2.0 + \
            self.distance_matrix[3,3] + \
            self.distance_matrix[4,3] + \
            self.distance_matrix[4,4]

        res = DistanceReconstructor(
            "symmetric2", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetric2,
            self.ts_y_warping_path_symmetric2
        ).calc_raw_ts_distance()

        self.assertAlmostEqual(expected_res, res)

    def test_calc_raw_ts_distance_simple_compound(self):
        # Warping path = [(0, 0), (1, 2), (3, 3), (4, 4)]
        # Warping steps = [(1, 2), (2, 1), (1, 1)]
        expected_res = self.distance_matrix[0,0] + \
                       (self.distance_matrix[1, 1]*2.0 + self.distance_matrix[1, 2]) + \
                       (self.distance_matrix[2,3]*2.0 + self.distance_matrix[3,3]) + \
                       self.distance_matrix[4,4]*2.0

        res = DistanceReconstructor(
            "symmetricP05", self.ts_x, self.ts_y,
            self.ts_x_warping_path_symmetricP05,
            self.ts_y_warping_path_symmetricP05
        ).calc_raw_ts_distance()

        self.assertAlmostEqual(expected_res, res)

class TestShapeDTW(unittest.TestCase):

    ts_x = np.array([1., 2.5, 1.2, 4.5, 3.4])
    ts_y = np.array([4., 2.1, 2.6, 3.1, 4.5])
    dist_matrix = cdist(
        np.atleast_2d(ts_x).T, np.atleast_2d(ts_y).T
    )
    distance = 7.5
    normalized_distance = 0.75

    def test_calc_raw_series_distance(self):
        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )

        # ts_x warping path = [0, 1, 1, 1, 2, 3, 4]
        # ts_y warping path = [0, 1, 2, 3, 3, 4, 4]
        expected_res = self.dist_matrix[0, 0] + \
            self.dist_matrix[1, 1] * 2.0 + \
            self.dist_matrix[1, 2] + \
            self.dist_matrix[1, 3] + \
            self.dist_matrix[2, 3] + \
            self.dist_matrix[3, 4] * 2.0 + \
            self.dist_matrix[4, 4]

        res = shape_dtw_res._calc_raw_series_distance("euclidean")

        self.assertAlmostEqual(expected_res, res)

    def test_calc_raw_series_normalized_distance(self):

        shape_dtw_res_symmetric2 = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y), step_pattern="symmetric2"
        )
        shape_dtw_res_asymmetricP05 = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y), step_pattern="asymmetricP05"
        )
        shape_dtw_res_mori2006 = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y), step_pattern="mori2006"
        )
        shape_dtw_res_symmetric1 = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y), step_pattern="symmetric1"
        )

        self.assertEqual(
            shape_dtw_res_symmetric2._calc_raw_series_normalized_distance(self.distance),
            self.distance / (5+5)
        )
        self.assertEqual(
            shape_dtw_res_asymmetricP05._calc_raw_series_normalized_distance(self.distance),
            self.distance / (5)
        )
        self.assertEqual(
            shape_dtw_res_mori2006._calc_raw_series_normalized_distance(self.distance),
            self.distance / (5)
        )
        self.assertTrue(
            math.isnan(
                shape_dtw_res_symmetric1._calc_raw_series_normalized_distance(self.distance)
            )
        )

    def test_calc_distances(self):

        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )
        distances = shape_dtw_res._calc_distances()

        self.assertIsInstance(
            distances, ShapeDTWResults
        )

        self.assertEqual(
            distances.distance,
            self.distance
        )

        self.assertEqual(
            distances.normalized_distance,
            self.normalized_distance
        )

        self.assertEqual(
            distances.distance,
            distances.shape_distance
        )

        self.assertEqual(
            distances.normalized_distance,
            distances.shape_normalized_distance
        )

    def test_get_distance(self):
        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )

        with self.assertRaises(ShapeDTWNotCalculatedYet):
            return shape_dtw_res.distance

    def test_get_normalized_distance(self):
        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )

        with self.assertRaises(ShapeDTWNotCalculatedYet):
            return shape_dtw_res.distance

    def test_get_shape_descriptor_distance(self):
        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )

        with self.assertRaises(ShapeDTWNotCalculatedYet):
            return shape_dtw_res.shape_distance

    def test_get_shape_descriptor_normalized_distance(self):
        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )

        with self.assertRaises(ShapeDTWNotCalculatedYet):
            return shape_dtw_res.shape_normalized_distance

    def test_set_distance(self):
        shape_dtw_res = ShapeDTW(
            self.ts_x, self.ts_y, dtw_res=dtw(self.ts_x, self.ts_y)
        )

        with self.assertRaises(DistanceSettingNotPossible):
            shape_dtw_res.distance = 10.1

        with self.assertRaises(DistanceSettingNotPossible):
            shape_dtw_res.normalized_distance = 10.1

        with self.assertRaises(DistanceSettingNotPossible):
            shape_dtw_res.shape_distance = 10.1

        with self.assertRaises(DistanceSettingNotPossible):
            shape_dtw_res.shape_normalized_distance = 10.1

class TestUnivariateShapeDTW(unittest.TestCase):

    ts_x = np.array([5.4, 3.4, 9.0, 1.2, 4.5, 6.7, 12.4])
    ts_y = np.array([10.6, 3.4, 2.1, 7.8, 2.3, 13.4, 11.3])

    def test_raw_subseries_descriptor_zero_width(self):
        """
        For raw subseries descriptor with zero width results
        of shape dtw are expected to be the same as for 'standard'
        dynamic time warping
        """
        raw_shape_descriptor = RawSubsequenceDescriptor()
        shape_dtw_res = UnivariateShapeDTW(
            self.ts_x, self.ts_y
        ).calc_shape_dtw(0, raw_shape_descriptor)

        self.assertAlmostEqual(
            shape_dtw_res.distance,
            shape_dtw_res.shape_distance
        )

        self.assertAlmostEqual(
            shape_dtw_res.normalized_distance,
            shape_dtw_res.shape_normalized_distance
        )

    def test_calc_shape_dtw(self):
        expected_distance = 42.5
        expected_normalized_distance = expected_distance / (len(self.ts_x) + len(self.ts_y))
        expected_shape_distance = 90.82777378665594
        expected_shape_normalized_distance = expected_shape_distance / (
                len(self.ts_x) + len(self.ts_y)
        )

        slope_shape_desc = SlopeDescriptor(2)

        shape_dtw_res = UnivariateShapeDTW(
            self.ts_x, self.ts_y
        ).calc_shape_dtw(3, slope_shape_desc)

        self.assertAlmostEqual(
            expected_distance,
            shape_dtw_res.distance
        )

        self.assertAlmostEqual(
            expected_normalized_distance,
            shape_dtw_res.normalized_distance
        )

        self.assertAlmostEqual(
            expected_shape_distance,
            shape_dtw_res.shape_distance
        )

        self.assertAlmostEqual(
            expected_shape_normalized_distance,
            shape_dtw_res.shape_normalized_distance
        )


class TestMultivariateShapeDTWDependent(unittest.TestCase):
    ts_x = np.array([
        [9., 8.1],
        [4.5, 7.8],
        [1.2, 3.4],
        [5.6, 1.1],
        [7.8, 1.2]
    ])
    ts_y = np.array([
        [4.5, 3.4],
        [1.9, 6.3],
        [3.5, 1.4],
        [9.5, 9. ],
        [1.1, 9.5]
    ])

    def test_raw_subseries_descriptor_zero_width(self):
        """
        For raw subseries descriptor with zero width results
        of shape dtw are expected to be the same as for 'standard'
        dynamic time warping
        """
        raw_shape_descriptor = RawSubsequenceDescriptor()
        shape_dtw_res = MultivariateShapeDTWDependent(
            self.ts_x, self.ts_y
        ).calc_shape_dtw(0, raw_shape_descriptor)

        self.assertAlmostEqual(
            shape_dtw_res.distance,
            shape_dtw_res.shape_distance
        )

        self.assertAlmostEqual(
            shape_dtw_res.normalized_distance,
            shape_dtw_res.shape_normalized_distance
        )

    def test_calc_shape_dtw(self):
        expected_distance = 51.475780044481546
        expected_normalized_distance = expected_distance / (len(self.ts_x) + len(self.ts_y))
        expected_shape_distance = 81.14705796798373
        expected_shape_normalized_distance = expected_shape_distance / (
                len(self.ts_x) + len(self.ts_y)
        )

        slope_shape_desc = SlopeDescriptor(2)

        shape_dtw_res = MultivariateShapeDTWDependent(
            self.ts_x, self.ts_y
        ).calc_shape_dtw(3, slope_shape_desc)

        self.assertAlmostEqual(
            expected_distance,
            shape_dtw_res.distance
        )

        self.assertAlmostEqual(
            expected_normalized_distance,
            shape_dtw_res.normalized_distance
        )

        self.assertAlmostEqual(
            expected_shape_distance,
            shape_dtw_res.shape_distance
        )

        self.assertAlmostEqual(
            expected_shape_normalized_distance,
            shape_dtw_res.shape_normalized_distance
        )

class TestMultivariateShapeDTWIndependent(unittest.TestCase):

    ts_x = np.array([
        [3.4, 10.5],
        [4.5, 6.1],
        [1.4, 8.6],
        [6.7, 13.5]
    ])
    ts_y = np.array([
        [5.6, 13.4],
        [1.3, 8.7],
        [8.6, 11.1],
        [1.4, 9.8]
    ])

    shape_descriptor = CompoundDescriptor(
        [SlopeDescriptor(2), PAADescriptor(2)]
    )

    def test_calc_raw_series_distance(self):
        raw_series_dist_matrix_1 = cdist(
            np.atleast_2d(self.ts_x[:,0]).T,
            np.atleast_2d(self.ts_y[:,0]).T
        )
        raw_series_dist_matrix_2 = cdist(
            np.atleast_2d(self.ts_x[:, 1]).T,
            np.atleast_2d(self.ts_y[:, 1]).T
        )

        shape_dtw_res_independent = MultivariateShapeDTWIndependent(
            self.ts_x, self.ts_y
        ).calc_shape_dtw(2, self.shape_descriptor)

        # Warping path 1: [(0, 0), (1, 0), (2, 1), (3, 2), (3, 3)]
        # Warping path 2: [(0, 0), (1, 1), (1, 2), (2, 2), (3, 3)]
        dist_1 = raw_series_dist_matrix_1[0, 0] + \
            raw_series_dist_matrix_1[1, 0] + \
            raw_series_dist_matrix_1[2, 1]*2.0 + \
            raw_series_dist_matrix_1[3, 2]*2.0 + \
            raw_series_dist_matrix_1[3, 3]

        dist_2 = raw_series_dist_matrix_2[0, 0] + \
            raw_series_dist_matrix_2[1, 1]*2.0 + \
            raw_series_dist_matrix_2[1, 2] + \
            raw_series_dist_matrix_2[2, 2] + \
            raw_series_dist_matrix_2[3, 3]*2.0

        expected_dist = dist_1+dist_2
        self.assertAlmostEqual(
            expected_dist,
            shape_dtw_res_independent.distance
        )

    def test_calc_distances(self):
        expected_distance = 35.599999999999994
        expected_normalized_distance = expected_distance / (len(self.ts_x)+len(self.ts_y))
        expected_shape_distance = 98.53608711667624
        expected_shape_normalized_distance = expected_shape_distance / (len(self.ts_x) + len(self.ts_y))

        ts_x_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_x, 2). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(self.shape_descriptor)

        ts_y_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_y, 2). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(self.shape_descriptor)

        dist_matrices = ts_x_shape_descriptor.calc_distance_matrices(
            ts_y_shape_descriptor
        )

        dtw_results = [
            dtw(dist_mat.dist_matrix)
            for dist_mat in dist_matrices.distance_matrices_list
        ]

        shape_dtw_independent_res = MultivariateShapeDTWIndependent(
            self.ts_x, self.ts_y, dtw_results=dtw_results
        )._calc_distances()

        self.assertAlmostEqual(
            expected_distance,
            shape_dtw_independent_res.distance
        )

        self.assertAlmostEqual(
            expected_normalized_distance,
            shape_dtw_independent_res.normalized_distance
        )

        self.assertAlmostEqual(
            expected_shape_distance,
            shape_dtw_independent_res.shape_distance
        )

        self.assertAlmostEqual(
            expected_shape_normalized_distance,
            shape_dtw_independent_res.shape_normalized_distance
        )

    def test_calc_shape_dtw(self):
        expected_distance = 35.599999999999994
        expected_normalized_distance = expected_distance / (len(self.ts_x) + len(self.ts_y))
        expected_shape_distance = 98.53608711667624
        expected_shape_normalized_distance = expected_shape_distance / (len(self.ts_x) + len(self.ts_y))

        shape_dtw_independent_res = MultivariateShapeDTWIndependent(
            self.ts_x, self.ts_y
        ).calc_shape_dtw(2, self.shape_descriptor)

        self.assertAlmostEqual(
            expected_distance,
            shape_dtw_independent_res.distance
        )

        self.assertAlmostEqual(
            expected_normalized_distance,
            shape_dtw_independent_res.normalized_distance
        )

        self.assertAlmostEqual(
            expected_shape_distance,
            shape_dtw_independent_res.shape_distance
        )

        self.assertAlmostEqual(
            expected_shape_normalized_distance,
            shape_dtw_independent_res.shape_normalized_distance
        )

if __name__ == '__main__':
    unittest.main()