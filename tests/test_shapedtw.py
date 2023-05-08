import unittest

import numpy as np

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




if __name__ == '__main__':
    unittest.main()