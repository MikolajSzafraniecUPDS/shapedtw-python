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

if __name__ == '__main__':
    unittest.main()