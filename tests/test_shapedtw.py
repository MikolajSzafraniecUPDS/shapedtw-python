import unittest

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

if __name__ == '__main__':
    unittest.main()