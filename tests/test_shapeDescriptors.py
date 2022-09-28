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

if __name__ == '__main__':
    unittest.main()
