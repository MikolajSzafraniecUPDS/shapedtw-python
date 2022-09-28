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


if __name__ == '__main__':
    unittest.main()
