import unittest

import numpy as np

from shapedtw.utils import Utils
from dtw import stepPattern
from shapedtw.exceptions import TooManyDimensions, IncompatibleDimensionality


class TestStepPatternCanonicalization(unittest.TestCase):

    def test_step_pattern_canonicalization_from_string(self):
        sp_txt = "symmetric1"
        sp_canonicalized = Utils.canonicalizeStepPattern(sp_txt)
        self.assertIs(sp_canonicalized, stepPattern.symmetric1)

    def test_step_pattern_canonicalization_from_object(self):
        sp_obj = stepPattern.symmetric1
        sp_canonicalized = Utils.canonicalizeStepPattern(sp_obj)
        self.assertIs(sp_canonicalized, sp_obj)

    def test_step_pattern_exceptions_for_wrong_name(self):
        sp_txt = "non_existing_step_pattern"
        with self.assertRaises(AttributeError):
            Utils.canonicalizeStepPattern(sp_txt)


class TestClassesCompatibilityChecker(unittest.TestCase):

    class ParentClass:
        pass

    class ChildClass(ParentClass):
        pass

    def test_positive_check(self):
        some_list = []
        another_list = [1, 2]
        verification_results = Utils.are_objects_of_same_classes(some_list, another_list)
        self.assertTrue(verification_results)

    def test_negative_check(self):
        list_obj = []
        string_obj = "foobar"
        verification_results = Utils.are_objects_of_same_classes(list_obj, string_obj)
        self.assertFalse(verification_results)

    def test_positive_check_inheritance(self):
        parent_object = self.ParentClass()
        child_object = self.ChildClass()
        verification_results = Utils.are_objects_of_same_classes(
            reference_obj=child_object, other_obj=parent_object
        )
        self.assertTrue(verification_results)

    def test_negative_check_inheritance(self):
        parent_object = self.ParentClass()
        child_object = self.ChildClass()
        verification_results = Utils.are_objects_of_same_classes(
            reference_obj=parent_object, other_obj=child_object
        )
        self.assertFalse(verification_results)

class TestDimensionalityCheckers(unittest.TestCase):

    array_1 = np.array([1])
    array_2_2 = np.random.randn(2, 2)
    array_2_3 = np.random.randn(2, 3)
    array_2_2_2 = np.random.randn(2, 2, 2)

    def test_dimension_number_getter(self):
        array_dims = [
            Utils.get_number_of_dimensions(arr)
            for arr in [self.array_1, self.array_2_2, self.array_2_3, self.array_2_2_2]
        ]

        self.assertEqual(array_dims, [1, 2, 2, 3])

    def test_number_of_dimension_checker_positive(self):
        one_dim_positive = Utils.number_of_dimensions_equal(self.array_1, self.array_1)
        two_dim_positive = Utils.number_of_dimensions_equal(self.array_2_2, self.array_2_3)
        self.assertTrue(
            all([one_dim_positive, two_dim_positive])
        )

    def test_number_of_dimension_checker_negative(self):
        one_two_dim_negative = Utils.number_of_dimensions_equal(self.array_1, self.array_2_2)
        two_three_dim_negative = Utils.number_of_dimensions_equal(self.array_2_2, self.array_2_2_2)
        self.assertFalse(
            any([one_two_dim_negative, two_three_dim_negative])
        )

    def test_number_of_series_checker_positive(self):
        one_series_positive = Utils.number_of_series_equal(self.array_1, self.array_1)
        two_series_positive = Utils.number_of_series_equal(self.array_2_2, self.array_2_2)
        self.assertTrue(
            all([one_series_positive, two_series_positive])
        )

    def test_number_of_series_checker_negative(self):
        two_series_negative = Utils.number_of_series_equal(self.array_2_2, self.array_2_3)
        self.assertFalse(two_series_negative)

    def test_too_many_dimensions_exception(self):
        with self.assertRaises(TooManyDimensions):
            Utils.number_of_series_equal(self.array_2_2_2, self.array_2_2_2)

    def test_incompatible_dimensionality_exception(self):
        with self.assertRaises(IncompatibleDimensionality):
            Utils.number_of_series_equal(self.array_1, self.array_2_2)


class TestOddChecker(unittest.TestCase):

    def test_odd_positive(self):
        test_1 = Utils.is_odd(1)
        test_3 = Utils.is_odd(3)
        test_57 = Utils.is_odd(57)
        self.assertTrue(all(
            [test_1, test_3, test_57]
        ))

    def test_odd_negative(self):
        test_2 = Utils.is_odd(2)
        test_4 = Utils.is_odd(4)
        test_56 = Utils.is_odd(56)
        self.assertTrue(all(
            [test_2, test_4, test_56]
        ))