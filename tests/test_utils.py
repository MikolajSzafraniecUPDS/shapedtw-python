import unittest
from shapedtw.utils import Utils
from dtw import stepPattern
from shapedtw.exceptions import ObjectOfWrongClass


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
