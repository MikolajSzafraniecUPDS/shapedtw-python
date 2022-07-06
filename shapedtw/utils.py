import sys

from .exceptions import ObjectOfWrongClass

class Utils:

    @staticmethod
    def canonicalizeStepPattern(s):
        """
        Function taken directly from the dtw package due to the
        issues related to importing it from there.
        :param s: Object of class StepPattern or string representing step pattern name
        :return: StepPattern object
        """
        if hasattr(s, "mx"):
            return s
        else:
            return getattr(sys.modules["dtw.stepPattern"], s)

    @staticmethod
    def verify_classes_compatibility(reference_obj: object, other_obj: object) -> None:
        are_same_class = isinstance(reference_obj, other_obj.__class__)
        if not are_same_class:
            raise ObjectOfWrongClass(
                actual_cls=other_obj.__class__,
                expected_cls=reference_obj.__class__
            )
