import sys

import numpy as np
from shapedtw.exceptions import IncompatibleDimensionality, TooManyDimensions, ProvidedStepPatternDoesNotExists


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
            res = s
        else:
            try:
                res = getattr(sys.modules["dtw.stepPattern"], s)
            except AttributeError as ae:
                raise ProvidedStepPatternDoesNotExists(
                    "There is no such step pattern. Please check if there is no type in the step pattern name."
                )

        return res

    @staticmethod
    def are_objects_of_same_classes(reference_obj: object, other_obj: object) -> bool:
        are_same_class = isinstance(reference_obj, other_obj.__class__)
        return are_same_class

    @staticmethod
    def get_number_of_dimensions(x: np.ndarray) -> int:
        return len(x.shape)

    @staticmethod
    def number_of_dimensions_equal(ts_x: np.ndarray, ts_y: np.ndarray) -> bool:
        ts_x_dim_number = Utils.get_number_of_dimensions(ts_x)
        ts_y_dim_number = Utils.get_number_of_dimensions(ts_y)

        return ts_x_dim_number == ts_y_dim_number

    @staticmethod
    def number_of_series_equal(ts_x: np.ndarray, ts_y: np.ndarray) -> bool:

        ts_x_dim_number = Utils.get_number_of_dimensions(ts_x)
        ts_y_dim_number = Utils.get_number_of_dimensions(ts_y)

        if (ts_x_dim_number > 2) or (ts_y_dim_number > 2):
            raise TooManyDimensions(ts_x, ts_y)

        if not Utils.number_of_dimensions_equal(ts_x, ts_y):
            raise IncompatibleDimensionality(ts_x, ts_y)

        if (ts_x_dim_number == 1) and (ts_y_dim_number == 1):
            return True
        elif (ts_x_dim_number == 2) and (ts_y_dim_number == 2):
            ts_x_n_series = ts_x.shape[1]
            ts_y_n_series = ts_y.shape[1]
            return ts_x_n_series == ts_y_n_series

    @staticmethod
    def is_odd(num):
        return num & 0x1
