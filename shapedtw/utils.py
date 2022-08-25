import sys

import numpy as np

from .exceptions import ObjectOfWrongClass, IncompatibleDimensionality, IncompatibleSeriesNumber


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
    def are_objects_of_same_classes(reference_obj: object, other_obj: object) -> bool:
        are_same_class = isinstance(reference_obj, other_obj.__class__)
        return are_same_class

    @staticmethod
    def verify_shape_compatibility(ts_x: np.ndarray, ts_y: np.ndarray) -> None:
        ts_x_shape = ts_x.shape
        ts_y_shape = ts_y.shape

        ts_x_dim = len(ts_x_shape)
        ts_y_dim = len(ts_y_shape)

        if ts_x_dim != ts_y_dim:
            raise IncompatibleDimensionality(ts_x_dim, ts_y_dim)

        if ts_x_dim == 2:
            ts_x_n_series = ts_x_shape[1]
            ts_y_n_series = ts_y_shape[1]
            if ts_x_n_series != ts_y_n_series:
                raise IncompatibleSeriesNumber(ts_x_n_series, ts_y_n_series)

    @staticmethod
    def is_odd(num):
        return num & 0x1
