import numpy as np

from dtw import *

from shapedtw.preprocessing import *
from shapedtw.exceptions import *
from shapedtw.shapeDescriptors import *

class ShapeDTW:

    def __init__(self, dtw_obj: DTW, ts_x: array, ts_y: array, step_pattern: str = "symmetric1"):
        self.dtw_results = dtw_obj
        self.ts_x = ts_x
        self.ts_y = ts_y
        self.step_pattern = step_pattern
        self.set_distance()

    @staticmethod
    def _canonicalizeStepPattern(s):
        """Return object by string"""
        if hasattr(s, "mx"):
            return s
        else:
            return getattr(sys.modules["dtw.stepPattern"], s)

    def _calc_raw_series_distance(self, dist_method: str = "euclidean"):
        x, y = np.atleast_2d(self.ts_x).T, np.atleast_2d(self.ts_y).T
        raw_series_dist_matrix = cdist(x, y, dist_method)
        ind_x = self.dtw_results.index1
        ind_y = self.dtw_results.index2
        path = raw_series_dist_matrix[ind_x, ind_y]
        return path.sum()

    def _calc_raw_series_normalized_distance(self):
        step_pattern = self._canonicalizeStepPattern(self.step_pattern)
        norm = step_pattern.hint

        n, m = len(self.ts_x), len(self.ts_y)

        if norm == "N+M":
            normalized_distance = self.distance / (n + m)
        elif norm == "N":
            normalized_distance = self.distance / n
        elif norm == "M":
            normalized_distance = self.distance / m
        else:
            normalized_distance = np.nan

        return normalized_distance


    def set_distance(self, distance_calc_method: str = "raw_series_distance"):
        self.distance = self._calc_raw_series_distance()
        self.normalizedDistance = self._calc_raw_series_normalized_distance()
        self.shape_distance = self.dtw_results.distance
        self.shape_normalizedDistance = self.dtw_results.normalizedDistance

def shape_dtw(x: array, y: array,
              subsequence_width: int,
              shape_descriptor: ShapeDescriptor,
              step_pattern: str = "symmetric2",
              **kwargs):

    if "dist_method" not in kwargs:
        kwargs["dist_method"] = "euclidean"

    ts_x_shape_descriptor = SubsequenceBuilder(x, subsequence_width).\
        transform_time_series_to_subsequences().\
        get_shape_descriptors(shape_descriptor)

    ts_y_shape_descriptor = SubsequenceBuilder(y, subsequence_width).\
        transform_time_series_to_subsequences().\
        get_shape_descriptors(shape_descriptor)

    distance_matrix = ts_x_shape_descriptor.calc_distance_matrix(
        ts_y_shape_descriptor, dist_method=kwargs["dist_method"]
    )

    dtw_results = dtw(distance_matrix.dist_matrix, step_pattern=step_pattern, **kwargs)
    shape_dtw_results = ShapeDTW(dtw_results, x, y, step_pattern)

    return shape_dtw_results