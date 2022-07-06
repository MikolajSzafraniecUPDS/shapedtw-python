from dtw import *

from shapedtw.preprocessing import *
from shapedtw.exceptions import *
from shapedtw.shapeDescriptors import *


class StepPatternMatrixTransformator:

    """
    Class for transfoming step pattern matrix to more convenient
    form of dictionary
    """

    def __init__(self, step_pattern: str):
        self.step_pattern = step_pattern
        self.step_pattern_matrix = self._get_step_pattern_matrix()

    def _get_step_pattern_matrix(self) -> array:
        step_pattern_obj = ShapeDTW._canonicalizeStepPattern(self.step_pattern)
        return step_pattern_obj.mx

    def _get_segments_number(self) -> int:
        return int(self.step_pattern_matrix[:, 0].max())

    def _get_matrix_segment(self, segment_number: int) -> array:
        ind_mask = self.step_pattern_matrix[:, 0] == segment_number
        return self.step_pattern_matrix[ind_mask, :].copy()

    def _get_segment_length(self, segment_number: int) -> int:
        mask = self.step_pattern_matrix[:, 0] == segment_number
        return mask.sum()-1

    def _get_segment_pattern(self, segment_number: int) -> tuple:
        segment = self._get_matrix_segment(segment_number)
        return int(segment[0, 1]), int(segment[0, 2])

    def _segment_to_dict(self, segment_num):
        segment = self._get_matrix_segment(segment_num)
        segment_length = self._get_segment_length(segment_num)
        res = {
            i: {
                "x_index": int(segment[i+1, 1]),
                "y_index": int(segment[i+1, 2]),
                "weight": segment[i+1, 3]
            } for i in range(segment_length)
        }

        return res

    def step_pattern_matrix_to_dict(self):
        segments_number = self._get_segments_number()
        segments_iter = range(1, segments_number+1)
        res = {
            self._get_segment_pattern(i): self._segment_to_dict(i)
            for i in segments_iter
        }

        return res


class DistanceReconstructor:

    """
    Class for reconstructing raw time series distance based on its
    distance matrix and warping path
    """

    def __init__(self,
                 step_pattern: str,
                 ts_x: array,
                 ts_y: array,
                 ts_x_wp: list,
                 ts_y_wp: list,
                 dist_method: str = "euclidean"):

        self.ts_x = ts_x,
        self.ts_y = ts_y,
        self.ts_x_warping_path = ts_x_wp
        self.ts_y_warping_path = ts_y_wp
        self.dist_method = dist_method
        self.distance_matrix = self._calc_distance_matrix()
        self.step_pattern_dictionary = StepPatternMatrixTransformator(step_pattern).step_pattern_matrix_to_dict()

    def _calc_distance_matrix(self):
        dist_matrix = DistanceMatrixCalculator(self.ts_x, self.ts_y, self.dist_method).\
            calc_distance_matrix()
        return dist_matrix

    def _calc_single_distance(self, x_index: int, y_index: int, single_pattern_dict: dict) -> float:
        target_x_index = x_index - single_pattern_dict["x_index"]
        target_y_index = y_index - single_pattern_dict["y_index"]
        distance = self.distance_matrix[target_x_index, target_y_index] * single_pattern_dict["weight"]
        return distance

    def _calc_distance_for_given_pattern(self, x_index: int, y_index: int, pattern: tuple) -> float:
        pattern_segment = self.step_pattern_dictionary[pattern]
        pattern_distance = sum([
            self._calc_single_distance(x_index,y_index,pattern_segment[key])
            for key in pattern_segment
        ])
        return pattern_distance

    def _get_indices_pairs(self):
        return list(
            zip(
                self.ts_x_warping_path,
                self.ts_y_warping_path
            )
        )

    def _get_indices_patterns(self):
        x_indices_diff = self.ts_x_warping_path[1:] - self.ts_x_warping_path[:-1]
        y_indices_diff = self.ts_y_warping_path[1:] - self.ts_y_warping_path[:-1]

        return list(zip(x_indices_diff, y_indices_diff))

    def calc_raw_ts_distance(self):
        indices_pairs = self._get_indices_pairs()
        indices_patterns = self._get_indices_patterns()
        raw_series_distance = self.distance_matrix[0,0]
        distances_list = [
            self._calc_distance_for_given_pattern(x_ind, y_ind, indices_patterns[i])
            for (i, (x_ind, y_ind)) in enumerate(indices_pairs[1:])
        ]

        raw_series_distance += sum(distances_list)

        return raw_series_distance


class ShapeDTW:

    def __init__(self,
                 dtw_obj: DTW,
                 ts_x: array,
                 ts_y: array,
                 step_pattern: str = "symmetric1",
                 dist_method: str = "euclidean"):

        self.dtw_results = dtw_obj
        self.ts_x = ts_x
        self.ts_y = ts_y
        self.step_pattern = step_pattern
        self.dist_method = dist_method
        self.set_distance()

    @staticmethod
    def _canonicalizeStepPattern(s):
        """Return object by string"""
        if hasattr(s, "mx"):
            return s
        else:
            return getattr(sys.modules["dtw.stepPattern"], s)

    def _calc_raw_series_distance(self, dist_method: str = "euclidean"):
        dist_reconstructor = DistanceReconstructor(
            step_pattern=self.step_pattern,
            ts_x=self.ts_x,
            ts_y=self.ts_y,
            ts_x_wp=self.dtw_results.index1s,
            ts_y_wp=self.dtw_results.index2s,
            dist_method=dist_method
        )

        return dist_reconstructor.calc_raw_ts_distance()

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

    def set_distance(self):
        self.distance = self._calc_raw_series_distance(self.dist_method)
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
    shape_dtw_results = ShapeDTW(dtw_results, x, y, step_pattern, kwargs["dist_method"])

    return shape_dtw_results
