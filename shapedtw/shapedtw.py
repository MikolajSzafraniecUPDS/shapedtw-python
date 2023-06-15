##
## Copyright (c) of Mikołaj Szafraniec
##
## This file is part of the ShapeDTW package.
##
## ShapeDTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## ShapeDTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ShapeDTW.  If not, see <http://www.gnu.org/licenses/>.
##

"""Classes and methods implementing shape dtw algorithm"""

from __future__ import annotations

from dtw import *

from shapedtw.preprocessing import *
from shapedtw.shapeDescriptors import *
from shapedtw.utils import Utils
from dataclasses import dataclass
from pandas import Series, DataFrame
from numpy import ndarray

class StepPatternMatrixTransformator:

    """
    Class for transforming step pattern matrix to more convenient
    form of dictionary. It is required to reproduce the distance
    for raw time series after the warping path is determined using
    distance matrix calculated based on shape descriptors.

    Attributes
    ---------------
    step_pattern: str:
        name of step pattern which was used to calculate warping path
    """

    def __init__(self, step_pattern: str):
        self.step_pattern = step_pattern
        self.step_pattern_matrix = self._get_step_pattern_matrix()

    def _get_step_pattern_matrix(self) -> ndarray:
        """
        Get StepPattern object based on step pattern name and
        retrieve step pattern matrix

        Returns
        ---------------
        :return: step pattern matrix as numpy array

        Examples
        --------
        >> from shapedtw.shapedtw import StepPatternMatrixTransformator
        >> spmt = StepPatternMatrixTransformator("symmetric2")
        >> res = spmt._get_step_pattern_matrix()
        >> print(res)
        [[ 1.  1.  1. -1.]
         [ 1.  0.  0.  2.]
         [ 2.  0.  1. -1.]
         [ 2.  0.  0.  1.]
         [ 3.  1.  0. -1.]
         [ 3.  0.  0.  1.]]
        """
        step_pattern_obj = Utils.canonicalizeStepPattern(self.step_pattern)
        return step_pattern_obj.mx

    def _get_segments_number(self) -> int:
        """
        Get number of step pattern's possible transitions
        schemas

        Returns
        ---------------
        :return: int - number of step pattern segments (transition schemas)
        """
        return int(self.step_pattern_matrix[:, 0].max())

    def _segment_number_in_range(self, segment_number: int) -> bool:
        """
        Checks that the number of segment is in the valid range

        Parameters
        ---------------
        :param segment_number: int - index of segment to check

        Returns
        ---------------
        :return: bool - result of test
        """
        return (segment_number > 0) & (segment_number <= self._get_segments_number())

    def _check_segment_number(self, segment_number: int) -> None:
        """
        Verify that segment number is in a valid range. If not
        SegmentIndexOutOfRange exception is raised.

        Parameters
        ---------------
        :param segment_number: int - index of segment to check

        Raises
        ---------------
        :raises SegmentIndexOutOfRange: index of segment is lower than
            zero or greater than the number of given step pattern's segments
        """
        if not self._segment_number_in_range(segment_number):
            raise SegmentIndexOutOfRange(
                provided_segment_number = segment_number,
                actual_number_of_segments = self._get_segments_number()
            )

    def _get_matrix_segment(self, segment_number: int) -> ndarray:
        """
        Get rows of step pattern's matrix representing particular
        segment (transition schema).

        Parameters
        ---------------
        :param segment_number: index of segment

        Returns
        ---------------
        :return: numpy array - rows of step pattern's matrix
            representing particular segment
        """
        self._check_segment_number(segment_number)
        ind_mask = self.step_pattern_matrix[:, 0] == segment_number
        return self.step_pattern_matrix[ind_mask, :].copy()

    def _get_segment_length(self, segment_number: int) -> int:
        """
        Get a length of given segment (transition schema). It
        tells us how many values we need to consider in distance
        reconstruction for particular step.

        Parameters
        ---------------
        :param segment_number: index of segment

        Returns
        ---------------
        :return: int - number of values the segment consists of
        """
        self._check_segment_number(segment_number)
        mask = self.step_pattern_matrix[:, 0] == segment_number
        return mask.sum()-1

    def _get_segment_pattern(self, segment_number: int) -> tuple:
        """
        Get the transition schema for given segment. For example
        output (1, 1) means that warping path moves to the next
        observation of both x (query) and y (reference) time series.
        (1, 0) would mean that we go to the next observation of
        query time series, but stick to the current observation of
        y (reference) series.

        Parameters
        ---------------
        :param segment_number: int - index of segment

        Returns
        ---------------
        :return: a tuple - transition schema for given segment
        """
        self._check_segment_number(segment_number)
        segment = self._get_matrix_segment(segment_number)
        return int(segment[0, 1]), int(segment[0, 2])

    def _segment_to_dict(self, segment_number: int) -> dict:
        """
        Transform given segment (transition schema) to python
        dictionary.

        Parameters
        ---------------
        :param segment_number: int - index of segment

        Returns
        ---------------
        :return: dictionary representing given segment (transition
            schema) of a step pattern

        Examples
        --------
        >> from shapedtw.shapedtw import StepPatternMatrixTransformator
        >> spmt = StepPatternMatrixTransformator("symmetric2")
        >> spmt._segment_to_dict(1)
        {0: {'x_index': 0, 'y_index': 0, 'weight': 2.0}}
        """
        self._check_segment_number(segment_number)
        segment = self._get_matrix_segment(segment_number)
        segment_length = self._get_segment_length(segment_number)
        res = {
            i: {
                "x_index": int(segment[i+1, 1]),
                "y_index": int(segment[i+1, 2]),
                "weight": segment[i+1, 3]
            } for i in range(segment_length)
        }

        return res

    def step_pattern_matrix_to_dict(self) -> dict:
        """
        Convert step pattern matrix to dictionary

        Returns
        ---------------
        :return: dictionary representing given step pattern

        Examples
        --------
        >> from shapedtw.shapedtw import StepPatternMatrixTransformator
        >> spmt = StepPatternMatrixTransformator("symmetric2")
        >> spmt.step_pattern_matrix_to_dict()
        {(1, 1): {0: {'x_index': 0, 'y_index': 0, 'weight': 2.0}},
         (0, 1): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}},
         (1, 0): {0: {'x_index': 0, 'y_index': 0, 'weight': 1.0}}}
        """
        segments_number = self._get_segments_number()
        segments_iter = range(1, segments_number+1)
        res = {
            self._get_segment_pattern(i): self._segment_to_dict(i)
            for i in segments_iter
        }

        return res


class DistanceReconstructor:

    """
    This class allows to calculate distance between raw time series
    based on provided warping paths. ShapeDTW algorithm calculates optimal
    warping paths using distance matrix determined based on shape descriptors.
    We can use such total distance in several applications, however one might
    prefer to use raw time series distance. Class representing shape DTW results
    contains both versions.

    Attributes
    ---------------
    step_pattern: str:
        name of step pattern used to calculate warping paths
    ts_x: ndarray:
        query time series
    ts_y: ndarray:
        reference time series
    ts_x_wp:
        warping path for query time series as a
        list of indices
    ts_y_wp:
        warping path for reference time series as a
        list of indices
    dist_method: str:
        type of distance used for determine warping paths
    """

    def __init__(self,
                 step_pattern: str,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 ts_x_wp: list,
                 ts_y_wp: list,
                 dist_method: str = "euclidean"):
        """
        Constructs a DistanceReconstructor object

        :param step_pattern: name of step pattern used to calculate warping paths
        :param ts_x: query time series
        :param ts_y: reference time series
        :param ts_x_wp: warping path for query time series as a list of indices
        :param ts_y_wp: warping path for reference time series as a list of indices
        :param dist_method: type of distance used for determine warping paths
        """
        self.ts_x = ts_x
        self.ts_y = ts_y
        self.ts_x_warping_path = ts_x_wp
        self.ts_y_warping_path = ts_y_wp
        self.dist_method = dist_method
        self.distance_matrix = self._calc_distance_matrix()
        self.step_pattern_dictionary = StepPatternMatrixTransformator(step_pattern).step_pattern_matrix_to_dict()

    def _calc_distance_matrix(self) -> ndarray:
        """
        Calculates distance matrix for raw time series

        Returns
        ---------------
        :return: distance matrix as ndarray
        """
        dist_matrix = DistanceMatrixCalculator(self.ts_x, self.ts_y, self.dist_method).\
            calc_distance_matrix()
        return dist_matrix

    def _calc_single_distance(self, x_index: int, y_index: int, step_pattern_segment_dict: dict) -> float:
        """
        Calculates weighted distance between a pair of time series observations for given
        element of step pattern segment.

        Parameters
        ---------------
        :param x_index: index of query time series
        :param y_index: index of reference time series
        :param step_pattern_segment_dict: dictionary for given element of step pattern segment

        Returns
        ---------------
        :return: weighted distance between time series elements
        """
        target_x_index = x_index - step_pattern_segment_dict["x_index"]
        target_y_index = y_index - step_pattern_segment_dict["y_index"]
        distance = self.distance_matrix[target_x_index, target_y_index] * step_pattern_segment_dict["weight"]
        return distance

    def _calc_distance_for_given_pattern(self, x_index: int, y_index: int, pattern: tuple) -> float:
        """
        Calculates distance between a pair of time series observations for given
        step pattern segment.

        Parameters
        ---------------
        :param x_index: index of query time series
        :param y_index: index of reference time series
        :param pattern: transition pattern as a tuple of indices diff

        Returns
        ---------------
        :return: distance for given time series elements and step pattern segment
        """
        pattern_segment = self.step_pattern_dictionary[pattern]
        pattern_distance = sum([
            self._calc_single_distance(x_index, y_index, pattern_segment[key])
            for key in pattern_segment
        ])
        return pattern_distance

    def _get_indices_pairs(self) -> List[tuple]:
        """
        Converts warping paths into a list of tuples, representing pairs of
        time series indices linked by the shape dtw algorithm

        Returns
        ---------------
        :return: warping path in the form of a list of tuples,
            representing pairs of time series indices

        Examples
        --------
        >> import numpy as np
        >> from shapedtw.shapedtw import DistanceReconstructor
        >> ts_x = np.array([1.1, 1.5, 6.7, 4.5, 1.3])
        >> ts_y = np.array([2.1, 2.5, 1.7, 3.3, 6.6])
        >> ts_x_wp = np.array([0, 0, 1, 1, 2, 3, 4])
        >> ts_y_wp = np.array([0, 1, 2, 3, 4, 4, 4])
        >> ds = DistanceReconstructor("symmetric2", ts_x, ts_y, ts_x_wp, ts_y_wp)
        >> res = ds._get_indices_pairs()
        >> print(res)
        [(0, 0), (0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (4, 4)]
        """
        return list(
            zip(
                self.ts_x_warping_path,
                self.ts_y_warping_path
            )
        )

    def _get_indices_patterns(self) -> List[tuple]:
        """
        Calculates differences between a pairs of time series linked by the
        shape dtw algorithm. It allows to reconstruct step pattern segments
        required for calculate dtw distance between raw time series.

        Returns
        ---------------
        :return: step pattern transitions in the form of a
            list of tuples

        Examples
        --------
        >> import numpy as np
        >> from shapedtw.shapedtw import DistanceReconstructor
        >> ts_x = np.array([1.1, 1.5, 6.7, 4.5, 1.3])
        >> ts_y = np.array([2.1, 2.5, 1.7, 3.3, 6.6])
        >> ts_x_wp = np.array([0, 0, 1, 1, 2, 3, 4])
        >> ts_y_wp = np.array([0, 1, 2, 3, 4, 4, 4])
        >> ds = DistanceReconstructor("symmetric2", ts_x, ts_y, ts_x_wp, ts_y_wp)
        >> res = ds._get_indices_patterns()
        >> print(res)
        [(0, 1), (1, 1), (0, 1), (1, 1), (1, 0), (1, 0)]
        """
        x_indices_diff = self.ts_x_warping_path[1:] - self.ts_x_warping_path[:-1]
        y_indices_diff = self.ts_y_warping_path[1:] - self.ts_y_warping_path[:-1]

        return list(zip(x_indices_diff, y_indices_diff))

    def calc_raw_ts_distance(self) -> float:
        """
        Calculates distance between raw time series based on
        provided warping paths.

        Returns
        ---------------
        :return: distance as a float number

        Examples
        --------
        >> import numpy as np
        >> from shapedtw.shapedtw import DistanceReconstructor
        >> ts_x = np.array([1.1, 1.5, 6.7, 4.5, 1.3])
        >> ts_y = np.array([2.1, 2.5, 1.7, 3.3, 6.6])
        >> ts_x_wp = np.array([0, 0, 1, 1, 2, 3, 4])
        >> ts_y_wp = np.array([0, 1, 2, 3, 4, 4, 4])
        >> ds = DistanceReconstructor("symmetric2", ts_x, ts_y, ts_x_wp, ts_y_wp)
        >> res = ds.calc_raw_ts_distance()
        >> print(res)
        12.2
        """
        indices_pairs = self._get_indices_pairs()
        indices_patterns = self._get_indices_patterns()
        raw_series_distance = self.distance_matrix[0, 0]
        distances_list = [
            self._calc_distance_for_given_pattern(x_ind, y_ind, indices_patterns[i])
            for (i, (x_ind, y_ind)) in enumerate(indices_pairs[1:])
        ]

        raw_series_distance += sum(distances_list)

        return raw_series_distance


@dataclass
class ShapeDTWResults:

    """
    Dataclass representing distances between time series calculated
    using shape dtw warping paths

    Attributes
    ---------------
    distance: float:
        distance calculated based on raw time series values
    normalized_distance: float:
        normalized distance between raw time series - calculated
        only if it is possible for given step pattern; nan otherwise
    shape_distance: float:
        distance between time series calculated based on shape
        descriptor values
    shape_normalized_distance: float:
        normalized distance calculated based on shape descriptor
        values - - calculated only if it is possible for given
        step pattern; nan otherwise
    """

    distance: float
    normalized_distance: float
    shape_distance: float
    shape_normalized_distance: float


class ShapeDTW:

    """
    A class representing the results of the shape dtw and containing
    a set of methods used to calculate them. It is a parent class for
    classes representing more specific shape dtw variants (univariate,
    multivariate dependent and multivariate independent).

    Attributes
    ---------------
    ts_x: ndarray:
        query time series
    ts_y: ndarray:
        reference time series
    step_pattern: str:
        name of step pattern used to calculate warping paths
    dist_method: str:
        type of distance
    dtw_res: DTW | List[DTW]:
        results of dtw algorithm applied to shape dtw distance
        matrix; it contains all needed metadata, such as warping
        paths, distance, normalized distance, etc.
    shape_dtw_results: ShapeDTWResults:
        shape dtw distances in the form of ShapeDTWResults class
    """

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_res: DTW | List[DTW] = None,
                 shape_dtw_results: ShapeDTWResults = None):
        """
        Constructs a ShapeDTW object

        Parameters
        ---------------
        :param ts_x: query time series
        :param ts_y: reference time series
        :param step_pattern: name of step pattern used to calculate warping paths
        :param dist_method: type of distance
        :param dtw_res: results of dtw algorithm applied to shape dtw distance
            matrix; it contains all needed metadata, such as warping
            paths, distance, normalized distance, etc.
        :param shape_dtw_results: shape dtw distances in the form of
            ShapeDTWResults class
        """
        self._dtw_results = dtw_res
        self.ts_x = ts_x
        self.ts_y = ts_y
        self._step_pattern = step_pattern
        self._dist_method = dist_method
        self._shape_dtw_results = shape_dtw_results

    def _calc_raw_series_distance(self) -> float:
        """
        Calculates distance between raw values of time series
        based on warping paths determined using shape dtw
        algorithm

        Returns
        ---------------
        :return: distance between raw time series as a float
            number
        """
        dist_reconstructor = DistanceReconstructor(
            step_pattern=self._step_pattern,
            ts_x=self.ts_x,
            ts_y=self.ts_y,
            ts_x_wp=self._dtw_results.index1s,
            ts_y_wp=self._dtw_results.index2s,
            dist_method=self._dist_method
        )

        return dist_reconstructor.calc_raw_ts_distance()

    def _calc_raw_series_normalized_distance(self, distance: float) -> float:
        """
        Calculates normalized distance between raw values of
        time series based on warping paths determined using
        shape dtw algorithm

        Parameters
        ---------------
        :param distance: distance between raw values of time series
            calculated using '_calc_raw_series_distance' method

        Returns
        ---------------
        :return: normalized distance between raw values of time series
            if possible; nan for some types of step patterns for which
            normalization is not possible
        """
        step_pattern = Utils.canonicalizeStepPattern(self._step_pattern)
        norm = step_pattern.hint

        # In case of multidimensional time series 'len' function returns
        # number of rows, not number of all observations
        n, m = len(self.ts_x), len(self.ts_y)

        if norm == "N+M":
            normalized_distance = distance / (n + m)
        elif norm == "N":
            normalized_distance = distance / n
        elif norm == "M":
            normalized_distance = distance / m
        else:
            normalized_distance = np.nan

        return normalized_distance

    def _calc_distances(self) -> ShapeDTWResults:
        """
        Calculates a full set of shape dtw distances

        Returns
        ---------------
        :return: a set of distances in the form of ShapeDTWResults
            object
        """
        distance = self._calc_raw_series_distance()
        normalized_distance = self._calc_raw_series_normalized_distance(distance)
        shape_distance = self._dtw_results.distance
        shape_normalized_distance = self._dtw_results.normalizedDistance

        return ShapeDTWResults(
            distance, normalized_distance,
            shape_distance, shape_normalized_distance
        )

    def _get_distance(self) -> float:
        """
        Getter - get distance between raw time series if already
        calculated

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: distance between raw time series
        """
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_normalized_distance(self) -> float:
        """
        Getter - get normalized distance between raw time series
        if already calculated

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: normalized distance between raw time series
        """
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.normalized_distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_shape_descriptor_distance(self) -> float:
        """
        Getter - get distance between shape descriptors of time series
        if already calculated

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: distance between shape descriptors values
        """
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.shape_distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_shape_descriptor_normalized_distance(self) -> float:
        """
        Getter - get normalized distance between shape descriptors of
        time series if already calculated

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: normalized distance between shape descriptors values
        """
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.shape_normalized_distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _set_distance(self, value):
        """
        Setter preventing from setting distance explicitly, omitting
        actual process of calculating shape dtw

        Parameters
        ---------------
        :param value: distance value

        Raises
        ---------------
        :raises DistanceSettingNotPossible: distances can only be set
            by calculating shape dtw, not by explicit assignment
        """
        raise DistanceSettingNotPossible(
            "ShapeDTW distance can be set only using 'calc_shape_dtw' method"
        )

    def _get_index1(self) -> ndarray:
        """
        Getter - get warping path for query (x) series

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping path for query series as a numpy array
        """
        if self._dtw_results is not None:
            return self._dtw_results.index1
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_index2(self) -> ndarray:
        """
        Getter - get warping path for reference (y) series

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping path for reference series as a numpy array
        """
        if self._dtw_results is not None:
            return self._dtw_results.index2
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_index1s(self) -> ndarray:
        """
        Getter - get warping path for query (x) series with intermediate
            steps for multi-step patterns (like 'asymmetricP05()') excluded

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping path for query series as a numpy array, with
            intermediate steps excluded
        """
        if self._dtw_results is not None:
            return self._dtw_results.index1s
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_index2s(self) -> ndarray:
        """
        Getter - get warping path for reference (y) series with intermediate
            steps for multi-step patterns (like 'asymmetricP05()') excluded

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping path for reference series as a numpy array, with
            intermediate steps excluded
        """
        if self._dtw_results is not None:
            return self._dtw_results.index2s
        else:
            raise ShapeDTWNotCalculatedYet()

    def _set_index(self, value):
        """
        Setter preventing from setting warping paths explicitly, omitting
        actual process of calculating shape dtw

        Parameters
        ---------------
        :param value: warping path value

        Raises
        ---------------
        :raises WarpingPathSettingNotPossible: warping paths can only be set
            by calculating shape dtw, not by explicit assignment
        """
        raise WarpingPathSettingNotPossible(
            "Warping paths can be set only using 'calc_shape_dtw' method"
        )

    # Properties
    distance = property(_get_distance, _set_distance)
    normalized_distance = property(_get_normalized_distance, _set_distance)
    shape_distance = property(_get_shape_descriptor_distance, _set_distance)
    shape_normalized_distance = property(_get_shape_descriptor_normalized_distance, _set_distance)

    index1 = property(_get_index1, _set_index)
    index2 = property(_get_index2, _set_index)
    index1s = property(_get_index1s, _set_index)
    index2s = property(_get_index2s, _set_index)


class UnivariateShapeDTW(ShapeDTW):

    """
    Class representing results of univariate shape dtw and containing
    a method used to calculate them

    Attributes
    ---------------
    ts_x: ndarray:
        query time series
    ts_y: ndarray:
        reference time series
    step_pattern: str:
        name of step pattern used to calculate warping paths
    dist_method: str:
        type of distance
    dtw_res: DTW:
        results of dtw algorithm applied to shape dtw distance
        matrix; it contains all needed metadata, such as warping
        paths, distance, normalized distance, etc.
    shape_dtw_results: ShapeDTWResults:
        shape dtw distances in the form of ShapeDTWResults class
    """

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_results: DTW = None,
                 shape_dtw_results: ShapeDTWResults = None):
        """
        Constructs a UnivariateShapeDTW object

        :param ts_x: query time series
        :param ts_y: reference time series
        :param step_pattern: name of step pattern used to calculate warping paths
        :param dist_method: type of distance
        :param dtw_results: results of dtw algorithm applied to shape dtw distance
            matrix; it contains all needed metadata, such as warping
            paths, distance, normalized distance, etc.
        :param shape_dtw_results: shape dtw distances in the form of
            ShapeDTWResults class
        """
        super().__init__(ts_x, ts_y, step_pattern, dist_method, dtw_results, shape_dtw_results)

    def calc_shape_dtw(self,
                       subsequence_width: int,
                       shape_descriptor: ShapeDescriptor,
                       **kwargs) -> UnivariateShapeDTW:
        """
        Calculates univariate shape dtw and set proper attributes inside
        a given instance of the class (_dtw_results and _shape_dtw_results).
        In order to calculate shape dtw we need to get shape descriptors of
        query and reference time series, construct a distance matrix for them
        and pass such matrix to the 'dtw' function from dtw-python package. It
        calculates warping path using shape descriptor distances instead of raw
        time series values, as in case of 'standard' dtw.

        Parameters
        ---------------
        :param subsequence_width: width of subsequence
        :param shape_descriptor: shape descriptor
        :param kwargs: keyword arguments which will be passed to the 'dtw'
            function executing dtw algorithm on the top of the distance matrix
            determined using shape descriptors

        Returns
        ---------------
        :return: self - an instance of the UnivariateShapeDTW with proper attributes
            assigned ('_dtw_results' and '_shape_dtw_results')
        """
        ts_x_shape_descriptor = UnivariateSubsequenceBuilder(self.ts_x, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        ts_y_shape_descriptor = UnivariateSubsequenceBuilder(self.ts_y, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        distance_matrix = ts_x_shape_descriptor.calc_distance_matrix(
            ts_y_shape_descriptor, dist_method=self._dist_method
        )

        dtw_results = dtw(distance_matrix.dist_matrix,
                          step_pattern=self._step_pattern,
                          **kwargs)

        self._dtw_results = dtw_results
        self._shape_dtw_results = self._calc_distances()

        return self


class MultivariateShapeDTWDependent(ShapeDTW):
    """
    Class representing results of multivariate, dependent shape dtw
    and containing a method used to calculate them.

    For multivariate dependent shape dtw there is one common warping
    paths for all time series dimensions. Details are described in this
    paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/

    Attributes
    ---------------
    ts_x: ndarray:
        query time series
    ts_y: ndarray:
        reference time series
    step_pattern: str:
        name of step pattern used to calculate warping paths
    dist_method: str:
        type of distance
    dtw_res: DTW:
        results of dtw algorithm applied to shape dtw distance
        matrix; it contains all needed metadata, such as warping
        paths, distance, normalized distance, etc.
    shape_dtw_results: ShapeDTWResults:
        shape dtw distances in the form of ShapeDTWResults class
    """

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_results: DTW = None,
                 shape_dtw_results: ShapeDTWResults = None):
        """
        Constructs a MultivariateShapeDTWDependent object

        :param ts_x: query time series
        :param ts_y: reference time series
        :param step_pattern: name of step pattern used to calculate warping paths
        :param dist_method: type of distance
        :param dtw_results: results of dtw algorithm applied to shape dtw distance
            matrix; it contains all needed metadata, such as warping
            paths, distance, normalized distance, etc.
        :param shape_dtw_results: shape dtw distances in the form of
            ShapeDTWResults class
        """
        super().__init__(ts_x, ts_y, step_pattern, dist_method, dtw_results, shape_dtw_results)

    def calc_shape_dtw(self,
                       subsequence_width: int,
                       shape_descriptor: ShapeDescriptor,
                       **kwargs):
        """
        Calculates multivariate, dependent shape dtw and set proper attributes
        inside a given instance of the class (_dtw_results and _shape_dtw_results).
        In order to calculate shape dtw we need to get shape descriptors of
        each dimension of query and reference time series separately, construct
        distance matrices for them, sum them up and pass such summed distance matrix
        to the 'dtw' function from dtw-python package. It calculates warping path,
        common for all the time series dimensions.

        Parameters
        ---------------
        :param subsequence_width: width of subsequence
        :param shape_descriptor: shape descriptor
        :param kwargs: keyword arguments which will be passed to the 'dtw'
            function executing dtw algorithm on the top of the distance matrix
            determined using shape descriptors

        Returns
        ---------------
        :return: self - an instance of the MultivariateShapeDTWDependent with proper attributes
            assigned ('_dtw_results' and '_shape_dtw_results')
        """
        ts_x_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_x, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        ts_y_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_y, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        dist_matrix = ts_x_shape_descriptor.calc_summed_distance_matrix(
            ts_y_shape_descriptor, dist_method=self._dist_method
        )

        dtw_results = dtw(dist_matrix.distance_matrix, step_pattern=self._step_pattern, **kwargs)
        self._dtw_results = dtw_results
        self._shape_dtw_results = self._calc_distances()

        return self


class MultivariateShapeDTWIndependent(ShapeDTW):

    """
    Class representing results of multivariate, independent shape dtw
    and containing a method used to calculate them.

    For multivariate independent shape dtw warping paths are calculated
    for each time dimension independently. In order to calculate shape
    dtw results in this case we need to calculate distances individually
    for all the dimensions (as in case of univariate dtw) and then
    calculate the sum of all distances. Details are described in this
    paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/

    Attributes
    ---------------
    ts_x: ndarray:
        query time series
    ts_y: ndarray:
        reference time series
    step_pattern: str:
        name of step pattern used to calculate warping paths
    dist_method: str:
        type of distance
    dtw_res: DTW:
        list of results of dtw algorithm applied to shape dtw distance
        matrices; it contains all needed metadata, such as warping
        paths, distance, normalized distance, etc.
    shape_dtw_results: ShapeDTWResults:
        shape dtw distances in the form of ShapeDTWResults class
    """

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_results: List[DTW] = None,
                 shape_dtw_results: ShapeDTWResults = None):
        """
        Constructs a MultivariateShapeDTWIndependent object

        :param ts_x: query time series
        :param ts_y: reference time series
        :param step_pattern: name of step pattern used to calculate warping paths
        :param dist_method: type of distance
        :param dtw_results: list of results of dtw algorithm applied to shape dtw distance
            matrices; it contains all needed metadata, such as warping
            paths, distance, normalized distance, etc.
        :param shape_dtw_results: shape dtw distances in the form of
            ShapeDTWResults class
        """

        super().__init__(ts_x, ts_y, step_pattern, dist_method, dtw_results, shape_dtw_results)

    def _calc_raw_series_distance(self) -> float:
        """
        Calculates distance between raw time series values using
        warping paths calculated by the shape dtw algorithm. For
        multivariate independent shape dtw warping paths are calculated
        for each time dimension independently. That's why
        we calculate distances separately for each dimension and sum
        them up as a result.

        Returns
        ---------------
        :return: distance for raw time series values
        """
        n_dim = self.ts_x.shape[1]
        dist_reconstructors = [
            DistanceReconstructor(step_pattern=self._step_pattern,
                                  ts_x=self.ts_x[:, ind].copy(),
                                  ts_y=self.ts_y[:, ind].copy(),
                                  ts_x_wp=self._dtw_results[ind].index1s,
                                  ts_y_wp=self._dtw_results[ind].index2s,
                                  dist_method=self._dist_method)
            for ind in range(n_dim)
        ]

        distances = [dist_recon.calc_raw_ts_distance() for dist_recon in dist_reconstructors]
        res = sum(distances)

        return res

    def _calc_distances(self):

        """
        Calculates a full set of shape dtw distances

        Returns
        ---------------
        :return: a set of distances in the form of ShapeDTWResults
            object
        """

        distance = self._calc_raw_series_distance()
        normalized_distance = self._calc_raw_series_normalized_distance(distance)
        shape_distance = sum([dtw_res.distance for dtw_res in self._dtw_results])
        shape_normalized_distance = sum([dtw_res.normalizedDistance for dtw_res in self._dtw_results])

        return ShapeDTWResults(
            distance, normalized_distance,
            shape_distance, shape_normalized_distance
        )

    def calc_shape_dtw(self,
                       subsequence_width: int,
                       shape_descriptor: ShapeDescriptor,
                       **kwargs):

        """
        Calculates multivariate, independent shape dtw and set proper attributes
        inside a given instance of the class (_dtw_results and _shape_dtw_results).
        In order to calculate shape dtw we need to get shape descriptors of
        each dimension of query and reference time series separately and construct
        distance matrices for them. Afterward, individual warping path is determined
        for each dimension separately. Finally, we calculate distances for each
        dimension and sum them up as a result.

        Parameters
        ---------------
        :param subsequence_width: width of subsequence
        :param shape_descriptor: shape descriptor
        :param kwargs: keyword arguments which will be passed to the 'dtw'
            function determining the warping paths

        Returns
        ---------------
        :return: self - an instance of the MultivariateShapeDTWIndependent with proper attributes
            assigned ('_dtw_results' and '_shape_dtw_results')
        """

        ts_x_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_x, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        ts_y_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_y, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        dist_matrices = ts_x_shape_descriptor.calc_distance_matrices(
            ts_y_shape_descriptor, dist_method=self._dist_method
        )

        dtw_results_list = [
            dtw(dist_mat.dist_matrix, step_pattern=self._step_pattern, **kwargs)
            for dist_mat in dist_matrices.distance_matrices_list
        ]

        self._dtw_results = dtw_results_list
        self._shape_dtw_results = self._calc_distances()

        return self

    def _get_index1(self) -> List[ndarray]:
        """
        Getter - get warping paths for query (x) series

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping paths for query series as a list of numpy arrays
        """
        if self._dtw_results is not None:
            return [dtw_res.index1 for dtw_res in self._dtw_results]
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_index2(self) -> List[ndarray]:
        """
        Getter - get warping paths for reference (y) series

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping paths for reference series as a list of numpy arrays
        """
        if self._dtw_results is not None:
            return [dtw_res.index2 for dtw_res in self._dtw_results]
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_index1s(self) -> List[ndarray]:
        """
        Getter - get warping paths for query (x) series with intermediate
            steps for multi-step patterns (like 'asymmetricP05()') excluded

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping paths for reference series as a list of numpy arrays, with
            intermediate steps excluded
        """
        if self._dtw_results is not None:
            return [dtw_res.index1s for dtw_res in self._dtw_results]
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_index2s(self) -> List[ndarray]:
        """
        Getter - get warping paths for reference (y) series with intermediate
            steps for multi-step patterns (like 'asymmetricP05()') excluded

        Raises
        ---------------
        :raises ShapeDTWNotCalculatedYet: ShapeDTW object was already created but
            shape dtw results were not calculated yet

        Returns
        ---------------
        :return: warping paths for reference series as a list of numpy arrays, with
            intermediate steps excluded
        """
        if self._dtw_results is not None:
            return [dtw_res.index2s for dtw_res in self._dtw_results]
        else:
            raise ShapeDTWNotCalculatedYet()

    index1 = property(_get_index1, ShapeDTW._set_index)
    index2 = property(_get_index2, ShapeDTW._set_index)
    index1s = property(_get_index1s, ShapeDTW._set_index)
    index2s = property(_get_index2s, ShapeDTW._set_index)

class ValuesGetter:

    """
    Class responsible for transforming input time series into
    format required by the shape_dtw function. For numpy array
    no preprocessing is required. For pandas series and data frames
    we need to retrieve values from the input object. What's more,
    for data frames containing only 1 column 'flatten' method
    must be applied.

    Attributes
    ---------------
    input_ts: object:
        input time series as a numpy array, pandas series or pandas
        data frame. For other types InputTimeSeriesUnsupportedType
        exception will be raised
    """

    _SUPPORTED_TYPES = [Series, DataFrame, ndarray]

    def __init__(self, input_ts: object):
        """
        Constructs a ValuesGetter object

        Raises
        ---------------
        :raises InputTimeSeriesUnsupportedType: input must be numpy array, pandas Series
            or pandas DataFrame

        Parameters
        ---------------
        :param input_ts: input time series as a numpy array, pandas series or pandas
            data frame. For other types InputTimeSeriesUnsupportedType exception
            will be raised
        """
        self.input_ts = input_ts
        if not self._type_is_supported():
            raise InputTimeSeriesUnsupportedType(
                self._SUPPORTED_TYPES,
                self.input_ts.__class__.__name__
            )

    def _type_is_supported(self) -> bool:
        """
        Checks whether type of input object is in the list of supported types

        Returns
        ---------------
        :return: bool - a result of test
        """
        input_ts_type = type(self.input_ts)
        return input_ts_type in self._SUPPORTED_TYPES

    def _is_numpy_array(self) -> bool:
        """
        Checks whether input object is a numpy array

        Returns
        ---------------
        :return: bool - a result of test
        """
        return isinstance(self.input_ts, ndarray)

    def _is_pandas_series(self) -> bool:
        """
        Checks whether input object is a pandas Series

        Returns
        ---------------
        :return: bool - a result of test
        """
        return isinstance(self.input_ts, Series)

    def _is_pandas_dataframe(self) -> bool:
        """
        Checks whether input object is a pandas DataFrame

        Returns
        ---------------
        :return: bool - a result of test
        """
        return isinstance(self.input_ts, DataFrame)

    def _is_flatten_needed(self) -> bool:
        """
        Checks whether 'flatten' method has to be applied (required
        for pandas data frames containing only 1 column)

        Returns
        ---------------
        :return: bool - a result of test
        """
        if self._is_pandas_dataframe():
            return True if len(self.input_ts.columns) == 1 else False
        return False

    def _flatten_if_needed(self) -> ndarray:
        """
        Applies 'flatten' method for pandas data frame if needed. Otherwise,
        returns raw data frame values.

        Returns
        ---------------
        :return: raw values of data frame in the form of numpy array (flattened
            if needed)
        """
        if self._is_flatten_needed():
            res = self.input_ts.values.flatten()
        else:
            res = self.input_ts.values
        return res

    def get_values(self) -> ndarray:
        """
        Transforms input time series object if needed. For numpy array
        input object is returned without any changes. For pandas Series
        and DataFrames values are retrieved in the form of numpy array.
        For data frames containing only 1 column flattening is applied.

        Returns
        ---------------
        :return: input time series, transformed if needed

        Examples
        --------
        >> import pandas as pd
        >> import numpy as np
        >> from shapedtw.shapedtw import ValuesGetter
        >>
        >> np_array = np.array([[1., 2., 3.], [4.5, 5.5, 3.2]])
        >> pd_series = pd.Series([1.2, 3.3, 4.5])
        >> pd_df_one_col = pd.DataFrame({"ts_x": [1.2, 2.2, 3.4]})
        >> pd_df_two_cols = pd.DataFrame({"ts_x": [1.2, 2.2, 3.4], "ts_y": [5.5, 4.4, 3.3]})
        >>
        >> vg_array = ValuesGetter(np_array)
        >> vg_series = ValuesGetter(pd_series)
        >> vg_df_one_col = ValuesGetter(pd_df_one_col)
        >> vg_df_two_cols = ValuesGetter(pd_df_two_cols)
        >>
        >> vg_array.get_values()
        array([[1. , 2. , 3. ],
               [4.5, 5.5, 3.2]])
        >> vg_series.get_values()
        array([1.2, 3.3, 4.5])
        >> vg_df_one_col.get_values()
        array([1.2, 2.2, 3.4])
        >> vg_df_two_cols.get_values()
        array([[1.2, 5.5],
               [2.2, 4.4],
               [3.4, 3.3]])
        """
        if self._is_numpy_array():
            res = self.input_ts
        elif self._is_pandas_series():
            res = self.input_ts.values
        elif self._is_pandas_dataframe():
            res = self._flatten_if_needed()

        return res


def shape_dtw(x, y, subsequence_width: int,
              shape_descriptor: ShapeDescriptor,
              step_pattern: str = "symmetric2",
              dist_method: str ="euclidean",
              multivariate_version: str = "dependent",
              **kwargs) -> ShapeDTW:
    """
    Calculates shape dtw for given query and reference time series.
    In case of multivariate time series parameter 'multivariate_version'
    allows to decide whether dependent or independent variant will be
    applied (dependent is a default).

    -----------

    For univariate time series the flow of algorithm looks exactly the
    same as described in the referenced paper by Zhao and Itti available
    here: https://arxiv.org/pdf/1606.01601.pdf.

    As a first step time series is transformed to the matrix of subsequences
    (temporal points and their neighbours). Width of subsequences can be
    determined with 'subsequence_width' parameter. For subsequence_width=1
    we've got a subsequence of total length 3 observations
    (temporal point + one preceding neighbour + one following
    neighbour). For subsequence_width=2 we've got a subsequence of total length
    5 observations (temporal point + two precedings neighbours + one following
    neighbours). In case of temporal points at the beginning and at the end
    of time series we pad first and, respectively, last temporal points.

    As a next step we calculate shape descriptor for each of the subsequences.
    Shape descriptor is determined by the 'shape_descriptor' parameter. It can
    be both simple and compound descriptor.

    Finally, distance matrix between shape descriptors is calculated and passed
    to the 'dtw' function from the dtw-python package (detailed information about
    the project: https://dynamictimewarping.github.io/). It determines a warping
    path, which - usually, according to Zhao and Itti paper - is more relevant than
    warping path determined based on raw time series values. Please keep in mind that
    distance calculated by 'dtw' function in such a case is a distance between shape
    descriptors of query and reference time series. It can be used - for example - to
    find the nearest neighbour of given time series. However, one might prefer to use
    distance between raw time series values, calculated using warping path determined
    by shape dtw algorithm. For this purpose also such distance is reconstructed in
    the postprocessing phase.

    -----------

    As for multivariate time series we developed a fusion of the shape dtw algorithm
    and the concept presented in the paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/.
    It means, that we can apply shape dtw algorithm for multivariate time series using
    both dependent and independent version of multivariate dtw.

    For multivariate dependent shape dtw there is one common warping paths
    for all time series dimensions. As a first step we calculate shape descriptors
    for all the dimensions individually and then determine single, common distance matrix
    using all the shape descriptors.

    For multivariate independent shape dtw distance matrices between shape descriptors
    are calculated independently for each time dimension; the same stands for warping paths.
    Afterward, distances for all dimensions are summed up -  as being said in the
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/ paper:
    ,,DTW Independet is the cumulative distances of all dimensions independently measured under DTW".

    Parameters
    ---------------
    :param x: query time series - numpy array, pandas series or pandas data frame
    :param y: reference time series - numpy array, pandas series or pandas data frame
    :param subsequence_width: int - width of the time series subsequences
    :param shape_descriptor: ShapeDescriptor object defining how shape descriptors
        will be calculated
    :param step_pattern: name of the step pattern describing the local warping steps
        allowed with their cost
    :param dist_method: name of distance method to apply
    :param multivariate_version: variant of multidimensional dtw algorithm - must be
        one of following: 'dependent' or 'independent'
    :param kwargs: additional keyword arguments which will be passed to the 'dtw'
        function, such as 'window_type' or 'open_begin'. For more details visit
        https://dynamictimewarping.github.io/

    Returns
    ---------------
    :return: ShapeDTW object containing results of calculation

    Examples
    --------
    >> import numpy as np
    >> from shapedtw.shapedtw import shape_dtw
    >> from shapedtw.shapeDescriptors import SlopeDescriptor, PAADescriptor, CompoundDescriptor
    >>
    # Univariate case
    >> np.random.seed(10)
    >> ts_x_uni = np.random.randn(50)
    >> ts_y_uni = np.random.randn(50)
    >> slope_descriptor = SlopeDescriptor(slope_window=2)
    >> paa_descriptor = PAADescriptor(piecewise_aggregation_window=2)
    >> compound_desc = CompoundDescriptor(
    >>      shape_descriptors = [slope_descriptor, paa_descriptor],
    >>      descriptors_weights = [2.0, 1.0]
    >> )
    >> univariate_results = shape_dtw(
    >>      x=ts_x_uni,
    >>      y=ts_y_uni,
    >>      subsequence_width=3,
    >>      shape_descriptor=compound_desc
    >> )
    >> print(round(univariate_results.distance, 2))
    75.42
    >> print(round(univariate_results.normalized_distance, 2))
    0.75
    >> print(round(univariate_results.shape_distance, 2))
    417.99
    >> print(round(univariate_results.shape_normalized_distance, 2))
    4.18
    >> print(univariate_results.index1s)
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
    23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46
    46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 46 47 48
    49 49 49 49]

    # Multivariate, dependent case
    >> np.random.seed(10)
    >> ts_x_multi = np.random.randn(50, 2)+10
    >> ts_y_multi = np.random.randn(50, 2)+10
    >> slope_descriptor = SlopeDescriptor(slope_window=2)
    >> paa_descriptor = PAADescriptor(piecewise_aggregation_window=2)
    >> compound_desc = CompoundDescriptor(
    >>      shape_descriptors = [slope_descriptor, paa_descriptor],
    >>      descriptors_weights = [2.0, 1.0]
    >> )
    >> multivariate_dependent_results = shape_dtw(
    >>      x=ts_x_multi,
    >>      y=ts_y_multi,
    >>      subsequence_width=3,
    >>      shape_descriptor=compound_desc,
    >>      multivariate_version="dependent"
    >> )
    >> print(round(multivariate_dependent_results.distance, 2))
    147.11
    >> print(round(multivariate_dependent_results.normalized_distance, 2))
    1.47
    >> print(round(multivariate_dependent_results.shape_distance, 2))
    704.26
    >> print(round(multivariate_dependent_results.shape_normalized_distance, 2))
    7.04
    >> print(multivariate_dependent_results.index1s)
    [ 0  1  2  3  4  5  5  5  5  6  7  8  9 10 10 10 10 11 12 13 14 14 14 15
    16 16 17 18 18 19 20 21 22 23 24 24 24 24 24 24 24 24 24 24 24 24 24 24
    25 26 27 28 29 30 31 31 32 33 33 33 34 35 36 37 38 39 40 41 42 43 44 45
    46 47 48 49]

    # Multivariate, independent case
    >> np.random.seed(10)
    >> ts_x_multi = np.random.randn(50, 2)+10
    >> ts_y_multi = np.random.randn(50, 2)+10
    >> slope_descriptor = SlopeDescriptor(slope_window=2)
    >> paa_descriptor = PAADescriptor(piecewise_aggregation_window=2)
    >> compound_desc = CompoundDescriptor(
    >>      shape_descriptors = [slope_descriptor, paa_descriptor],
    >>      descriptors_weights = [2.0, 1.0]
    >> )
    >> multivariate_independent_results = shape_dtw(
    >>      x=ts_x_multi,
    >>      y=ts_y_multi,
    >>      subsequence_width=3,
    >>      shape_descriptor=compound_desc,
    >>      multivariate_version="independent"
    >> )
    >> print(round(multivariate_independent_results.distance, 2))
    157.64
    >> print(round(multivariate_independent_results.normalized_distance, 2))
    1.58
    >> print(round(multivariate_independent_results.shape_distance, 2))
    822.61
    >> print(round(multivariate_independent_results.shape_normalized_distance, 2))
    8.23
    >> print(multivariate_independent_results.index1s)
    [
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 14, 15,
            16, 17, 18, 19, 19, 20, 21, 22, 23, 24, 24, 25, 26, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 35, 35, 35, 36, 37, 38, 39, 40, 41, 42, 43,
            44, 45, 45, 45, 45, 45, 46, 47, 48, 49, 49, 49]),
        array([ 0,  1,  2,  3,  4,  5,  6,  6,  6,  7,  8,  9, 10, 11, 12, 13, 14,
            15, 16, 16, 16, 16, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18,
            18, 18, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
            33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49])
    ]

    References
    ----------
    1. Sakoe, H.; Chiba, S., *Dynamic programming algorithm optimization for
       spoken word recognition*, Acoustics, Speech, and Signal Processing,
       IEEE Transactions on , vol.26, no.1, pp. 43-49, Feb 1978.
       http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=1163055
    2. Itti, L.; Zhao, J., *shapeDTW: shape Dynamic Time Warping,*
       Pattern Recognition, Volume 74, pp. 171-184, Feb 2018.
       https://arxiv.org/pdf/1606.01601.pdf
    3. Hu, B; Jin, H.; Keogh, E.; Shokoohi-Yekta, M., *Generalizing DTW to
       the multi-dimensional case requires an adaptive approach*,
       Data Mining and Knowledge Discovery, vol. 31, pp. 1–31, 2017.
       https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5668684/
    """
    x = ValuesGetter(x).get_values()
    y = ValuesGetter(y).get_values()

    if not Utils.number_of_dimensions_equal(x, y):
        raise IncompatibleDimensionality(x, y)

    if not Utils.number_of_series_equal(x, y):
        raise IncompatibleSubseriesNumber(x, y)

    ts_x_shape = Utils.get_number_of_dimensions(x)

    if ts_x_shape == 1:
        shape_dtw_obj = UnivariateShapeDTW(
            ts_x=x, ts_y=y,
            step_pattern=step_pattern,
            dist_method=dist_method
        )
    elif multivariate_version == "dependent":
        shape_dtw_obj = MultivariateShapeDTWDependent(
            ts_x=x, ts_y=y,
            step_pattern=step_pattern,
            dist_method=dist_method
        )
    elif multivariate_version == "independent":
        shape_dtw_obj = MultivariateShapeDTWIndependent(
            ts_x=x, ts_y=y,
            step_pattern=step_pattern,
            dist_method=dist_method
        )
    else:
        raise WrongMultivariateVersionSpecified(multivariate_version)

    shape_dtw_results = shape_dtw_obj.calc_shape_dtw(
        subsequence_width=subsequence_width,
        shape_descriptor=shape_descriptor,
        **kwargs
    )

    return shape_dtw_results
