from __future__ import annotations

import operator

import numpy as np
from numpy import ndarray
from shapedtw.exceptions import *
from scipy.spatial.distance import cdist
from typing import List
from abc import ABC, abstractmethod
from shapedtw.shapeDescriptors import ShapeDescriptor
from functools import reduce
from shapedtw.utils import Utils


class Padder:

    """
    Auxiliary class, storing a set of methods which can be
    used to get subsequences padded with first or last values of
    time series. Padding is applied during the process of transforming
    time series into a set of subsequences. It has to be done in order
    to make shape descriptors for starting and ending points of time
    series well-defined.
    """

    def __init__(self, time_series: ndarray, subsequence_width: int):
        self.time_series = time_series
        self.ts_length = len(time_series)
        self.subsequence_width = subsequence_width

    def _get_pad_number(self, ts_index: int, pad_side: str) -> int:
        if pad_side == "left":
            n_padded = self.subsequence_width - ts_index
        elif pad_side == "right":
            n_padded = (ts_index+1)-self.ts_length+self.subsequence_width
        else:
            raise WrongPadType("Pad type must bo one of ('left', 'right')")

        return n_padded

    def pad_left(self, ts_index: int) -> ndarray:
        n_padded = self._get_pad_number(ts_index, "left")
        last_ind = ts_index+self.subsequence_width+1
        ts_first_observation = self.time_series[0]
        first_observation_repeated = np.repeat(ts_first_observation, n_padded)
        subsequence = np.concatenate(
            (first_observation_repeated, self.time_series[0:last_ind])
        )

        return subsequence

    def pad_right(self, ts_index: int) -> ndarray:
        ts_len = len(self.time_series)
        n_padded = self._get_pad_number(ts_index, "right")
        first_ind = ts_index-self.subsequence_width
        ts_last_observation = self.time_series[ts_len-1]
        ts_last_repeated = np.repeat(ts_last_observation, n_padded)
        subsequence = np.concatenate(
            (self.time_series[first_ind:], ts_last_repeated)
        )

        return subsequence

    def pad_both_side(self, ts_index: int) -> ndarray:
        ts_len = len(self.time_series)
        n_padded_left = self._get_pad_number(ts_index, "left")
        n_padded_right = self._get_pad_number(ts_index, "right")
        ts_first_observation = self.time_series[0]
        ts_last_observation = self.time_series[ts_len-1]
        ts_first_observation_repeated = np.repeat(ts_first_observation, n_padded_left)
        ts_last_observation_repeated = np.repeat(ts_last_observation, n_padded_right)
        subsequence = np.concatenate(
            (
                ts_first_observation_repeated,
                self.time_series,
                ts_last_observation_repeated
            )
        )

        return subsequence


class SubsequenceBuilder(ABC):

    @abstractmethod
    def transform_time_series_to_subsequences(self):
        pass


class UnivariateSubsequenceBuilder(SubsequenceBuilder):

    """
    This class is used for the purpose of transforming univariate time
    series into a set of subsequences. It is represented by two-dimensional
    array, where number of rows is equal to the time series length and number
    of column is equal to the specified subsequence length.
    """

    def __init__(self, time_series: ndarray, subsequence_width: int):

        if subsequence_width < 0:
            raise NegativeSubsequenceWidth()

        self.subsequence_width = subsequence_width
        self.time_series = time_series
        self.ts_length = len(time_series)
        self.padder = Padder(time_series, subsequence_width)

    def _apply_left_padding(self, ts_index: int) -> bool:
        subsequence_beginning = ts_index - self.subsequence_width
        return subsequence_beginning < 0

    def _apply_right_padding(self, ts_index: int) -> bool:
        subsequence_ending = ts_index + self.subsequence_width
        return (subsequence_ending+1) > self.ts_length

    def _apply_padding_both_sides(self, ts_index: int) -> bool:
        return self._apply_left_padding(ts_index) & self._apply_right_padding(ts_index)

    def _get_subsequence_without_padding(self, ts_index: int) -> ndarray:
        subsequence_start = ts_index - self.subsequence_width
        subsequence_end = ts_index + self.subsequence_width + 1
        return self.time_series[subsequence_start:subsequence_end]

    def _get_single_subsequence(self, ts_index: int) -> ndarray:
        if self._apply_padding_both_sides(ts_index):
            subsequence = self.padder.pad_both_side(ts_index)
        elif self._apply_left_padding(ts_index):
            subsequence = self.padder.pad_left(ts_index)
        elif self._apply_right_padding(ts_index):
            subsequence = self.padder.pad_right(ts_index)
        else:
            subsequence = self._get_subsequence_without_padding(ts_index)

        return subsequence

    def transform_time_series_to_subsequences(self) -> UnivariateSeriesSubsequences:
        indices = np.arange(start=0, stop=self.ts_length)
        subsequences_list = [self._get_single_subsequence(ts_index) for ts_index in indices]
        subsequences_array = np.vstack(subsequences_list)
        return UnivariateSeriesSubsequences(subsequences_array, origin_ts=self.time_series)


class MultivariateSubsequenceBuilder(SubsequenceBuilder):

    """
    Subsequence builder for multivariate time series
    """

    def __init__(self, time_series: ndarray, subsequence_width):
        self.time_series = time_series
        self.subsequence_width = subsequence_width
        self.dimensions_number = time_series.shape[1]

    def transform_time_series_to_subsequences(self) -> MultivariateSeriesSubsequences:
        sub_builders = [UnivariateSubsequenceBuilder(self.time_series[:, i], self.subsequence_width)
                        for i in range(self.dimensions_number)]
        subsequences = [sub_builder.transform_time_series_to_subsequences()
                        for sub_builder in sub_builders]
        return MultivariateSeriesSubsequences(subsequences, self.time_series)


class Subsequences(ABC):

    @abstractmethod
    def get_shape_descriptors(self, shape_descriptor: ShapeDescriptor):
        pass


class UnivariateSeriesSubsequences(Subsequences):

    def __init__(self, subsequences_array: ndarray, origin_ts: ndarray):
        self.subsequences = subsequences_array
        self.origin_ts = origin_ts

    def get_shape_descriptors(self, shape_descriptor: ShapeDescriptor) -> UnivariateSeriesShapeDescriptors:
        shape_descriptors = np.array([
            shape_descriptor.get_shape_descriptor(subsequence) for
            subsequence in self.subsequences
        ])

        return UnivariateSeriesShapeDescriptors(shape_descriptors, self.origin_ts)


class MultivariateSeriesSubsequences(Subsequences):

    def __init__(self, subsequences_list: List[UnivariateSeriesSubsequences], origin_ts: ndarray):
        self.subsequences_list = subsequences_list
        self.origin_ts = origin_ts

    def get_shape_descriptors(self, shape_descriptor: ShapeDescriptor) -> MultivariateSeriesShapeDescriptors:
        shape_descriptor_list = [univariate_subsequences.get_shape_descriptors(shape_descriptor)
                                 for univariate_subsequences in self.subsequences_list]

        return MultivariateSeriesShapeDescriptors(shape_descriptor_list, self.origin_ts)


class UnivariateSeriesShapeDescriptors:

    def __init__(self, descriptors_array: ndarray, origin_ts: ndarray):

        if self._array_is_empty(descriptors_array):
            raise EmptyShapeDescriptorsArray()
        elif not self._input_ts_descriptor_array_compatible(descriptors_array, origin_ts):
            input_ts_len = origin_ts.shape[0]
            shape_descriptor_array_nrow = descriptors_array.shape[0]
            raise UnivariateOriginTSShapeDescriptorsIncompatibility(
                input_ts_len, shape_descriptor_array_nrow
            )
        elif self._check_dimensions_number(descriptors_array, 1):
            descriptors_array = np.atleast_2d(descriptors_array).T
        elif not self._check_dimensions_number(descriptors_array, 2):
            n_dims = len(descriptors_array.shape)
            raise TooManyDimensionsArray(self, n_dims)

        self.shape_descriptors_array = descriptors_array
        self.origin_ts = origin_ts

    @staticmethod
    def _check_dimensions_number(descriptor_array: ndarray, n_dim: int) -> bool:
        return len(descriptor_array.shape) == n_dim

    @staticmethod
    def _array_is_empty(descriptor_array: ndarray):
        return np.size(descriptor_array) == 0

    @staticmethod
    def _input_ts_descriptor_array_compatible(descriptor_array: ndarray, origin_ts: ndarray):
        return origin_ts.shape[0] == descriptor_array.shape[0]

    def calc_distance_matrix(self, series_y_descriptor: UnivariateSeriesShapeDescriptors,
                             dist_method: str = "euclidean") -> UnivariateSeriesDistanceMatrix:

        if not Utils.are_objects_of_same_classes(self, series_y_descriptor):
            raise ObjectOfWrongClass(
                actual_cls=series_y_descriptor.__class__,
                expected_cls=self.__class__
            )

        distance_matrix = DistanceMatrixCalculator(
            self.shape_descriptors_array,
            series_y_descriptor.shape_descriptors_array,
            method=dist_method
        ).calc_distance_matrix()

        return UnivariateSeriesDistanceMatrix(distance_matrix, self.origin_ts, series_y_descriptor.origin_ts)


class MultivariateSeriesShapeDescriptors:

    def __init__(self, descriptors_list: List[UnivariateSeriesShapeDescriptors], origin_ts: ndarray):

        if self._is_one_dim_ts(origin_ts):
            origin_ts = np.atleast_2d(origin_ts)

        self.descriptors_list = descriptors_list
        self.origin_ts = origin_ts

        if not self._input_ts_descriptor_dimensions_compatible():
            origin_ts_dim = self.origin_ts.shape[1]
            descriptors_list_length = len(self)
            raise MultivariateOriginTSShapeDescriptorsDimIncompatibility(
                origin_ts_dim=origin_ts_dim,
                shape_descriptor_list_length=descriptors_list_length
            )

        if not self._input_ts_descriptors_length_compatible():
            ts_len = self.origin_ts.shape[0]
            shape_descriptors_lengths = [
                uni_sd.shape_descriptors_array.shape[0]
                for uni_sd in self.descriptors_list
            ]

            raise MultivariateOriginTSShapeDescriptorsLengthIncompatibility(
                origin_ts_length=ts_len,
                shape_descriptor_lengths=shape_descriptors_lengths
            )


    def __len__(self):
        return len(self.descriptors_list)

    @staticmethod
    def _is_one_dim_ts(origin_ts):
        return len(origin_ts.shape) == 1

    def _input_ts_descriptor_dimensions_compatible(self):
        return len(self) == self.origin_ts.shape[1]

    def _input_ts_descriptors_length_compatible(self):
        ts_len = self.origin_ts.shape[0]
        return all(
            [uni_sd.shape_descriptors_array.shape[0] == ts_len
             for uni_sd in self.descriptors_list]
        )

    def _verify_dimension_compatibility(self, other: MultivariateSeriesShapeDescriptors) -> None:
        ts_x_dim = len(self)
        ts_y_dim = len(other)
        if ts_x_dim != ts_y_dim:
            raise MultivariateSeriesShapeDescriptorsIncompatibility(ts_x_dim, ts_y_dim)

    def calc_distance_matrices(self, series_y_descriptor: MultivariateSeriesShapeDescriptors,
                               dist_method: str = "euclidean") -> MultivariateDistanceMatrixIndependent:

        if not Utils.are_objects_of_same_classes(self, series_y_descriptor):
            raise ObjectOfWrongClass(
                actual_cls=series_y_descriptor.__class__,
                expected_cls=self.__class__
            )

        self._verify_dimension_compatibility(series_y_descriptor)

        distance_matrices_list = [ts_x_descriptor.calc_distance_matrix(ts_y_descriptor, dist_method)
                                  for (ts_x_descriptor, ts_y_descriptor)
                                  in zip(self.descriptors_list, series_y_descriptor.descriptors_list)]

        return MultivariateDistanceMatrixIndependent(
            distance_matrices_list,
            self.origin_ts,
            series_y_descriptor.origin_ts
        )

    def calc_summed_distance_matrix(self, series_y_descriptor: MultivariateSeriesShapeDescriptors,
                                    dist_method: str = "euclidean") -> MultivariateDistanceMatrixDependent:

        univariate_dist_matrices = self.calc_distance_matrices(
            series_y_descriptor,
            dist_method).distance_matrices_list
        if dist_method == "euclidean":
            distance_matrices_list = [uni_mat.dist_matrix**2 for uni_mat in univariate_dist_matrices]
            distance_matrix = np.sqrt(reduce(operator.add, distance_matrices_list))
        else:
            distance_matrices_list = [uni_mat.dist_matrix for uni_mat in univariate_dist_matrices]
            distance_matrix = reduce(operator.add, distance_matrices_list)

        return MultivariateDistanceMatrixDependent(distance_matrix, self.origin_ts, series_y_descriptor.origin_ts)


class DistanceMatrixCalculator:

    def __init__(self, ts_x: ndarray, ts_y: ndarray, method: str = "euclidean"):
        self.ts_x = ts_x
        self.ts_y = ts_y
        self.method = method

    def _two_dim_at_most(self):
        return (len(self.ts_x.shape) < 3) & (len(self.ts_y.shape) < 3)

    def _series_shape_match(self):
        return len(self.ts_x.shape) == len(self.ts_y.shape)

    def _series_are_univariate(self):
        return (len(self.ts_x.shape) == 1) & (len(self.ts_y.shape) == 1)

    def _series_dimensions_match(self):
        return self.ts_x.shape[1] == self.ts_y.shape[1]

    def _verify_dimensions(self):

        if not self._two_dim_at_most():
            raise DimensionError("Only arrays of 1 and 2 dimensions are supported")

        if not self._series_shape_match():
            raise DimensionError("Number of time series dimensions doesn't match")

        if not self._series_are_univariate():
            if not self._series_dimensions_match():
                raise DimensionError("Number of time series columns doesn't match")

    def _convert_one_dimension_series(self):
        self.ts_x = np.atleast_2d(self.ts_x).T
        self.ts_y = np.atleast_2d(self.ts_y).T

    def calc_distance_matrix(self):
        self._verify_dimensions()
        if self._series_are_univariate():
            self._convert_one_dimension_series()

        dist_matrix = cdist(self.ts_x, self.ts_y, metric=self.method)

        return dist_matrix


class UnivariateSeriesDistanceMatrix:

    def __init__(self, dist_matrix: np.ndarray, ts_x: ndarray, ts_y: ndarray):
        self.dist_matrix = dist_matrix
        self.ts_x = ts_x
        self.ts_y = ts_y


class MultivariateDistanceMatrixIndependent:
    def __init__(self, distance_matrices_list: List[ndarray], ts_x, ts_y):
        self.distance_matrices_list = distance_matrices_list
        self.ts_x = ts_x
        self.ts_y = ts_y


class MultivariateDistanceMatrixDependent:
    def __init__(self, distance_matrix: ndarray, ts_x, ts_y):
        self.distance_matrix = distance_matrix
        self.ts_x = ts_x
        self.ts_y = ts_y
