from __future__ import annotations

import operator



from numpy import ndarray
from shapedtw.exceptions import *
from scipy.spatial.distance import cdist
from typing import List
from abc import ABC, abstractmethod
from shapedtw.shapeDescriptors import ShapeDescriptor
from functools import reduce
from shapedtw.utils import Utils


class SubsequenceBuilder(ABC):

    """
    Abstract class representing subsequence builder. It enforces all child classes
    to implement abstract method 'transform_time_series_to_subsequences', which
    transforms univariate or multivariate time series to its subsequences
    representation. Subsequences represent neighbourhood of given temporal point of
    a time series and afterward are used to calculate its shape descriptor.
    """


    @abstractmethod
    def transform_time_series_to_subsequences(self):
        """
        Transform input time series to its subsequences representation
        """
        pass


class UnivariateSubsequenceBuilder(SubsequenceBuilder):

    """
    Univariate time series builder.

    This class is used for the purpose of transforming univariate time
    series into a set of subsequences. Subsequences are represented by
    two-dimensional array, where number of rows is equal to the time series
    length and number of column is equal to the specified subsequence length,
    calculated as subsequence_width*2 + 1.

    Attributes
    ---------------
    time_series: ndarray:
        input time series as numpy array
    subsequence_width: int:
        width of the output subsequences

    Examples
    --------
    >> from shapedtw.preprocessing import UnivariateSubsequenceBuilder
    >> import numpy as np
    >> ts_x = np.array([1, 2, 3])
    >> usb = UnivariateSubsequenceBuilder(time_series=ts_x, subsequence_width=3)
    >> res = usb.transform_time_series_to_subsequences()
    >> print(res.subsequences)
    [[1 1 1 1 2 3 3]
    [1 1 1 2 3 3 3]
    [1 1 2 3 3 3 3]]
    """

    def __init__(self, time_series: ndarray, subsequence_width: int):
        """
        Constructs a UnivariateSubsequenceBuilder object

        Parameters
        ---------------
        :param time_series: input time series (1d numpy array)
        :param subsequence_width: width of a single subsequence

        Raises
        ---------------
        :raise NegativeSubsequenceWidth: subsequence width must be integer equal
            to or greater than 0
        """
        if subsequence_width < 0:
            raise NegativeSubsequenceWidth()

        self.subsequence_width = subsequence_width
        self.padded_time_series = self._get_padded_time_series(time_series)
        self.ts_length = len(time_series)

    def _get_padded_time_series(self, input_time_series: ndarray) -> ndarray:
        """
        Get time series padded with its edge values (first and last point).
        In order to make subsequences and shape descriptors well-defined for every
        temporal point we need to extend time series at the beginning and at the
        end.

        Parameters
        ---------------
        :param input_time_series: input time series as a numpy array

        Returns
        ---------------
        :returns: padded time series as a numpy array
        """
        padded_time_series = np.pad(
            input_time_series,
            self.subsequence_width,
            mode = "edge"
        )

        return padded_time_series

    def _get_central_indices(self) -> ndarray:
        """
        Get indices of temporal points for which the subsequences will be
        retrieved. It is equivalent of indices of origin time series,
        before padding was applied.

        Returns
        ---------------
        :returns: numpy array - indices of padded time series for which
            subsequences will be retrieved
        """
        central_indices = np.arange(
            start=self.subsequence_width,
            stop=self.ts_length+self.subsequence_width
        )

        return central_indices

    def _get_single_subsequence(self, central_index: int) -> ndarray:
        """
        Get subsequence for given index of padded time series

        Parameters
        ---------------
        :param central_index: int - index of padded time series for which
            subsequence will be retrieved

        Returns
        ---------------
        :return: subsequence for given index as a numpy array

        Examples
        --------
        >> from shapedtw.preprocessing import UnivariateSubsequenceBuilder
        >> import numpy as np
        >> ts_x = np.array([1, 2, 3])
        >> usb = UnivariateSubsequenceBuilder(time_series=ts_x, subsequence_width=3)
        >> res = usb._get_single_subsequence(3)
        >> print(res)
        array([1, 1, 1, 1, 2, 3, 3])
        """
        current_indices = np.arange(
            start=central_index-self.subsequence_width,
            stop=central_index+self.subsequence_width+1
        )

        return self.padded_time_series[current_indices]

    def transform_time_series_to_subsequences(self) -> UnivariateSeriesSubsequences:
        """
        Transforms univariate time series to the array of its subsequences

        Returns
        ---------------
        :return: UnivariateSeriesSubsequences object. It contains array of time series
            subsequences and origin time series (before padding was applied).

        Examples
        --------
        >> from shapedtw.preprocessing import UnivariateSubsequenceBuilder
        >> import numpy as np
        >> ts_x = np.array([1, 2, 3])
        >> usb = UnivariateSubsequenceBuilder(time_series=ts_x, subsequence_width=3)
        >> res = usb.transform_time_series_to_subsequences()
        >>
        >> print(res.subsequences)
        [[1 1 1 1 2 3 3]
        [1 1 1 2 3 3 3]
        [1 1 2 3 3 3 3]]
        >>
        >> print(res.origin_ts)
        array([1, 2, 3])
        """
        central_indices = self._get_central_indices()
        subsequences_list = [self._get_single_subsequence(central_index) for central_index in central_indices]
        subsequences_array = np.vstack(subsequences_list)
        return UnivariateSeriesSubsequences(subsequences_array, origin_ts=self.padded_time_series[central_indices])


class MultivariateSubsequenceBuilder(SubsequenceBuilder):

    """
    Multivariate time series builder.

    This class is used for the purpose of transforming multivariate time
    series into a set of subsequences. Subsequences are represented by
    MultivariateSeriesSubsequences, which contains a list of
    UnivariateSeriesSubsequences - one for each of time series
    dimensions.

    Attributes
    ---------------
    time_series: ndarray:
        input time series as a 2d numpy array
    subsequence_width: int:
        width of the output subsequences

    Examples
    --------
    >> from shapedtw.preprocessing import MultivariateSubsequenceBuilder
    >> import numpy as np
    >> ts_x = np.array(
    >>  [[1, 2],
    >>   [3, 4],
    >>   [5, 6]]
    >>  )
    >> msb = MultivariateSubsequenceBuilder(time_series=ts_x, subsequence_width=3)
    >> res = msb.transform_time_series_to_subsequences()
    >> for uss in res.subsequences_list:
    >>      print(uss.subsequences, '\n')
    [[1 1 1 1 3 5 5]
    [1 1 1 3 5 5 5]
    [1 1 3 5 5 5 5]]

    [[2 2 2 2 4 6 6]
    [2 2 2 4 6 6 6]
    [2 2 4 6 6 6 6]]
    """

    def __init__(self, time_series: ndarray, subsequence_width: int):
        """
        Constructs a MultivariateSubsequenceBuilder object

        Parameters
        ---------------
        :param time_series: input time series (2d numpy array)
        :param subsequence_width: width of a single subsequence

        Raises
        ---------------
        :raise NegativeSubsequenceWidth: subsequence width must be integer equal
            to or greater than 0
        """
        if subsequence_width < 0:
            raise NegativeSubsequenceWidth()

        self.time_series = time_series
        self.subsequence_width = subsequence_width
        self.dimensions_number = time_series.shape[1]

    def transform_time_series_to_subsequences(self) -> MultivariateSeriesSubsequences:
        """
        Transforms multivariate time series to the MultivariateSeriesSubsequences object

        Returns
        ---------------
        :return: MultivariateSeriesSubsequences object. It contains a list of
            UnivariateSeriesSubsequences (one per each of a time series dimension)
            and origin time series.

        Examples
        --------
        >> from shapedtw.preprocessing import MultivariateSubsequenceBuilder
        >> import numpy as np
        >> ts_x = np.array(
        >>  [[1, 2],
        >>   [3, 4],
        >>   [5, 6]]
        >>  )
        >> msb = MultivariateSubsequenceBuilder(time_series=ts_x, subsequence_width=3)
        >> res = msb.transform_time_series_to_subsequences()
        >>
        >> for uss in res.subsequences_list:
        >>      print(uss.subsequences, '\n')
        [[1 1 1 1 3 5 5]
        [1 1 1 3 5 5 5]
        [1 1 3 5 5 5 5]]

        [[2 2 2 2 4 6 6]
        [2 2 2 4 6 6 6]
        [2 2 4 6 6 6 6]]
        """
        sub_builders = [UnivariateSubsequenceBuilder(self.time_series[:, i], self.subsequence_width)
                        for i in range(self.dimensions_number)]
        subsequences = [sub_builder.transform_time_series_to_subsequences()
                        for sub_builder in sub_builders]
        return MultivariateSeriesSubsequences(subsequences, self.time_series)


class Subsequences(ABC):

    """
    Abstract class representing a subsequences objects, which is a set of
    time series temporal points with its neighbours. It enforces child
    classes to implement 'get_shape_descriptors' method, which takes
    ShapeDescriptor object as an argument and returns shape descriptors
    of given subsequences.
    """

    @abstractmethod
    def get_shape_descriptors(self, shape_descriptor: ShapeDescriptor) -> object:
        """
        Get shape descriptors for subsequences stored as Subsequences class
        attribute.

        Parameters
        ---------------
        :param shape_descriptor: instance of ShapeDescriptor child class

        Returns
        ---------------
        :returns: object containing shape descriptors of given subsequences
        """
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

    @staticmethod
    def _calc_sum_of_distance_matrices_euclidean(
            univariate_dist_matrices: List[UnivariateSeriesDistanceMatrix]
    ) -> np.ndarray:
        distance_matrices_list = [uni_mat.dist_matrix ** 2 for uni_mat in univariate_dist_matrices]
        distance_matrix = np.sqrt(reduce(operator.add, distance_matrices_list))
        return distance_matrix

    @staticmethod
    def _calc_sum_of_distance_matrices_non_euclidean(
            univariate_dist_matrices: List[UnivariateSeriesDistanceMatrix]
    ) -> np.ndarray:
        distance_matrices_list = [uni_mat.dist_matrix for uni_mat in univariate_dist_matrices]
        distance_matrix = reduce(operator.add, distance_matrices_list)
        return distance_matrix

    def calc_summed_distance_matrix(self, series_y_descriptor: MultivariateSeriesShapeDescriptors,
                                    dist_method: str = "euclidean") -> MultivariateDistanceMatrixDependent:

        univariate_dist_matrices = self.calc_distance_matrices(
            series_y_descriptor,
            dist_method).distance_matrices_list

        if dist_method == "euclidean":
            distance_matrix = self._calc_sum_of_distance_matrices_euclidean(univariate_dist_matrices)
        else:
            distance_matrix = self._calc_sum_of_distance_matrices_non_euclidean(univariate_dist_matrices)

        return MultivariateDistanceMatrixDependent(distance_matrix, self.origin_ts, series_y_descriptor.origin_ts)


class DistanceMatrixCalculator:

    def __init__(self, ts_x: ndarray, ts_y: ndarray, method: str = "euclidean"):
        self.ts_x = ts_x
        self.ts_y = ts_y
        self.method = method

    def _input_ts_empty(self):
        return (np.size(self.ts_x) == 0) | (np.size(self.ts_x) == 0)

    def _two_dim_at_most(self):
        return (len(self.ts_x.shape) < 3) & (len(self.ts_y.shape) < 3)

    def _series_shape_match(self):
        return len(self.ts_x.shape) == len(self.ts_y.shape)

    def _series_are_univariate(self):
        return (len(self.ts_x.shape) == 1) & (len(self.ts_y.shape) == 1)

    def _series_dimensions_match(self):
        return self.ts_x.shape[1] == self.ts_y.shape[1]

    def _verify_dimensions(self):

        if self._input_ts_empty():
            raise DimensionError("Empty arrays are not allowed.")

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
    def __init__(self, distance_matrices_list: List[UnivariateSeriesDistanceMatrix], ts_x, ts_y):
        self.distance_matrices_list = distance_matrices_list
        self.ts_x = ts_x
        self.ts_y = ts_y


class MultivariateDistanceMatrixDependent:
    def __init__(self, distance_matrix: ndarray, ts_x, ts_y):
        self.distance_matrix = distance_matrix
        self.ts_x = ts_x
        self.ts_y = ts_y
