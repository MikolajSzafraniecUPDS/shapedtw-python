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
        retrieved. It is equivalent of indices of the original time series,
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
            subsequences and the original time series (before padding was applied).

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

    def __init__(self, time_series: ndarray, get_shape_descriptors: int):
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
            and the original time series.

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

    """
    Class representing univariate time series split into a set of
    subsequences, required for calculating shape descriptors. It contains
    an array of subsequences and the original time series. Moreover, it provides
    a method which allows to calculate shape descriptors based on given
    ShapeDescriptor's object.

    Attributes
    ---------------
    subsequences_array: ndarray:
        2d numpy array containing subsequences
    origin_ts: ndarray:
        origin time series as 1d array
    """

    def __init__(self, subsequences_array: ndarray, origin_ts: ndarray):
        """
        Constructs a UnivariateSeriesSubsequences object

        Parameters
        ---------------
        :param subsequences_array: 2d numpy array - every row of
            it represents single subsequence of time series
        :param origin_ts: original time series as 1d array
        """
        self.subsequences = subsequences_array
        self.origin_ts = origin_ts

    def get_shape_descriptors(self, shape_descriptor: ShapeDescriptor) -> UnivariateSeriesShapeDescriptors:
        """
        Calculates shape descriptors for all subsequences

        Parameters
        ---------------
        :param shape_descriptor: shape descriptor object - it must be an instance of
            ShapeDescriptor child class

        Raises
        ---------------
        :raises NotShapeDescriptor: provided shape_descriptor object is not an
            instance of ShapeDescriptor's child class

        Returns
        ---------------
        :returns: UnivariateSeriesShapeDescriptors object, representing shape
            descriptors of the original time series

        Examples
        --------
        >> from shapedtw.preprocessing import UnivariateSeriesSubsequences
        >> from shapedtw.shapeDescriptors import PAADescriptor
        >> import numpy as np
        >> origin_ts = np.array([1, 2, 3, 4])
        >> subsequences_array = np.array([
        >>      [1, 1, 1, 2, 3],
        >>      [1, 1, 2, 3, 4],
        >>      [1, 2, 3, 4, 4],
        >>      [2, 3, 4, 4, 4]
        >> ])
        >> uss = UnivariateSeriesSubsequences(subsequences_array, origin_ts)
        >> paa_descriptor = PAADescriptor(2)
        >> res = uss.get_shape_descriptors(paa_descriptor)
        >> print(res.shape_descriptors_array)
        [[1.  1.5 3. ]
         [1.  2.5 4. ]
         [1.5 3.5 4. ]
         [2.5 4.  4. ]]
        """

        if not isinstance(shape_descriptor, ShapeDescriptor):
            raise NotShapeDescriptor(shape_descriptor)

        shape_descriptors = np.array([
            shape_descriptor.get_shape_descriptor(subsequence) for
            subsequence in self.subsequences
        ])

        return UnivariateSeriesShapeDescriptors(shape_descriptors, self.origin_ts)


class MultivariateSeriesSubsequences(Subsequences):
    """
    Class representing multivariate time series split into a set of
    subsequences, required for calculating shape descriptors. It contains
    a list of UnivariateSeriesSubsequences objects (one per each dimension of
    time series) and the original time series. Moreover, it provides
    a method which allows to calculate shape descriptors based on given
    ShapeDescriptor's object.

    Attributes
    ---------------
    subsequences_list: List[UnivariateSeriesSubsequences]:
        List containing a set of UnivariateSeriesSubsequences objects, one per
        each time series dimension
    origin_ts: ndarray:
        original time series as 2d array
    """

    def __init__(self, subsequences_list: List[UnivariateSeriesSubsequences], origin_ts: ndarray):
        """
        Constructs a MultivariateSeriesSubsequences object

        Parameters
        ---------------
        :param subsequences_list: List containing a set of UnivariateSeriesSubsequences objects,
            one per ach time series dimension
        :param origin_ts: original time series as 2d array
        """
        self.subsequences_list = subsequences_list
        self.origin_ts = origin_ts

    def get_shape_descriptors(self, shape_descriptor: ShapeDescriptor) -> MultivariateSeriesShapeDescriptors:
        """
        Calculates shape descriptors for subsequences of all time series
        dimensions

        Parameters
        ---------------
        :param shape_descriptor: shape descriptor object - it must be an instance of
            ShapeDescriptor child class

        Raises
        ---------------
        :raises NotShapeDescriptor: provided shape_descriptor object is not an
            instance of ShapeDescriptor's child class

        :returns: MultivariateSeriesShapeDescriptors object, representing shape
            descriptors of the original multivariate time series

        Examples
        --------
        >> from shapedtw.preprocessing import MultivariateSubsequenceBuilder
        >> from shapedtw.shapeDescriptors import PAADescriptor
        >> import numpy as np
        >>
        >> input_ts = np.array([
        >>  [1, 2],
        >>  [1.5, 5],
        >>  [4, 3]
        >> ])
        >> paa_descriptor = PAADescriptor(2)
        >> multivariate_series_subsequences = MultivariateSubsequenceBuilder(
        >>  input_ts, 2
        >>  ).transform_time_series_to_subsequences()
        >> res = multivariate_series_subsequences.get_shape_descriptors(paa_descriptor)
        >> for one_dim_descriptor in res.descriptors_list:
        >>      print(one_dim_descriptor.shape_descriptors_array)
        [[1.   1.25 4.  ]
         [1.   2.75 4.  ]
         [1.25 4.   4.  ]]
        [[2.  3.5 3. ]
         [2.  4.  3. ]
         [3.5 3.  3. ]]
        """
        if not isinstance(shape_descriptor, ShapeDescriptor):
            raise NotShapeDescriptor(shape_descriptor)

        shape_descriptor_list = [univariate_subsequences.get_shape_descriptors(shape_descriptor)
                                 for univariate_subsequences in self.subsequences_list]

        return MultivariateSeriesShapeDescriptors(shape_descriptor_list, self.origin_ts)


class UnivariateSeriesShapeDescriptors:

    """
    Class representing set of shape descriptors for univariate time series. It
    provides a set of methods allowing to verify consistency of provided
    descriptors and calculate distance matrix between descriptors for two
    instances of UnivariateSeriesShapeDescriptors class.

    Attributes
    ---------------
    descriptors_array: ndarray:
        Shape descriptors in the form of 2d numpy array. Each row
        represents shape descriptor of a single temporal point
    origin ts: ndarray:
        original time series
    """

    def __init__(self, descriptors_array: ndarray, origin_ts: ndarray):
        """
        Constructs a UnivariateSeriesShapeDescriptors object

        Parameters
        ---------------
        :param descriptors_array: Shape descriptors in the form of 2d numpy array. Each row
            represents shape descriptor of a single temporal point
        :param origin_ts: original time series

        Raises
        ---------------
        :raises EmptyShapeDescriptorsArray: Empty array was provided
        :raises UnivariateOriginTSShapeDescriptorsIncompatibility: Lenght of time
            series and number of row in shape descriptors array don't match
        :raises TooManyDimensionsArray: an array with too many dimensions has been
            provided
        """
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
        """
        Check number of dimensions for shape descriptors array

        Parameters
        ---------------
        :param descriptor_array: array of shape descriptors
        :param n_dim: number to compare with

        Returns
        ---------------
        :return: bool - results of test
        """
        return len(descriptor_array.shape) == n_dim

    @staticmethod
    def _array_is_empty(descriptor_array: ndarray) -> bool:
        """
        Verify that the array is empty

        Parameters
        ---------------
        :param descriptor_array: array of shape descriptors

        Returns
        ---------------
        :return: bool - results of test
        """
        return np.size(descriptor_array) == 0

    @staticmethod
    def _input_ts_descriptor_array_compatible(descriptor_array: ndarray, origin_ts: ndarray) -> bool:
        """
        Check that the supplied time series and the shape descriptor
        array are compatible, i.e., the length of the series is equal
        to the number of rows in the array

        Parameters
        ---------------
        :param descriptor_array: array of shape descriptors
        :param origin_ts: original time series

        Returns
        ---------------
        :return: bool - results of test
        """
        return origin_ts.shape[0] == descriptor_array.shape[0]

    def calc_distance_matrix(self, series_y_descriptor: UnivariateSeriesShapeDescriptors,
                             dist_method: str = "euclidean") -> UnivariateSeriesDistanceMatrix:
        """
        Calculates distance matrix between given array of shape descriptors
        and shape descriptors of another instance
        of the UnivariateSeriesShapeDescriptors

        Parameters
        ---------------
        :param series_y_descriptor: another instance of UnivariateSeriesShapeDescriptors
            for which a distance matrix will be calculated
        :param dist_method: method of distance calculation. Full list of methods available
            here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

        Raises
        ---------------
        :raises ObjectOfWrongClass: series_y_descriptor must be an isntance of
            UnivariateSeriesShapeDescriptors class

        Returns
        ---------------
        :returns: distance matrix - an instance of the UnivariateSeriesDistanceMatrix class
        """
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

    """
    Class representing a shape descriptors of multivariate time series.
    It provides a set of methods allowing to verify consistency of provided
    descriptors and calculate distance matrix between descriptors for two
    instances of MultivariateSeriesShapeDescriptors class. It has a capability
    to calculate a list of distance matrices separately for each dimension
    of time series (useful in case of multidimensional, independent dtw)
    as well as common distance matrix for all dimensions, used in case of
    multidimensional, dependent dtw.

    Attributes
    ---------------
    descriptors_list: List[UnivariateSeriesShapeDescriptors]:
        list of UnivariateSeriesShapeDescriptors instances, each representing
        shape descriptors for a single dimension of given time series
    origin_ts: ndarray:
        original time series as a 2d numpy array
    """

    def __init__(self, descriptors_list: List[UnivariateSeriesShapeDescriptors], origin_ts: ndarray):
        """
        Constructs a MultivariateSeriesShapeDescriptors object

        Parameters
        ---------------
        :param descriptors_list: list of UnivariateSeriesShapeDescriptors instances, each representing
            shape descriptors for a single dimension of given time series
        :param origin_ts: original time series as a 2d numpy array

        Raises
        ---------------
        :raises MultivariateOriginTSShapeDescriptorsDimIncompatibility: numbers
            of time series columns and length of shape descriptor list doesn't match
        :raises MultivariateOriginTSShapeDescriptorsLengthIncompatibility: there is a
            inconsistency between time series length and number of rows for at least
            one of shape descriptors array.
        """
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
        """
        Get length of the MultivariateSeriesShapeDescriptors as
        a number of univariate descriptors in the descriptors list

        Returns
        ---------------
        :returns: int - length of MultivariateSeriesShapeDescriptors object
        """
        return len(self.descriptors_list)

    @staticmethod
    def _is_one_dim_ts(origin_ts: ndarray) -> bool:
        """
        Verify that the provided time series is one-dimensional

        Parameters
        ---------------
        :param origin_ts: original time series in the form of
            numpy array

        Returns
        ---------------
        :return: bool - a results of the test
        """
        return len(origin_ts.shape) == 1

    def _input_ts_descriptor_dimensions_compatible(self) -> bool:
        """
        Verify that number of columns and number of
        UnivariateSeriesShapeDescriptors in an input list match

        Returns
        ---------------
        :return: results of a test
        """
        return len(self) == self.origin_ts.shape[1]

    def _input_ts_descriptors_length_compatible(self) -> bool:
        """
        Verify that the length of time series and number of rows in all
        of UnivariateSeriesShapeDescriptors match

        Returns
        ---------------
        :return: bool - a results of a test
        """
        ts_len = self.origin_ts.shape[0]
        return all(
            [uni_sd.shape_descriptors_array.shape[0] == ts_len
             for uni_sd in self.descriptors_list]
        )

    def _verify_dimension_compatibility(self, other: MultivariateSeriesShapeDescriptors) -> None:
        """
        Verify that numbers of dimensions (columns / univariate shape descriptors) for
        both MultivariateSeriesShapeDescriptors are equal. If not
        MultivariateSeriesShapeDescriptorsIncompatibility exception is raised.

        Parameters
        ---------------
        :param other: another MultivariateSeriesShapeDescriptors instance
            to compare with

        Raises
        ---------------
        :raises MultivariateSeriesShapeDescriptorsIncompatibility: numbers of dimensions
            (columns) of giver time series are different
        """
        ts_x_dim = len(self)
        ts_y_dim = len(other)
        if ts_x_dim != ts_y_dim:
            raise MultivariateSeriesShapeDescriptorsIncompatibility(ts_x_dim, ts_y_dim)

    def calc_distance_matrices(self, series_y_descriptor: MultivariateSeriesShapeDescriptors,
                               dist_method: str = "euclidean") -> MultivariateDistanceMatrixIndependent:
        """
        Calculates distance matrices between shape descriptors for each
        dimension of the time series. Used in case of multidimensional,
        independent shape dtw.

        Parameters
        ---------------
        :param series_y_descriptor: another instance of the MultivariateSeriesShapeDescriptors
            for which the distance matrix is to be calculated
        :param dist_method: str - distance method. Full list of available types of distance is to find
            here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

        Raises
        ---------------
        :raises ObjectOfWrongClass: series_y_descriptor is not an instance of
            MultivariateSeriesShapeDescriptors
        :raises MultivariateSeriesShapeDescriptorsIncompatibility: numbers of dimensions
            (columns) of giver time series are different

        Returns
        ---------------
        :return: MultivariateDistanceMatrixIndependent object, containing a list of
            distance matrices and the original time series

        Examples
        --------
        >> from shapedtw.preprocessing import MultivariateSubsequenceBuilder
        >> from shapedtw.shapeDescriptors import PAADescriptor
        >> import numpy as np
        >> ts_x = np.array([
        >>  [4.5, 1.3],
        >>  [6.7, 1.0],
        >>  [3.0, 1.1]
        >> ])
        >> ts_y = np.array([
        >>  [5.5, 2.5],
        >>  [3.3, 2.2],
        >>  [1.5, 2.6]
        >> ])
        >> paa_descriptor = PAADescriptor(2)
        >> ts_x_shape_descriptors = MultivariateSubsequenceBuilder(
        >>  ts_x, subsequence_width=2
        >>  ).transform_time_series_to_subsequences().get_shape_descriptors(
        >>  paa_descriptor
        >> )
        >> ts_y_shape_descriptors = MultivariateSubsequenceBuilder(
        >>  ts_y, subsequence_width=2
        >>  ).transform_time_series_to_subsequences().get_shape_descriptors(
        >>  paa_descriptor
        >> )
        >> dist_matrices = ts_x_shape_descriptors.calc_distance_matrices(
        >>  ts_y_shape_descriptors
        >> )
        >> for dist_matrix in dist_matrices.distance_matrices_list:
        >>      print(dist_matrix.dist_matrix)
        [[2.16564078 3.67287353 4.36692111]
         [1.85809042 3.04179223 3.67185239]
         [2.05426386 1.61864141 2.43721152]]
        [[2.26495033 2.29183333 2.33559414]
         [2.3194827  2.34787138 2.39895811]
         [2.37381549 2.40052078 2.43721152]]
        """
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
    ) -> ndarray:
        """
        Calculates a sum of distance matrices for Euclidean distance

        Parameters
        ---------------
        :param univariate_dist_matrices: list of distance matrices calculated for
            all dimensions of given time series

        Returns
        ---------------
        :return: sum of distance matrices as a numpy array
        """
        distance_matrices_list = [uni_mat.dist_matrix ** 2 for uni_mat in univariate_dist_matrices]
        distance_matrix = np.sqrt(reduce(operator.add, distance_matrices_list))
        return distance_matrix

    @staticmethod
    def _calc_sum_of_distance_matrices_non_euclidean(
            univariate_dist_matrices: List[UnivariateSeriesDistanceMatrix]
    ) -> np.ndarray:
        """
        Calculates a sum of distance matrices for non-euclidean distance

        Parameters
        ---------------
        :param univariate_dist_matrices: list of distance matrices calculated for
            all dimensions of given time series

        Returns
        ---------------
        :return: sum of distance matrices as a numpy array
        """
        distance_matrices_list = [uni_mat.dist_matrix for uni_mat in univariate_dist_matrices]
        distance_matrix = reduce(operator.add, distance_matrices_list)
        return distance_matrix

    def calc_summed_distance_matrix(self, series_y_descriptor: MultivariateSeriesShapeDescriptors,
                                    dist_method: str = "euclidean") -> MultivariateDistanceMatrixDependent:
        """
        Calculates distance matrices between shape descriptors for each
        dimension of the time series and adds them up. Used in case of
        multidimensional, dependent shape dtw.

        Parameters
        ---------------
        :param series_y_descriptor: another instance of the MultivariateSeriesShapeDescriptors
            for which the distance matrix is to be calculated
        :param dist_method: str - distance method. Full list of available types of distance is to find
            here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

        Returns
        ---------------
        :return: MultivariateDistanceMatrixDependent object, containing a single, common
            distance matrix and original time series.

        Examples
        --------
        >> from shapedtw.preprocessing import MultivariateSubsequenceBuilder
        >> from shapedtw.shapeDescriptors import PAADescriptor
        >> import numpy as np
        >> ts_x = np.array([
        >>  [4.5, 1.3],
        >>  [6.7, 1.0],
        >>  [3.0, 1.1]
        >> ])
        >> ts_y = np.array([
        >>  [5.5, 2.5],
        >>  [3.3, 2.2],
        >>  [1.5, 2.6]
        >> ])
        >> paa_descriptor = PAADescriptor(2)
        >> ts_x_shape_descriptors = MultivariateSubsequenceBuilder(
        >>  ts_x, subsequence_width=2
        >>  ).transform_time_series_to_subsequences().get_shape_descriptors(
        >>  paa_descriptor
        >> )
        >> ts_y_shape_descriptors = MultivariateSubsequenceBuilder(
        >>  ts_y, subsequence_width=2
        >>  ).transform_time_series_to_subsequences().get_shape_descriptors(
        >>  paa_descriptor
        >> )
        >> dist_matrix = ts_x_shape_descriptors.calc_summed_distance_matrix(
        >>  ts_y_shape_descriptors
        >> )
        >> print(dist_matrix.distance_matrix)
        [[3.13368792 4.32926091 4.95227221]
         [2.97195222 3.84252521 4.38605746]
         [3.13926743 2.89525474 3.44673759]]
        """

        univariate_dist_matrices = self.calc_distance_matrices(
            series_y_descriptor,
            dist_method
        ).distance_matrices_list

        if dist_method == "euclidean":
            distance_matrix = self._calc_sum_of_distance_matrices_euclidean(univariate_dist_matrices)
        else:
            distance_matrix = self._calc_sum_of_distance_matrices_non_euclidean(univariate_dist_matrices)

        return MultivariateDistanceMatrixDependent(distance_matrix, self.origin_ts, series_y_descriptor.origin_ts)


class DistanceMatrixCalculator:

    """
    Class allowing to calculate distance matrix between two time series using
    scipy 'cdist' method. It contains a set of functions which verify that
    both time series are fully compatible

    Attributes
    ---------------
    ts_x: ndarray:
        first time series as numpy array
    ts_y: ndarray:
        second time series as numpy array
    method: str:
        type of distance, 'euclidean' as a default. Full list of distances is to find
        here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
    """

    def __init__(self, ts_x: ndarray, ts_y: ndarray, method: str = "euclidean"):
        """
        Constructs a DistanceMatrixCalculator object

        Parameters
        ---------------
        :param ts_x: query time series as numpy array
        :param ts_y: reference time series as numpy array
        :param method: type of distance, full list of possible values is to find
            here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        """
        self.ts_x = ts_x
        self.ts_y = ts_y
        self.method = method

    def _input_ts_empty(self):
        """
        Checks that one or both time series are empty

        Returns
        ---------------
        :return: bool - result of test
        """
        return (np.size(self.ts_x) == 0) | (np.size(self.ts_x) == 0)

    def _two_dim_at_most(self):
        """
        Checks that time series are not > 2d arrays

        Returns
        ---------------
        :return: bool - result of test
        """
        return (len(self.ts_x.shape) < 3) & (len(self.ts_y.shape) < 3)

    def _series_shape_match(self):
        """
        Checks that shapes (number of dimensions) of time series match

        Returns
        ---------------
        :return: bool - result of test
        """
        return len(self.ts_x.shape) == len(self.ts_y.shape)

    def _series_are_univariate(self):
        """
        Checks that both time series are univariate

        Returns
        ---------------
        :return: bool - result of test
        """
        return (len(self.ts_x.shape) == 1) & (len(self.ts_y.shape) == 1)

    def _series_dimensions_match(self):
        """
        Checks that number of time series dimensions (columns) match

        Returns
        ---------------
        :return: bool - result of test
        """
        return self.ts_x.shape[1] == self.ts_y.shape[1]

    def _verify_dimensions(self):
        """
        Verify time series correctness and compatibility

        Raises
        ---------------
        :raises DimensionError: there is an issue with dimensions - details
            explained in the error message
        """
        if self._input_ts_empty():
            raise DimensionError("Empty arrays are not allowed.")

        if not self._two_dim_at_most():
            raise DimensionError("Only arrays of 1 and 2 dimensions are supported")

        if not self._series_shape_match():
            raise DimensionError("Shapes of time series are different")

        if not self._series_are_univariate():
            if not self._series_dimensions_match():
                raise DimensionError("Number of time series columns doesn't match")

    def _convert_one_dimension_series(self):
        """
        Transpone univariate time series to the form of
        2d matrix, required by calculation process
        """
        self.ts_x = np.atleast_2d(self.ts_x).T
        self.ts_y = np.atleast_2d(self.ts_y).T

    def calc_distance_matrix(self) -> ndarray:
        """
        Calculates distance matrix for given time series

        Raises
        ---------------
        :raises DimensionError: there is an issue with dimensions - details
            explained in the error message

        Returns
        ---------------
        :return: distance matrix in the form of numpy array

        Examples
        --------
        >> from shapedtw.preprocessing import DistanceMatrixCalculator
        >> import numpy as np
        >> ts_x = np.array([[1., 2.], [5., 2.], [4., 4.]])
        >> ts_y = np.array([[4., 1.], [3., 3.], [1., 6.]])
        >> dmc = DistanceMatrixCalculator(ts_x, ts_y, "euclidean")
        >> res = dmc.calc_distance_matrix()
        >> print(res)
        [[3.16227766 2.23606798 4.        ]
         [1.41421356 2.23606798 5.65685425]
         [3.         1.41421356 3.60555128]]
        """
        self._verify_dimensions()
        if self._series_are_univariate():
            self._convert_one_dimension_series()

        dist_matrix = cdist(self.ts_x, self.ts_y, metric=self.method)

        return dist_matrix


class UnivariateSeriesDistanceMatrix:
    """
    Class representing univariate series distance matrix. It stores
    output distance matrix and time series based on which it was
    calculated.
    """
    def __init__(self, dist_matrix: np.ndarray, ts_x: ndarray, ts_y: ndarray):
        self.dist_matrix = dist_matrix
        self.ts_x = ts_x
        self.ts_y = ts_y


class MultivariateDistanceMatrixIndependent:
    """
    Class representing distance matrices calculated for the purpose
    of multivariate, independent shape dtw. It stores list of
    UnivariateSeriesDistanceMatrix (for each of time series
    dimension / column respectively) and time series based on
    which it was calculated.
    """
    def __init__(self, distance_matrices_list: List[UnivariateSeriesDistanceMatrix], ts_x, ts_y):
        self.distance_matrices_list = distance_matrices_list
        self.ts_x = ts_x
        self.ts_y = ts_y


class MultivariateDistanceMatrixDependent:
    """
    Class representing multivariate series distance matrix, calculated
    for the purpose of multivariate, dependent shape dtw. It stores
    output distance matrix and time series based on which it was
    calculated.
    """
    def __init__(self, distance_matrix: ndarray, ts_x, ts_y):
        self.distance_matrix = distance_matrix
        self.ts_x = ts_x
        self.ts_y = ts_y
