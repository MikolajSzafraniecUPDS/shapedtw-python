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

"""Module containing useful exceptions"""

import numpy as np
from typing import List

def get_number_of_dimensions(x: np.ndarray) -> int:
    """
    Get number of array's dimension (1 for 1d array, 2 for 2d array, etc.)

    Parameters
    ---------------
    :param x: numpy array to verify

    Returns
    ---------------
    :return: int - number of dimensions
    """
    return len(x.shape)

class SubsequenceShorterThanWindow(Exception):

    """
    Exception informing that the length of given subsequence is shorter
    than window specified for chosen shape descriptor (for example slope
    descriptor or paa descriptor)
    """

    def __init__(self, subsequence_size: int, window_size: int):
        """
        Constructs a SubsequenceShorterThanWindow exception

        Parameters
        ---------------
        :param subsequence_size: size of time series subsequence
        :param window_size: size of provided window for shape descriptor
        """
        error_msg = "Subsequence length: {0}, window size: {1}".format(
            subsequence_size,
            window_size
        )
        super().__init__(error_msg)


class SubsequenceTooShort(Exception):
    """
    Exception informing that given subsequence is too short to calculate
    given shape descriptor. For example derivative shape descriptor requires
    subsequences of length 3 at least.
    """
    def __init__(self, subsequence_size: int, min_required: int):
        """
        Constructs a SubsequenceTooShort exception

        Parameters
        ---------------
        :param subsequence_size: size of time series subsequence
        :param window_size: size of provided window for shape descriptor
        """
        error_msg = "Subsequence length: {0}, minimal required length: {1}".format(
            subsequence_size,
            min_required
        )
        super().__init__(error_msg)


class NotShapeDescriptor(Exception):
    """
    Exception informing that provided object is not a child class of
    ShapeDescriptor in case when it is required.
    """
    def __init__(self, obj):
        """
        Constructs a SubsequenceShorterThanWindow exception

        Parameters
        ---------------
        :param obj: provided object
        """
        error_msg = "Provided argument is object of class {0} instead of 'ShapeDescriptor'".format(
            obj.__class__.__name__
        )
        super().__init__(error_msg)


class WrongWeightsNumber(Exception):
    """
    Exception raised in CompoundDescriptor constructor. It informs that
    number of provided shape descriptors and number of shape descriptor weights
    are not equal.
    """
    pass


class NegativeSubsequenceWidth(Exception):
    """
    Exception raised in case when user provided negative subsequence
    width.
    """
    pass


class ObjectOfWrongClass(Exception):
    """
    Exception informing that the object of wrong class was provided
    """
    def __init__(self, actual_cls, expected_cls):
        """
        Constructs ObjectOfWrongClass exception

        Parameters
        ---------------
        :param actual_cls: type of provided object
        :param expected_cls: expected type
        """
        error_msg = "Object of class {0} was provided, expected object of class {1}".format(
            actual_cls.__name__,
            expected_cls.__name__
        )
        super().__init__(error_msg)


class DimensionError(Exception):
    """
    Parent class for exceptions related to dimensionality errors
    """
    pass


class TooManyDimensionsArray(DimensionError):
    """
    Exception raised in case when user provided time series in the form of
    array having too many dimensions in particular situation
    """
    def __init__(self, obj, n_dim):
        """
        Constructs TooManyDimensionsArray exception

        Parameters
        ---------------
        :param obj: object for which exception is being raised
        :param n_dim: number of dimension of provided array
        """
        error_msg = "{0} requires single or two dimensions array as an input, {1}-dimension array was provided".format(
            obj.__class__.__name__,
            n_dim
        )
        super().__init__(error_msg)


class TooManyDimensions(DimensionError):
    """
    Exception informing that at least one of provided time series
    has a form of >2d array which is not supported
    """
    def __init__(self, ts_x: np.ndarray, ts_y: np.ndarray):
        """
        Constructs TooManyDimensions exception

        Parameters
        ---------------
        :param ts_x: query time series
        :param ts_y: reference time series
        """
        ts_x_dim_number = get_number_of_dimensions(ts_x)
        ts_y_dim_number = get_number_of_dimensions(ts_y)

        error_msg = """Only arrays which have 1 or 2 dimensions are supported.
                    "Number of x dims = {0}, number of y dims = {1}""".format(ts_x_dim_number, ts_y_dim_number)

        super().__init__(error_msg)


class IncompatibleDimensionality(DimensionError):
    """
    Exception informing that provided time series are incompatible in
    terms of dimensionality - for example 1d and 2d array were provided
    """
    def __init__(self, ts_x: np.ndarray, ts_y: np.ndarray):
        """
        Constructs IncompatibleDimensionality exception

        Parameters
        ---------------
        :param ts_x: query time series in the form of numpy array
        :param ts_y: reference time series in the form of numpy array
        """
        ts_x_dim_number = get_number_of_dimensions(ts_x)
        ts_y_dim_number = get_number_of_dimensions(ts_y)

        error_msg = "Incompatible dimensionality, series x dim = {0}d, series y dim = {1}d".format(
            ts_x_dim_number, ts_y_dim_number
        )

        super().__init__(error_msg)

class MultivariateSeriesShapeDescriptorsIncompatibility(DimensionError):
    """
    Exception informing that the number of shape descriptor matrices between
    MultivariateSeriesShapeDescriptors objects doesn't match. It might mean
    for example that one of provided MultivariateSeriesShapeDescriptors was
    built for time series containing two subseries (two columns) and second
    one was built for time series containing three susberies (three columns).
    """
    def __init__(self, ts_x_dim: int, ts_y_dim: int):
        """
        Constructs MultivariateSeriesShapeDescriptorsIncompatibility exception

        Parameters
        ---------------
        :param ts_x_dim: number of columns in query time series
        :param ts_y_dim: number of columns in reference time series
        """
        error_msg = "Incompatible dimensionality, series x dim = {0}d, series y dim = {1}d".format(
            ts_x_dim, ts_y_dim
        )

        super().__init__(error_msg)


class IncompatibleSubseriesNumber(DimensionError):
    """
    Exception informing that number of columns between provided multidimensional
    time series doesn't match.
    """
    def __init__(self, ts_x: np.ndarray, ts_y: np.ndarray):
        """
        Constructs IncompatibleSubseriesNumber exception

        Parameters
        ---------------
        :param ts_x: query series as numpy array
        :param ts_y: reference series as numpy array
        """
        ts_x_series_num = ts_x.shape[1]
        ts_y_series_num = ts_y.shape[1]

        error_msg = "Incompatible number of series, series x = {0}, series y = {1}".format(
            ts_x_series_num, ts_y_series_num
        )

        super().__init__(error_msg)


class ShapeDTWNotCalculatedYet(Exception):
    """
    Exception raised in case when user attempts to retrieve shape dtw results
    (distances / warping paths) before they were calculated.
    """
    pass


class DistanceSettingNotPossible(Exception):
    """
    Exception raised in case when user attempts to set shape dtw
    results (distances) by assignment instead of calculating them
    """
    pass


class WarpingPathSettingNotPossible(Exception):
    """
    Exception raised in case when user attempts to set shape dtw
    results (warping paths) by assignment instead of calculating them
    """
    pass


class ShapeDescriptorError(Exception):
    """
    Parent class for exceptions related to shape descriptors
    """
    pass

class EmptyShapeDescriptorsArray(ShapeDescriptorError):
    """
    Exception informing that provided shape descriptors
    array is an empty array
    """
    pass

class OriginTSShapeDescriptorIncompatibility(ShapeDescriptorError):
    """
    Parent class for exceptions related to original time series and its
    shape descriptor array incompatibility
    """
    pass

class UnivariateOriginTSShapeDescriptorsIncompatibility(OriginTSShapeDescriptorIncompatibility):
    """
    Exception informing that there is a mismatch between a length of univariate time
    series and number of rows of provided shape descriptors array (those numbers are
    supposed to be equal).
    """
    def __init__(self, origin_ts_len: int, shape_descriptor_array_row_num: int):
        """
        Constructs UnivariateOriginTSShapeDescriptorsIncompatibility exception

        Parameters
        ---------------
        :param origin_ts_len: length of original univariate time series
        :param shape_descriptor_array_row_num: number of rows of provided shape
            descriptors array
        """
        error_msg = """Origin time series and shape descriptor array incompatible. 
        Time series length = {0}, shape descriptor array row number = {1}
        """.format(origin_ts_len, shape_descriptor_array_row_num)

        super().__init__(error_msg)

class MultivariateOriginTSShapeDescriptorsDimIncompatibility(OriginTSShapeDescriptorIncompatibility):
    """
    Exception informing that there is a mismatch between a number of subseries
    (columns) of multivariate time series and number of provided shape descriptors
    matrices
    """

    def __init__(self, origin_ts_dim: int, shape_descriptor_list_length: int):
        """
        Constructs MultivariateOriginTSShapeDescriptorsDimIncompatibility exception

        Parameters
        ---------------
        :param origin_ts_dim: number of original time series subseries (columns)
        :param shape_descriptor_list_length: number of shape descriptors matrices
        """
        error_msg = """Origin time series and shape descriptor list incompatible.
        Number of time series dimensions = {0}, number of provided shape descriptors = {1}
        """.format(origin_ts_dim, shape_descriptor_list_length)

        super().__init__(error_msg)

class MultivariateOriginTSShapeDescriptorsLengthIncompatibility(OriginTSShapeDescriptorIncompatibility):
    """
    Exception informing that there is a mismatch between original time series length
    and number of rows of provided shape descriptor array (those numbers are supposed
    to be equal).
    """
    def __init__(self, origin_ts_length: int, shape_descriptor_lengths: List[int]):
        """
        Constructs MultivariateOriginTSShapeDescriptorsLengthIncompatibility exception

        Parameters
        ---------------
        :param origin_ts_length: length (number of rows) of original multivariate time series
        :param shape_descriptor_lengths: number of rows of shape descriptor array
        """
        error_msg = """Origin time series and shape descriptor list incompatible.
        Length of time series = {0}, lengths of shape descriptors = {1}
        """.format(origin_ts_length, shape_descriptor_lengths)

        super().__init__(error_msg)

class WrongSlopeWindow(ShapeDescriptorError):
    """
    Exception informing that provided slope window for slope shape descriptor
    is not an integer greater than 1, as supposed
    """
    def __init__(self, slope_window):
        """
        Constructs WrongSlopeWindow exception

        Parameters
        ---------------
        :param slope_window: provided slope window
        """
        error_msg = "Slope window must be integer greater than 1, got {0} instead".format(
            slope_window
        )

        super().__init__(error_msg)

class ProvidedStepPatternDoesNotExists(Exception):
    """
    Exception informing that the name of step pattern provided by user
    is wrong - there is no such step pattern
    """
    pass


class SegmentIndexOutOfRange(Exception):
    """
    Exception informing that number of provided step pattern segment is invalid - it
    should be greater than zero and lower or equal to the number of given step pattern's
    segments. Related to StepPatternMatrixTransformator class.
    """
    def __init__(self, provided_segment_number, actual_number_of_segments):
        """
        Constructs SegmentIndexOutOfRange exception

        Parameters
        ---------------
        :param provided_segment_number: provided number of step pattern segment
        :param actual_number_of_segments: number of given step pattern's segment
        """
        error_msg = "Provided number of segment: {0}. Range of segments: (1, {1})".format(
            provided_segment_number, actual_number_of_segments
        )

        super().__init__(error_msg)

class WrongMultivariateVersionSpecified(Exception):
    """
    Exception informing that provided name of multivariate shape dtw version
    is invalid. It should be one of: 'dependent', 'independent'
    """
    def __init__(self, provided_version: str):
        """
        Constructs WrongMultivariateVersionSpecified exception

        Parameters
        ---------------
        :param provided_version: 'multivariate_version' parameter which was provided to 'shape_dtw' function
        """
        error_msg = "Multivariate shape dtw version must be one of following: ['dependent', 'independent']. Provided: '{0}'".format(
            provided_version
        )

        super().__init__(error_msg)

class InputTimeSeriesUnsupportedType(Exception):
    """
    Exception informing that type of object provided as an input time series to 'shape_dtw'
    function is not supported yet
    """
    def __init__(self, supported_types: List[type], type_of_provided_object: str):
        """
        Constructs InputTimeSeriesUnsupportedType exception

        Parameters
        ---------------
        :param supported_types: list of types of objects which are currently supported as
            input time series
        :param type_of_provided_object: type of provided object
        """
        error_msg = "Time series must be one of following types: {0}. Provided type '{1}' is not currently supported".format(
            supported_types,
            type_of_provided_object
        )

        super().__init__(error_msg)