import numpy as np
from typing import List

def get_number_of_dimensions(x: np.ndarray) -> int:
    return len(x.shape)

class SubsequenceShorterThanWindow(Exception):
    def __init__(self, subsequence_size: int, window_size: int):
        error_msg = "Subsequence length: {0}, window size: {1}".format(
            subsequence_size,
            window_size
        )
        super().__init__(error_msg)


class SubsequenceTooShort(Exception):
    def __init__(self, subsequence_size: int, min_required: int):
        error_msg = "Subsequence length: {0}, minimal required length: {1}".format(
            subsequence_size,
            min_required
        )
        super().__init__(error_msg)


class NotShapeDescriptor(Exception):
    def __init__(self, obj):
        error_msg = "Provided argument is object of class {0} instead of 'ShapeDescriptor'".format(
            obj.__class__.__name__
        )
        super().__init__(error_msg)


class WindowOfSizeOne(Exception):
    def __init__(self, descriptor_class):
        error_msg = "Window for {0} cannot be of size 1. Please specify another window width or subsequence length".format(
            descriptor_class
        )
        super().__init__(error_msg)


class WrongWeightsNumber(Exception):
    pass


class WrongPadType(Exception):
    pass


class NegativeSubsequenceWidth(Exception):
    pass


class TooManyDimensionsArray(Exception):
    def __init__(self, obj, n_dim):
        error_msg = "{0} requires single or two dimensions array as an input, {1}-dimension arrat was provided".format(
            obj.__class__.__name__,
            n_dim
        )
        super().__init__(error_msg)


class ObjectOfWrongClass(Exception):
    def __init__(self, actual_cls, expected_cls):
        error_msg = "Object of class {0} was provided, expected object of class {1}".format(
            actual_cls.__name__,
            expected_cls.__name__
        )
        super().__init__(error_msg)


class DimensionError(Exception):
    pass


class TooManyDimensions(DimensionError):

    def __init__(self, ts_x, ts_y):
        ts_x_dim_number = get_number_of_dimensions(ts_x)
        ts_y_dim_number = get_number_of_dimensions(ts_y)

        error_msg = """Only arrays which have 1 or 2 dimensions are supported.
                    "Number of x dims = {0}, number of y dims = {1}""".format(ts_x_dim_number, ts_y_dim_number)

        super().__init__(error_msg)


class IncompatibleDimensionality(DimensionError):
    def __init__(self, ts_x, ts_y):
        ts_x_dim_number = get_number_of_dimensions(ts_x)
        ts_y_dim_number = get_number_of_dimensions(ts_y)

        error_msg = "Incompatible dimensionality, series x dim = {0}d, series y dim = {1}d".format(
            ts_x_dim_number, ts_y_dim_number
        )

        super().__init__(error_msg)

class MultivariateSeriesShapeDescriptorsIncompatibility(DimensionError):

    def __init__(self, ts_x_dim, ts_y_dim):
        error_msg = "Incompatible dimensionality, series x dim = {0}d, series y dim = {1}d".format(
            ts_x_dim, ts_y_dim
        )

        super().__init__(error_msg)


class IncompatibleSeriesNumber(DimensionError):
    def __init__(self, ts_x: np.ndarray, ts_y: np.ndarray):

        ts_x_series_num = ts_x.shape[1]
        ts_y_series_num = ts_y.shape[1]

        error_msg = "Incompatible number of series, series x = {0}, series y = {1}".format(
            ts_x_series_num, ts_y_series_num
        )

        super().__init__(error_msg)

class DTWNotCalculatedYet(Exception):
    pass

class ShapeDTWNotCalculatedYet(Exception):
    pass


class DistanceSettingNotPossible(Exception):
    pass


class WarpingPathSettingNotPossible(Exception):
    pass


class ShapeDescriptorError(Exception):
    pass

class EmptyShapeDescriptorsArray(Exception):
    pass

class OriginTSShapeDescriptorIncompatibility(Exception):
    pass

class UnivariateOriginTSShapeDescriptorsIncompatibility(OriginTSShapeDescriptorIncompatibility):
    def __init__(self, origin_ts_len: int, shape_descriptor_array_row_num: int):
        error_msg = """Origin time series and shape descriptor array incompatible. 
        Time series length = {0}, shape descriptor array row number = {1}
        """.format(origin_ts_len, shape_descriptor_array_row_num)

        super().__init__(error_msg)

class MultivariateOriginTSShapeDescriptorsDimIncompatibility(OriginTSShapeDescriptorIncompatibility):

    def __init__(self, origin_ts_dim: int, shape_descriptor_list_length: int):
        error_msg = """Origin time series and shape descriptor list incompatible.
        Number of time series dimensions = {0}, number of provided shape descriptors = {1}
        """.format(origin_ts_dim, shape_descriptor_list_length)

        super().__init__(error_msg)

class MultivariateOriginTSShapeDescriptorsLengthIncompatibility(OriginTSShapeDescriptorIncompatibility):

    def __init__(self, origin_ts_length: int, shape_descriptor_lengths: List[int]):
        error_msg = """Origin time series and shape descriptor list incompatible.
        Length of time series = {0}, lengths of shape descriptors = {1}
        """.format(origin_ts_length, shape_descriptor_lengths)

        super().__init__(error_msg)

class WrongSlopeWindow(ShapeDescriptorError):

    def __init__(self, slope_window):
        error_msg = "Slope window must be integer greater than 1, got {0} instead".format(
            slope_window
        )

        super().__init__(error_msg)

class ProvidedStepPatternDoesNotExists(Exception):
    pass


class SegmentIndexOutOfRange(Exception):
    def __init__(self, provided_segment_number, actual_number_of_segments):
        error_msg = "Provided number of segment: {0}. Range of segments: (1, {1})".format(
            provided_segment_number, actual_number_of_segments
        )

        super().__init__(error_msg)

class WrongMultivariateVersionSpecified(Exception):
    def __init__(self, provided_version: str):
        error_msg = "Multivariate shape dtw version must be one of following: ['dependent', 'independent']. Provided: '{0}'".format(
            provided_version
        )

        super().__init__(error_msg)