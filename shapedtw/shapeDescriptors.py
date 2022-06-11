import numpy as np

from abc import ABC, abstractmethod
from numpy import array
from typing import List
from .exceptions import SubsequenceShorterThanWindow


class ShapeDescriptor(ABC):
    """
    According to Zhao and Itti concept so-called shape descriptors are the basis of the shape DTW algorithm. They allow us to
    transform subsequence of given time series to a vector of values representing it's local shape. Shape DTW algorithm uses shape
    descriptors instead of raw time series values to calculate optimal warping path.

    There are a few shape descriptors described in the Zhao and Itti paper and one can define his own descriptor.
    ShapeDescriptor is an abstract class, intended to be a parent class for all of the shape descriptors. It forces
    to implement a 'get_shape_descriptor' method, transforming a given time series chunk to its shape descriptor.
    """

    @abstractmethod
    def get_shape_descriptor(self, time_series_subsequence: array) -> array:
        pass


class RawSubsequenceDescriptor(ShapeDescriptor):

    """
    The most basic shape descriptor, returning given raw subsequence itself.
    """

    def get_shape_descriptor(self, time_series_subsequence: array) -> array:
        return time_series_subsequence


class PAADescriptor(ShapeDescriptor):

    """
    Piecewise aggregation approximation is an y-variant shape descriptor. Given subsequence is split
    into m equally length chunks. For each of the chunks mean values of temporal points falling within
    an interval is calculated and a vector af mean values is used as a shape descriptor.

    Length of intervals is specified by "piecewise_aggregation_window" argument provided in the class
    constructor. If it is impossible to split array into chunks of equal length, then the last chunk
    is adequately shorter.
    """

    def __init__(self, piecewise_aggregation_window: int = 2):
        self.piecewise_aggregation_window = piecewise_aggregation_window

    def subsequence_is_shorter_than_window_size(self, ts_subsequence: array) -> bool:
        return len(ts_subsequence) < self.piecewise_aggregation_window

    def split_into_windows(self, ts_subsequence: array) -> List[array]:
        indices_to_split = np.arange(
            self.piecewise_aggregation_window,
            len(ts_subsequence),
            self.piecewise_aggregation_window
        )
        return np.split(ts_subsequence, indices_to_split)

    @staticmethod
    def get_windows_means(windows):
        windows_means = array([np.mean(window) for window in windows])
        return windows_means

    def get_shape_descriptor(self, ts_subsequence: array) -> array:

        if self.subsequence_is_shorter_than_window_size(ts_subsequence):
            error_msg = "Subsequence length: {0}, window size: {1}".format(
                len(ts_subsequence),
                self.piecewise_aggregation_window
            )
            raise SubsequenceShorterThanWindow(error_msg)

        windows = self.split_into_windows(ts_subsequence)
        paa_descriptor = self.split_into_windows(windows)

        return paa_descriptor