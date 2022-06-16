import numpy as np

from pywt import Wavelet, wavedec
from abc import abstractmethod
from numpy import array
from typing import List
from .exceptions import SubsequenceShorterThanWindow


class ShapeDescriptor:
    """
    According to Zhao and Itti concept so-called shape descriptors are the basis of the shape DTW algorithm. They allow us to
    transform subsequence of given time series to a vector of values representing it's local shape. Shape DTW algorithm uses shape
    descriptors instead of raw time series values to calculate optimal warping path.

    There are a few shape descriptors described in the Zhao and Itti paper and one can define his own descriptor.
    ShapeDescriptor is an abstract class, intended to be a parent class for all of the shape descriptors. It forces
    to implement a 'get_shape_descriptor' method, transforming a given time series chunk to its shape descriptor.
    """

    @abstractmethod
    def get_shape_descriptor(self, ts_subsequence: array) -> array:
        pass

    @staticmethod
    def _split_into_windows(ts_subsequence: array, window_size: int) -> List[array]:
        indices_to_split = np.arange(
            window_size,
            len(ts_subsequence),
            window_size
        )
        return np.split(ts_subsequence, indices_to_split)


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

    def _subsequence_is_shorter_than_window_size(self, ts_subsequence: array) -> bool:
        return len(ts_subsequence) < self.piecewise_aggregation_window

    @staticmethod
    def _get_windows_means(windows):
        windows_means = array([np.mean(window) for window in windows])
        return windows_means

    def get_shape_descriptor(self, ts_subsequence: array) -> array:

        if self._subsequence_is_shorter_than_window_size(ts_subsequence):
            error_msg = "Subsequence length: {0}, window size: {1}".format(
                len(ts_subsequence),
                self.piecewise_aggregation_window
            )
            raise SubsequenceShorterThanWindow(error_msg)

        windows = self._split_into_windows(ts_subsequence, self.piecewise_aggregation_window)
        paa_descriptor = self._get_windows_means(windows)

        return paa_descriptor

class DWTDescriptor(ShapeDescriptor):

    """
    Discrete Wavelet Transform (DWT) is another widely used
    technique to approximate time series instances. Again, here we use
    DWT to approximate subsequences. Concretely, we use a Haar
    wavelet basis (as a default) to decompose each subsequence si into 3 levels.
    The detail wavelet coefficients of all three levels and the approximation
    coefficients of the third level are concatenated to form the
    approximation, which is used the shape descriptor di of si, i.e.,
    F(·) = DWT, di = DWT (si).
    """

    def __init__(self, wave_type: str = "haar", mode: str = "sym", level: int = 3):
        self.wave_type = wave_type
        self.mode = mode
        self.level = level

    def get_shape_descriptor(self, ts_subsequence: array) -> array:
        wavelet = Wavelet(self.wave_type)
        coefs_list = wavedec(ts_subsequence, wavelet, mode=self.mode, level = self.level)
        dwt_descriptor = np.concatenate(coefs_list)
        return dwt_descriptor
