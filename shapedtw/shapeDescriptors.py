import numpy as np

from pywt import Wavelet, wavedec
from abc import abstractmethod, ABC
from numpy import ndarray
from typing import List
from scipy.stats import linregress
from shapedtw.exceptions import *
from itertools import repeat


class ShapeDescriptor(ABC):
    """
    According to Zhao and Itti concept so-called shape descriptors are the basis of the shape DTW algorithm. They allow
    us to transform subsequence of given time series to a vector of values representing it's local shape.
    Shape DTW algorithm uses shape descriptors instead of raw time series values to calculate optimal warping path.

    There are a few shape descriptors described in the Zhao and Itti paper and one can define his own descriptor.
    ShapeDescriptor is an abstract class, intended to be a parent class for all of the shape descriptors. It forces
    to implement a 'get_shape_descriptor' method, transforming a given time series chunk to its shape descriptor.
    """

    @abstractmethod
    def get_shape_descriptor(self, ts_subsequence: ndarray) -> ndarray:
        pass

    @staticmethod
    def _subsequence_is_shorter_than_window_size(subsequence_len: int, window_size: int) -> bool:
        return subsequence_len < window_size

    @staticmethod
    def _split_into_windows(ts_subsequence: ndarray, window_size: int) -> List[ndarray]:

        subsequence_len = len(ts_subsequence)

        if ShapeDescriptor._subsequence_is_shorter_than_window_size(subsequence_len, window_size):
            raise SubsequenceShorterThanWindow(subsequence_len, window_size)

        indices_to_split = np.arange(
            window_size,
            subsequence_len,
            window_size
        )
        return np.split(ts_subsequence, indices_to_split)


class RawSubsequenceDescriptor(ShapeDescriptor):

    """
    The most basic shape descriptor, returning given raw subsequence itself.
    """

    def get_shape_descriptor(self, time_series_subsequence: ndarray) -> ndarray:
        return time_series_subsequence


class PAADescriptor(ShapeDescriptor):

    """
    Piecewise aggregation approximation is an y-shift variant shape descriptor. Given subsequence is split
    into m equally length chunks. For each of the chunks mean values of temporal points falling within
    an interval is calculated and a vector af mean values is used as a shape descriptor.

    Length of intervals is specified by "piecewise_aggregation_window" argument provided in the class
    constructor. If it is impossible to split array into chunks of equal length, then the last chunk
    is adequately shorter.
    """

    def __init__(self, piecewise_aggregation_window: int = 2):
        self.piecewise_aggregation_window = piecewise_aggregation_window

    @staticmethod
    def _get_windows_means(windows: List[ndarray]) -> ndarray:
        windows_means = np.array([np.mean(window) for window in windows])
        return windows_means

    def get_shape_descriptor(self, ts_subsequence: ndarray) -> ndarray:
        windows = self._split_into_windows(ts_subsequence, self.piecewise_aggregation_window)
        paa_descriptor = self._get_windows_means(windows)

        return paa_descriptor


class DWTDescriptor(ShapeDescriptor):

    """
    Definition after Zhao and Itti:
    'Discrete Wavelet Transform (DWT) is another widely used
    technique to approximate time series instances. Again, here we use
    DWT to approximate subsequences. Concretely, we use a Haar
    wavelet basis (as a default) to decompose each subsequence si into 3 levels.
    The detail wavelet coefficients of all three levels and the approximation
    coefficients of the third level are concatenated to form the
    approximation, which is used the shape descriptor di of si, i.e.,
    F(·) = DWT, di = DWT (si).'
    """

    def __init__(self, wave_type: str = "haar", mode: str = "sym", level: int = 3):
        self.wave_type = wave_type
        self.mode = mode
        self.level = level

    def get_shape_descriptor(self, ts_subsequence: ndarray) -> ndarray:
        wavelet = Wavelet(self.wave_type)
        coefs_list = wavedec(ts_subsequence, wavelet, mode=self.mode, level=self.level)
        dwt_descriptor = np.concatenate(coefs_list)
        return dwt_descriptor


class SlopeDescriptor(ShapeDescriptor):

    """
    Slope descriptor is a shape descriptor that is invariant to y-shift. It means, that
    two subsequences of the same shape which values are shifted on y axis by some delta
    will be characterized by equal descriptors despite this difference.

    Definition after Zhao and Itti:
    'Given a l-dimensional subsequence si, it is divided into m (m ≤ l)
    equal-lengthed intervals. Within each interval, we employ the total
    least square (TLS) line fitting approach [11] to fit a line according
    to points falling within that interval. By concatenating the slopes
    of the fitted lines from all intervals, we obtain a m-dimensional
    vector representation, which is the slope representation of si, i.e.,
    F(·) = Slope, di = Slope(si).'
    """

    def __init__(self, slope_window: int = 2):
        self.slope_window = slope_window

    @staticmethod
    def _get_single_slope(input_vector: ndarray) -> float:
        vector_length = len(input_vector)
        if vector_length == 1:
            return float(0)
        x_vec = np.arange(vector_length)
        linregress_res = linregress(x=x_vec, y=input_vector)
        return float(linregress_res.slope)

    @staticmethod
    def _get_windows_slopes(windows: List[ndarray]) -> ndarray:
        windows_slopes = np.array([SlopeDescriptor._get_single_slope(window) for window in windows])
        return windows_slopes

    def get_shape_descriptor(self, ts_subsequence: ndarray) -> ndarray:
        windows = self._split_into_windows(ts_subsequence, self.slope_window)
        slope_descriptor = self._get_windows_slopes(windows)

        return slope_descriptor


class DerivativeShapeDescriptor(ShapeDescriptor):

    """
    Definition after Zhao and Itti:
    'Similar to Slope, Derivative is y-shift invariant if it is used to
    represent shapes. Given a subsequence s, its first-order derivative
    sequence is s′, where s′ is the first order derivative according
    to time t. To keep consistent with derivatives used in derivative
    Dynamic Time Warping (E. Keogh and M. Pazzani. Derivative dynamic time warping. In SDM,
    volume 1, pages 5–7. SIAM, 2001.) (dDTW), we follow their formula to
    compute numeric derivatives.'
    """

    @staticmethod
    def _get_first_order_diff(ts_subsequence: ndarray) -> ndarray:
        return ts_subsequence[1:] - ts_subsequence[:-1]

    @staticmethod
    def _get_second_order_diff(ts_subsequence: ndarray) -> ndarray:
        return (ts_subsequence[2:] - ts_subsequence[:-2]) / 2

    @staticmethod
    def _get_derivative(first_order_diff: ndarray, second_order_diff: ndarray) -> ndarray:
        return (first_order_diff[:-1] + second_order_diff) / 2

    def get_shape_descriptor(self, ts_subsequence: ndarray) -> ndarray:
        subsequence_length = len(ts_subsequence)
        if subsequence_length < 3:
            raise SubsequenceTooShort(subsequence_size=subsequence_length, min_required=3)

        first_order_diff = self._get_first_order_diff(ts_subsequence)
        second_order_diff = self._get_second_order_diff(ts_subsequence)
        derivative_descriptor = self._get_derivative(first_order_diff, second_order_diff)

        return derivative_descriptor


class CompoundDescriptor(ShapeDescriptor):

    """
    Compound shape descriptor is a simple concatenation of provided shape descriptors. It is possible
    to specify a weights for each of them - it is worth to do if scales of values of chosen descriptors
    differs significantly.
    """

    def __init__(self, *shape_descriptors: List[ShapeDescriptor], descriptors_weights: List[float] = None):

        descriptors_number = len(shape_descriptors)

        if descriptors_weights is None:
            descriptors_weights = list(repeat(1, descriptors_number))

        weights_len = len(descriptors_weights)

        if weights_len != descriptors_number:
            raise WrongWeightsNumber("Number of weights and shape descriptors must match")

        for descriptor in shape_descriptors:
            if not isinstance(descriptor, ShapeDescriptor):
                raise NotShapeDescriptor(descriptor)

        self.shape_descriptors = shape_descriptors
        self.descriptors_weights = descriptors_weights

    def _calc_descriptors(self, ts_subsequence: ndarray) -> List[ndarray]:
        descriptors_list = [
            descriptor.get_shape_descriptor(ts_subsequence) * weight for
            (descriptor, weight) in
            zip(self.shape_descriptors, self.descriptors_weights)
        ]

        return descriptors_list

    def get_shape_descriptor(self, ts_subsequence: ndarray) -> ndarray:
        descriptors_list = self._calc_descriptors(ts_subsequence)
        compound_descriptor = np.concatenate(descriptors_list)

        return compound_descriptor
