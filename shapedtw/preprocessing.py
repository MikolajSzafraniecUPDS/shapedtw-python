import numpy as np

from numpy import array
from .exceptions import *

class Padder:

    """
    Auxiliary class, storing a set of methods which can be
    used to get subsequences padded with first or last values of
    time series. Padding is applied during the process of transforming
    time series into a set of subsequences. It has to be done in order
    to make shape descriptors for starting and ending points of time
    series well-defined.
    """

    def __init__(self, time_series: array, subsequence_width: int):
        self.time_series = time_series
        self.ts_length = len(time_series)
        self.subsequence_width = subsequence_width

    def _get_pad_number(self, ts_index: int, pad_side: str) -> int:
        if pad_side == "left":
            n_padded = self.subsequence_width - ts_index
        elif pad_side == "right":
            n_padded = (ts_index+1)-self.ts_length+self.subsequence_width
        else:
            raise WrongPadType("Pad type must bo one of ('left', 'rigth')")

        return n_padded

    def pad_left(self, ts_index: int) -> array:
        n_padded = self._get_pad_number(ts_index, "left")
        last_ind = ts_index+self.subsequence_width+1
        ts_first_observation = self.time_series[0]
        first_observation_repeated = np.repeat(ts_first_observation,n_padded)
        subsequence = np.concatenate(
            (first_observation_repeated, self.time_series[0:last_ind])
        )

        return subsequence

    def pad_right(self, ts_index: int) -> array:
        ts_len = len(self.time_series)
        n_padded = self._get_pad_number(ts_index, "right")
        first_ind = ts_index-self.subsequence_width
        ts_last_observation = self.time_series[ts_len-1]
        ts_last_repeated = np.repeat(ts_last_observation,n_padded)
        subsequence = np.concatenate(
            (self.time_series[first_ind:], ts_last_repeated)
        )

        return subsequence

    def pad_both_side(self, ts_index: int) -> array:
        ts_len = len(self.time_series)
        n_padded_left = self._get_pad_number(ts_index, "left")
        n_padded_rigth = self._get_pad_number(ts_index, "right")
        ts_first_observation = self.time_series[0]
        ts_last_observation = self.time_series[ts_len-1]
        ts_first_observation_repeated = np.repeat(ts_first_observation, n_padded_left)
        ts_last_observation_repeated = np.repeat(ts_last_observation, n_padded_rigth)
        subsequence = np.concatenate(
            (
                ts_first_observation_repeated,
                self.time_series,
                ts_last_observation_repeated
            )
        )

        return subsequence


class SubsequenceBuilder:

    """
    This class is used for the purpose of transforming univariate time
    series into a set of subsequences. It is represented by two-dimensional
    array, where number of rows is equal to the time series length and number
    of column is equal to the specified subsequence length.
    """

    def __init__(self, time_series: array, subsequence_width: int):

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

    def _get_subsequence_without_padding(self, ts_index: int) -> array:
        subsequence_start = ts_index - self.subsequence_width
        subsequence_end = ts_index + self.subsequence_width + 1
        return self.time_series[subsequence_start:subsequence_end]

    def _get_single_subsequence(self, ts_index: int) -> array:
        if self._apply_padding_both_sides(ts_index):
            subsequence = self.padder.pad_both_side(ts_index)
        elif self._apply_left_padding(ts_index):
            subsequence = self.padder.pad_left(ts_index)
        elif self._apply_right_padding(ts_index):
            subsequence = self.padder.pad_right(ts_index)
        else:
            subsequence = self._get_subsequence_without_padding(ts_index)

        return subsequence

    def transform_time_series_to_subsequences(self):
        indices = np.arange(start=0, stop=self.ts_length)
        subsequences_list = [self._get_single_subsequence(ts_index) for ts_index in indices]
        subsequences_array = np.vstack(subsequences_list)
        return UnivariateSeriesSubsequences(subsequences_array)


class UnivariateSeriesSubsequences:

    def __init__(self, subsequences_array: array):
        self.subsequences = subsequences_array

    def get_shape_descriptors(self, shape_descriptor):
        shape_descriptors = np.array([
            shape_descriptor.get_shape_descriptor(subsequence) for
            subsequence in self.subsequences
        ])

        return UnivariatSeriesShapeDescriptors(shape_descriptors)


class UnivariatSeriesShapeDescriptors:

    def __init__(self, descriptors_array: array):
        self.shape_descriptors_array = descriptors_array
