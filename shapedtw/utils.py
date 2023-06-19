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

"""Classes and methods which can be reused in different parts of code"""

import sys

import numpy as np
from shapedtw.exceptions import IncompatibleDimensionality, TooManyDimensions, ProvidedStepPatternDoesNotExists
from dtw.stepPattern import StepPattern

class Utils:

    """
    Set of general utilities. This class contains only static methods
    """

    @staticmethod
    def canonicalizeStepPattern(s: object) -> StepPattern:
        """
        Function taken directly from the dtw package due to the
        issues related to importing it from there. It takes StepPattern
        object or its name (string) as an input and returns
        StepPattern as an output

        Parameters
        ---------------
        :param s: Object of class StepPattern or string representing step pattern name

        Raises
        ---------------
        :raise ProvidedStepPatternDoesNotExists: There is no step pattern of such a name

        Returns
        ---------------
        :returns: StepPattern object
        """
        if hasattr(s, "mx"):
            res = s
        else:
            try:
                res = getattr(sys.modules["dtw.stepPattern"], s)
            except AttributeError as ae:
                raise ProvidedStepPatternDoesNotExists(
                    "There is no such step pattern. Please check if there is no typo in the step pattern name."
                )

        return res

    @staticmethod
    def are_objects_of_same_classes(reference_obj: object, other_obj: object) -> bool:
        """
        Check whether two objects are instances of the same class

        Parameters
        ---------------
        :param reference_obj: Reference object
        :param other_obj: Object to compare

        Returns
        ---------------
        :returns: Bool - results of comparison
        """
        are_same_class = isinstance(reference_obj, other_obj.__class__)
        return are_same_class

    @staticmethod
    def get_number_of_dimensions(x: np.ndarray) -> int:
        """
        Get number of numpy array dimensions - indicates
        whether array is 1d, 2d, 3d etc.

        Parameters
        ---------------
        :param x: Numpy arrat

        Returns
        ---------------
        :returns: Int - number of dimension
        """
        return len(x.shape)

    @staticmethod
    def number_of_dimensions_equal(ts_x: np.ndarray, ts_y: np.ndarray) -> bool:
        """
        Check whether two numpy arrays have the same number of dimensions
        (both of them are 1d arrays / 2d arrays, etc.)

        Parameters
        ---------------
        :param ts_x: First array
        :param ts_y: Second array

        Returns
        ---------------
        :returns: Bool - results of check
        """
        ts_x_dim_number = Utils.get_number_of_dimensions(ts_x)
        ts_y_dim_number = Utils.get_number_of_dimensions(ts_y)

        return ts_x_dim_number == ts_y_dim_number

    @staticmethod
    def number_of_series_equal(ts_x: np.ndarray, ts_y: np.ndarray) -> bool:
        """
        Check if number of multidimensional time series matches
        For 1d series returns always true
        For 2d series returns true if number of columns in both arrays is equal.
        For >2d series raises TooManyDimensions exception (ShapeDTW package is able
            to handle only 1d and 2d series)
        If number of dimensions is not equal (for example 1d and 2d arrays was provided)
            raises IncompatibleDimensionality exception

        Parameters
        ---------------
        :param ts_x: First numpy array (time series)
        :param ts_y: Second numpy array (time series)

        Raises
        ---------------
        :raise TooManyDimensions: >2d series was provided - package is able to hanlde only
            1d and 2d series
        :raise IncompatibleDimensionality: Series of different number of dimensions
            were provided (for example 1d and 2d array).

        Returns
        ---------------
        :returns: Bool as a result of check
        """
        ts_x_dim_number = Utils.get_number_of_dimensions(ts_x)
        ts_y_dim_number = Utils.get_number_of_dimensions(ts_y)

        if (ts_x_dim_number > 2) or (ts_y_dim_number > 2):
            raise TooManyDimensions(ts_x, ts_y)

        if not Utils.number_of_dimensions_equal(ts_x, ts_y):
            raise IncompatibleDimensionality(ts_x, ts_y)

        if (ts_x_dim_number == 1) and (ts_y_dim_number == 1):
            return True
        elif (ts_x_dim_number == 2) and (ts_y_dim_number == 2):
            ts_x_n_series = ts_x.shape[1]
            ts_y_n_series = ts_y.shape[1]
            return ts_x_n_series == ts_y_n_series

    @staticmethod
    def is_odd(num: int) -> bool:
        """
        Check if provided number is odd

        Parameters
        ---------------
        :param num: Int number to check

        Returns
        ---------------
        :returns: Bool - a result of check
        """
        return bool(num & 0x1)

