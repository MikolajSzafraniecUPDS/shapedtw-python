from __future__ import annotations

from dtw import *
import math

from shapedtw.preprocessing import *
from shapedtw.shapeDescriptors import *
from .utils import Utils
from dataclasses import dataclass
from .dtwPlot import *


class StepPatternMatrixTransformator:

    """
    Class for transforming step pattern matrix to more convenient
    form of dictionary
    """

    def __init__(self, step_pattern: str):
        self.step_pattern = step_pattern
        self.step_pattern_matrix = self._get_step_pattern_matrix()

    def _get_step_pattern_matrix(self) -> ndarray:
        step_pattern_obj = Utils.canonicalizeStepPattern(self.step_pattern)
        return step_pattern_obj.mx

    def _get_segments_number(self) -> int:
        return int(self.step_pattern_matrix[:, 0].max())

    def _get_matrix_segment(self, segment_number: int) -> ndarray:
        ind_mask = self.step_pattern_matrix[:, 0] == segment_number
        return self.step_pattern_matrix[ind_mask, :].copy()

    def _get_segment_length(self, segment_number: int) -> int:
        mask = self.step_pattern_matrix[:, 0] == segment_number
        return mask.sum()-1

    def _get_segment_pattern(self, segment_number: int) -> tuple:
        segment = self._get_matrix_segment(segment_number)
        return int(segment[0, 1]), int(segment[0, 2])

    def _segment_to_dict(self, segment_num):
        segment = self._get_matrix_segment(segment_num)
        segment_length = self._get_segment_length(segment_num)
        res = {
            i: {
                "x_index": int(segment[i+1, 1]),
                "y_index": int(segment[i+1, 2]),
                "weight": segment[i+1, 3]
            } for i in range(segment_length)
        }

        return res

    def step_pattern_matrix_to_dict(self):
        segments_number = self._get_segments_number()
        segments_iter = range(1, segments_number+1)
        res = {
            self._get_segment_pattern(i): self._segment_to_dict(i)
            for i in segments_iter
        }

        return res


class DistanceReconstructor:

    """
    Class for reconstructing raw time series distance based on its
    distance matrix and warping path
    """

    def __init__(self,
                 step_pattern: str,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 ts_x_wp: list,
                 ts_y_wp: list,
                 dist_method: str = "euclidean"):

        self.ts_x = ts_x
        self.ts_y = ts_y
        self.ts_x_warping_path = ts_x_wp
        self.ts_y_warping_path = ts_y_wp
        self.dist_method = dist_method
        self.distance_matrix = self._calc_distance_matrix()
        self.step_pattern_dictionary = StepPatternMatrixTransformator(step_pattern).step_pattern_matrix_to_dict()

    def _calc_distance_matrix(self):
        dist_matrix = DistanceMatrixCalculator(self.ts_x, self.ts_y, self.dist_method).\
            calc_distance_matrix()
        return dist_matrix

    def _calc_single_distance(self, x_index: int, y_index: int, single_pattern_dict: dict) -> float:
        target_x_index = x_index - single_pattern_dict["x_index"]
        target_y_index = y_index - single_pattern_dict["y_index"]
        distance = self.distance_matrix[target_x_index, target_y_index] * single_pattern_dict["weight"]
        return distance

    def _calc_distance_for_given_pattern(self, x_index: int, y_index: int, pattern: tuple) -> float:
        pattern_segment = self.step_pattern_dictionary[pattern]
        pattern_distance = sum([
            self._calc_single_distance(x_index, y_index, pattern_segment[key])
            for key in pattern_segment
        ])
        return pattern_distance

    def _get_indices_pairs(self):
        return list(
            zip(
                self.ts_x_warping_path,
                self.ts_y_warping_path
            )
        )

    def _get_indices_patterns(self):
        x_indices_diff = self.ts_x_warping_path[1:] - self.ts_x_warping_path[:-1]
        y_indices_diff = self.ts_y_warping_path[1:] - self.ts_y_warping_path[:-1]

        return list(zip(x_indices_diff, y_indices_diff))

    def calc_raw_ts_distance(self):
        indices_pairs = self._get_indices_pairs()
        indices_patterns = self._get_indices_patterns()
        raw_series_distance = self.distance_matrix[0, 0]
        distances_list = [
            self._calc_distance_for_given_pattern(x_ind, y_ind, indices_patterns[i])
            for (i, (x_ind, y_ind)) in enumerate(indices_pairs[1:])
        ]

        raw_series_distance += sum(distances_list)

        return raw_series_distance


@dataclass
class ShapeDTWResults:

    distance: float
    normalized_distance: float
    shape_distance: float
    shape_normalized_distance: float


class ShapeDTW:

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_res: DTW | List[DTW] = None,
                 shape_dtw_results: ShapeDTWResults = None):

        self._dtw_results = dtw_res
        self.ts_x = ts_x
        self.ts_y = ts_y
        self._step_pattern = step_pattern
        self._dist_method = dist_method
        self._shape_dtw_results = shape_dtw_results

    def _calc_raw_series_distance(self, dist_method: str = "euclidean"):
        dist_reconstructor = DistanceReconstructor(
            step_pattern=self._step_pattern,
            ts_x=self.ts_x,
            ts_y=self.ts_y,
            ts_x_wp=self._dtw_results.index1s,
            ts_y_wp=self._dtw_results.index2s,
            dist_method=dist_method
        )

        return dist_reconstructor.calc_raw_ts_distance()

    def _calc_raw_series_normalized_distance(self, distance: float):
        step_pattern = Utils.canonicalizeStepPattern(self._step_pattern)
        norm = step_pattern.hint

        n, m = len(self.ts_x), len(self.ts_y)

        if norm == "N+M":
            normalized_distance = distance / (n + m)
        elif norm == "N":
            normalized_distance = distance / n
        elif norm == "M":
            normalized_distance = distance / m
        else:
            normalized_distance = np.nan

        return normalized_distance

    def _calc_distances(self):
        distance = self._calc_raw_series_distance(self._dist_method)
        normalized_distance = self._calc_raw_series_normalized_distance(distance)
        shape_distance = self._dtw_results.distance
        shape_normalized_distance = self._dtw_results.normalizedDistance

        return ShapeDTWResults(
            distance, normalized_distance,
            shape_distance, shape_normalized_distance
        )

    def _get_distance(self):
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_normalized_distance(self):
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.normalized_distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_shape_descriptor_distance(self):
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.shape_distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _get_shape_descriptor_normalized_distance(self):
        if self._shape_dtw_results is not None:
            return self._shape_dtw_results.shape_normalized_distance
        else:
            raise ShapeDTWNotCalculatedYet()

    def _set_distance(self, value):
        raise DistanceSettingNotPossible(
            "ShapeDTW distance can be set only using 'calc_shape_dtw' method"
        )

    distance = property(_get_distance, _set_distance)
    normalized_distance = property(_get_normalized_distance, _set_distance)
    shape_distance = property(_get_shape_descriptor_distance, _set_distance)
    shape_normalized_distance = property(_get_shape_descriptor_normalized_distance, _set_distance)

    def plot(self, plot_type, **kwargs):
        if plot_type == "alignment":
            return self._dtw_plot_alignment(**kwargs)
        elif plot_type == "twoway":
            return self._dtw_plot_twoway(**kwargs)
        elif plot_type == "threeway":
            return self._dtw_plot_threeway(**kwargs)
        else:
            return self._dtw_plot_density(**kwargs)

    def _get_figure_nrow(self):
        dim_num = self.ts_x.shape[1]
        res = 1 if dim_num < 2 else math.ceil(dim_num/2)
        return res

    def _get_figure_ncol(self):
        dim_num = self.ts_x.shape[1]
        res = 1 if dim_num < 2 else 2
        return res

    @staticmethod
    def _get_ax_indices(dim_num: int, total_dim_num: int):
        row_ind = () if total_dim_num < 3 else (dim_num // 2, )
        col_ind = (dim_num % 2, )

        res = row_ind + col_ind
        return res

    @staticmethod
    def _clean_unnecessary_ax(axis_to_clean, total_dim_num: int):
        ax_ind = ShapeDTW._get_ax_indices(total_dim_num, total_dim_num)
        axis_to_clean[ax_ind].remove()

    def _get_dtw_res_list(self):
        res = self._dtw_results[0] if isinstance(self._dtw_results, list) else self._dtw_results
        return res

    def _dtw_plot_alignment(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotAlignment(dtw_res, **kwargs)

    def _dtw_plot_twoway(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotTwoWay(
            dtw_res,
            xts=self.ts_x,
            yts=self.ts_y,
            **kwargs
        )

    def _dtw_plot_threeway(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotThreeWay(
            dtw_res,
            xts=self.ts_x,
            yts=self.ts_y,
            **kwargs
        )

    def _dtw_plot_density(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotDensity(
            dtw_res,
            **kwargs
        )


class UnivariateShapeDTW(ShapeDTW):

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_results: DTW = None):

        super().__init__(ts_x, ts_y, step_pattern, dist_method, dtw_results)

    def calc_shape_dtw(self,
                       subsequence_width: int,
                       shape_descriptor: ShapeDescriptor,
                       **kwargs):

        ts_x_shape_descriptor = UnivariateSubsequenceBuilder(self.ts_x, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        ts_y_shape_descriptor = UnivariateSubsequenceBuilder(self.ts_y, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        distance_matrix = ts_x_shape_descriptor.calc_distance_matrix(
            ts_y_shape_descriptor, dist_method=self._dist_method
        )

        dtw_results = dtw(distance_matrix.dist_matrix,
                          step_pattern=self._step_pattern,
                          **kwargs)

        self._dtw_results = dtw_results
        self._shape_dtw_results = self._calc_distances()

        return self


class MultivariateShapeDTWDependent(ShapeDTW):

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_results: DTW = None):

        super().__init__(ts_x, ts_y, step_pattern, dist_method, dtw_results)

    def calc_shape_dtw(self,
                       subsequence_width: int,
                       shape_descriptor: ShapeDescriptor,
                       **kwargs):

        ts_x_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_x, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        ts_y_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_y, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        dist_matrix = ts_x_shape_descriptor.calc_summed_distance_matrix(
            ts_y_shape_descriptor, dist_method=self._dist_method
        )

        dtw_results = dtw(dist_matrix.distance_matrix, **kwargs)
        self._dtw_results = dtw_results
        self._shape_dtw_results = self._calc_distances()

        return self

    def _dtw_plot_twoway(self, fig_width=8, fig_height=5, **kwargs):

        total_dim_num = self.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_twoway(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(ncols=fig_ncol, nrows=fig_nrow, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):
            ax_ind = self._get_ax_indices(dim_number, total_dim_num)

            dtwPlotTwoWay(
                self._dtw_results,
                xts=self.ts_x[:, dim_number],
                yts=self.ts_y[:, dim_number],
                axis=ax[ax_ind],
                **kwargs
            )
            ax[ax_ind].set_title("Dimension " + str(dim_number+1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def _dtw_plot_threeway(self, fig_width=7, fig_height=7, **kwargs):
        total_dim_num = self.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_threeway(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig = plt.figure(figsize=(fig_width*fig_ncol, fig_nrow*fig_height), constrained_layout=True)
        outer_fig = fig.add_gridspec(nrows=fig_nrow, ncols=fig_ncol, height_ratios=[1]*fig_nrow, hspace=2)

        for dim_number in range(total_dim_num):

            ax_ind = self._get_ax_indices(dim_number, total_dim_num)
            # Operation necessary due to the 'Unrecognized subplot spec' error
            # for 2-dimensional time series
            if len(ax_ind) == 1:
                ax_ind = ax_ind[0]

            inner = outer_fig[ax_ind].subgridspec(
                2, 2,
                width_ratios=[1, 3],
                height_ratios=[3, 1]
            )

            dtwPlotThreeWay(
                self._dtw_results,
                xts=self.ts_x[:, dim_number],
                yts=self.ts_y[:, dim_number],
                inner_figure=inner,
                dim_num=dim_number+1,
                **kwargs
            )

        # if Utils.is_odd(total_dim_num):
        #     self._clean_unnecessary_ax(outer_fig, total_dim_num)

        plt.show()


class MultivariateShapeDTWIndependent(ShapeDTW):

    def __init__(self,
                 ts_x: ndarray,
                 ts_y: ndarray,
                 step_pattern: str = "symmetric2",
                 dist_method: str = "euclidean",
                 dtw_results: List[DTW] = None):

        super().__init__(ts_x, ts_y, step_pattern, dist_method, dtw_results)

    def _calc_raw_series_distance(self, dist_method: str = "euclidean"):
        n_dim = self.ts_x.shape[1]
        dist_reconstructors = [
            DistanceReconstructor(step_pattern=self._step_pattern,
                                  ts_x=self.ts_x[:, ind].copy(),
                                  ts_y=self.ts_y[:, ind].copy(),
                                  ts_x_wp=self._dtw_results[ind].index1s,
                                  ts_y_wp=self._dtw_results[ind].index2s,
                                  dist_method=self._dist_method)
            for ind in range(n_dim)
        ]

        distances = [dist_recon.calc_raw_ts_distance() for dist_recon in dist_reconstructors]
        res = sum(distances)

        return res

    def _calc_distances(self):
        distance = self._calc_raw_series_distance(self._dist_method)
        normalized_distance = self._calc_raw_series_normalized_distance(distance)
        shape_distance = sum([dtw_res.distance for dtw_res in self._dtw_results])
        shape_normalized_distance = sum([dtw_res.normalizedDistance for dtw_res in self._dtw_results])

        return ShapeDTWResults(
            distance, normalized_distance,
            shape_distance, shape_normalized_distance
        )

    def calc_shape_dtw(self,
                       subsequence_width: int,
                       shape_descriptor: ShapeDescriptor,
                       **kwargs):

        ts_x_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_x, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        ts_y_shape_descriptor = MultivariateSubsequenceBuilder(self.ts_y, subsequence_width). \
            transform_time_series_to_subsequences(). \
            get_shape_descriptors(shape_descriptor)

        dist_matrices = ts_x_shape_descriptor.calc_distance_matrices(
            ts_y_shape_descriptor, dist_method=self._dist_method
        )

        dtw_results_list = [
            dtw(dist_mat.dist_matrix, **kwargs)
            for dist_mat in dist_matrices.distance_matrices_list
        ]

        self._dtw_results = dtw_results_list
        self._shape_dtw_results = self._calc_distances()

        return self

    def _dtw_plot_alignment(self, fig_width=6, fig_height=5, **kwargs):

        total_dim_num = self.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_alignment(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(nrows=fig_nrow, ncols=fig_ncol, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):
            ax_ind = self._get_ax_indices(dim_number, total_dim_num)
            dtwPlotAlignment(self._dtw_results[dim_number], axis=ax[ax_ind], **kwargs)
            ax[ax_ind].set_title("Dimension " + str(dim_number + 1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def _dtw_plot_twoway(self, fig_width=8, fig_height=5, **kwargs):

        total_dim_num = self.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_twoway(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(nrows=fig_nrow, ncols=fig_ncol, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):

            ax_ind = self._get_ax_indices(dim_number, total_dim_num)

            dtwPlotTwoWay(
                self._dtw_results[dim_number],
                xts=self.ts_x[:, dim_number],
                yts=self.ts_y[:, dim_number],
                axis=ax[ax_ind],
                **kwargs
            )
            ax[ax_ind].set_title("Dimension " + str(dim_number + 1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.4)
        plt.show()


    def _dtw_plot_threeway(self, fig_width=7, fig_height=7, **kwargs):

        total_dim_num = self.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_threeway(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig = plt.figure(figsize=(fig_width*fig_ncol, fig_nrow*fig_height), constrained_layout=True)
        outer_fig = fig.add_gridspec(nrows=fig_nrow, ncols=fig_ncol, height_ratios=[1]*fig_nrow, hspace=2)

        for dim_number in range(total_dim_num):

            ax_ind = self._get_ax_indices(dim_number, total_dim_num)
            # Operation necessary due to the 'Unrecognized subplot spec' error
            # for 2-dimensional time series
            if len(ax_ind) == 1:
                ax_ind = ax_ind[0]

            inner = outer_fig[ax_ind].subgridspec(
                2, 2,
                width_ratios=[1, 3],
                height_ratios=[3, 1]
            )

            dtwPlotThreeWay(
                self._dtw_results[dim_number],
                xts=self.ts_x[:, dim_number],
                yts=self.ts_y[:, dim_number],
                inner_figure=inner,
                dim_num=dim_number + 1,
                **kwargs
            )

        # if Utils.is_odd(total_dim_num):
        #     self._clean_unnecessary_ax(outer_fig, total_dim_num)

        plt.show()

    def _dtw_plot_density(self, fig_width=5, fig_height=6, **kwargs):

        total_dim_num = self.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_density(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(nrows=fig_nrow, ncols=fig_ncol, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):
            ax_ind = self._get_ax_indices(dim_number, total_dim_num)
            dtwPlotDensity(
                self._dtw_results[dim_number],
                axis=ax[ax_ind],
                **kwargs
            )
            ax[ax_ind].set_title("Dimension " + str(dim_number + 1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.1)
        plt.show()


def shape_dtw(x: ndarray, y: ndarray,
              subsequence_width: int,
              shape_descriptor: ShapeDescriptor,
              step_pattern: str = "symmetric2",
              dist_method="euclidean",
              multivariate_version: str = "dependent",
              **kwargs):

    Utils.verify_shape_compatibility(ts_x=x, ts_y=y)

    ts_x_shape = len(x.shape)

    if ts_x_shape == 1:
        shape_dtw_obj = UnivariateShapeDTW(
            ts_x=x, ts_y=y,
            step_pattern=step_pattern,
            dist_method=dist_method
        )
    elif multivariate_version == "dependent":
        shape_dtw_obj = MultivariateShapeDTWDependent(
            ts_x=x, ts_y=y,
            step_pattern=step_pattern,
            dist_method=dist_method
        )
    else:
        shape_dtw_obj = MultivariateShapeDTWIndependent(
            ts_x=x, ts_y=y,
            step_pattern=step_pattern,
            dist_method=dist_method
        )

    shape_dtw_results = shape_dtw_obj.calc_shape_dtw(
        subsequence_width=subsequence_width,
        shape_descriptor=shape_descriptor,
        **kwargs
    )

    return shape_dtw_results
