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

## Code taken directly from the https://github.com/DynamicTimeWarping/dtw-python repo and modified
## by adding possibility to plot multivariate variants of shape dtw.


"""DTW plotting functions"""

import math
import matplotlib.pyplot as plt
import numpy as np
from dtw.dtw import DTW
from matplotlib.pyplot import Axes
from matplotlib import collections as mc
from shapedtw.shapedtw import ShapeDTW, MultivariateShapeDTWDependent, MultivariateShapeDTWIndependent
from shapedtw.utils import Utils
from typing import Tuple, List, Union
from matplotlib.gridspec import GridSpecFromSubplotSpec

def dtwPlot(x: Union[DTW, ShapeDTW],
            plot_type: str,
            axis: Axes = None,
            **kwargs):
    # IMPORT_RDOCSTRING plot.dtw
    """Plotting of dynamic time warp results
    Methods for plotting dynamic time warp alignment objects returned by
    [dtw()].
    **Details**
    ``dtwPlot`` displays alignment contained in ``dtw`` objects.
    Various plotting styles are available, passing strings to the ``plot_type``
    argument (may be abbreviated):
    -  ``alignment`` plots the warping curve in ``d``;
    -  ``twoway`` plots a point-by-point comparison, with matching lines;
        see [dtwPlotTwoWay()];
    -  ``threeway`` vis-a-vis inspection of the timeseries and their warping
        curve; see [dtwPlotThreeWay()];
    -  ``density`` displays the cumulative cost landscape with the warping
        path overimposed; see [dtwPlotDensity()]
    Additional parameters are passed to the plotting functions: use with
    care.

    In order to allow users to plot shape dtw results for multidimensional
    time series as well some modifications comparing to the 'dtw'
    package were required. If ``x`` param is an instance of `ShapeDTW`,
    `MultivariateShapeDTWDependent` or `MultivariateShapeDTWIndependent`
    then it is passed to constructor of adequate `ShapeDTWPlot` subclass.
    As a next step ``plot`` method of created object is called using all
    other provided parameters. Internally it uses 'standard' plotting functions
    provided already by the 'dtw' package [dtwPlotAlignment()], [dtwPlotTwoWay()],
    etc. However, those standard functions also required some modification, for
    example ability to pass pyplot 'axis' as a parameter was added.

    In order to keep backward compatibility with 'dtw' package this function
    works as originally when x is a `dtw` object.

    Parameters
    ---------------
    :param: x: `dtw` or `shapedtw` object, result of call to [dtw()] or [shape_dtw()]
    :param plot_type: general style for the plot
    :param axis: pyplot Axes to plot on - for internal usage in case of multivariate
        shape dtw version
    :param kwargs: additional arguments, passed to plotting functions
    """
    # ENDIMPORT

    if isinstance(x, MultivariateShapeDTWDependent):
        return ShapeDTWPlotMultivariateDependent(x).plot(plot_type, **kwargs)
    elif isinstance(x, MultivariateShapeDTWIndependent):
        return ShapeDTWPlotMultivariateIndependent(x).plot(plot_type, **kwargs)
    elif isinstance(x, ShapeDTW):
        return ShapeDTWPlot(x).plot(plot_type, **kwargs)

    if plot_type == "alignment":
        return dtwPlotAlignment(x, axis=axis, **kwargs)
    elif plot_type == "twoway":
        return dtwPlotTwoWay(x, axis=axis, **kwargs)
    elif plot_type == "threeway":
        return dtwPlotThreeWay(x, **kwargs)
    elif plot_type == "density":
        return dtwPlotDensity(x, axis=axis, **kwargs)


def dtwPlotAlignment(d: DTW,
                     axis: Axes=None,
                     xlab: str = "Query index",
                     ylab: str = "Reference index",
                     **kwargs) -> Axes:
    """
    Alignement plot for univariate time series or single dimension of
    multivariate time series

    Parameters
    ---------------
    :param d: dtw results as DTW / ShapeDTW class containing all needed
        metadata (warping paths)
    :param axis: pyplot axis; if None it will be created
    :param xlab: label for query series
    :param ylab: label for reference series
    :param kwargs: additional keyword params which will be passed to the
        ploting function

    Returns
    ---------------
    :return: pyplot Axes object
    """
    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax = axis


    ax.plot(d.index1, d.index2, **kwargs)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if axis is None:
        plt.show()

    return ax


def dtwPlotTwoWay(d: DTW,
                  axis: Axes = None,
                  xts: np.ndarray=None,
                  yts: np.ndarray=None,
                  xoffset: float = None,
                  yoffset: float =None,
                  match_indices: List[int]=None,
                  match_col: str = "gray",
                  xlab: str = "Index",
                  ylab: str = "Query value",
                  **kwargs) -> None:

    """Plotting of dynamic time warp results: pointwise comparison
    Display the query and reference time series and their alignment,
    arranged for visual inspection.

    **Details**
    The two vectors are displayed via the [matplot()] functions; their
    appearance can be customized via the ``type`` and ``pch`` arguments
    (constants or vectors of two elements). If ``xoffset`` or ``yoffset``
    are set, the query or - respectively - reference is shifted vertically
    by the given amount; this will be reflected by the *right-hand* axis.
    Argument ``match_indices`` is used to draw a visual guide to matches; if
    a vector is given, guides are drawn for the corresponding indices in the
    warping curve (match lines). If integer, it is used as the number of
    guides to be plotted. The corresponding style is customized via the
    ``match_col`` argument.
    If ``xts`` and ``yts`` are not supplied, they will be recovered from
    ``d``, as long as it was created with the two-argument call of [dtw()]
    with ``keep_internals=True``.

    Only single-variate time series can be plotted this way. In case of
    multivariate time series this function is called for each dimension
    separately.

    Parameters
    ---------------
    :param d: an alignment result, object of class `dtw`
    :param axis: pyplot axis
    :param xts: query vector
    :param yts: reference vector
    :param xoffset: displacement between the timeseries, summed to query
    :param yoffset: displacement between the timeseries, summed to reference
    :param match_indices: indices for which to draw a visual guide
    :param match_col: color of the match guide lines
    :param xlab: x-axis labels
    :param ylab: y-axis labels
    :param kwargs: additional keyword arguments, passed to `matplot`
    """

    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required. You should provide 'xts' and 'yts' params or set 'keep_internals' argument as 'True' in dtw call")

    if xoffset is not None:
        xts = xts + xoffset

    if yoffset is not None:
        yts = yts + yoffset

    maxlen = max(len(xts), len(yts))
    times = np.arange(maxlen)
    xts = np.pad(xts, (0, maxlen - len(xts)), "constant", constant_values=np.nan)
    yts = np.pad(yts, (0, maxlen - len(yts)), "constant", constant_values=np.nan)

    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax.plot(times, xts, color='k', **kwargs)
    ax.plot(times, yts, **kwargs)

    # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
    if match_indices is None:
        idx = np.linspace(0, len(d.index1) - 1)
    elif not hasattr(match_indices, "__len__"):
        idx = np.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = np.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(d.index1[i], xts[d.index1[i]]),
                    (d.index2[i], yts[d.index2[i]])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    if axis is None:
        plt.show()

def dtwPlotThreeWay(d: DTW,
                    inner_figure: GridSpecFromSubplotSpec = None,
                    dim_num: int = None,
                    xts: np.ndarray=None,
                    yts: np.ndarray=None,
                    match_indices: List[int]=None,
                    match_col: str="gray",
                    xlab: str="Query index",
                    ylab: str="Reference index") -> None:

    """Plotting of dynamic time warp results: annotated warping function
    Display the query and reference time series and their warping curve,
    arranged for visual inspection.

    **Details**
    The query time series is plotted in the bottom panel, with indices
    growing rightwards and values upwards. Reference is in the left panel,
    indices growing upwards and values leftwards. The warping curve panel
    matches indices, and therefore element (1,1) will be at the lower left,
    (N,M) at the upper right.
    Argument ``match_indices`` is used to draw a visual guide to matches; if
    a vector is given, guides are drawn for the corresponding indices in the
    warping curve (match lines). If integer, it is used as the number of
    guides to be plotted. The corresponding style is customized via the
    ``match_col`` argument.
    If ``xts`` and ``yts`` are not supplied, they will be recovered from
    ``d``, as long as it was created with the two-argument call of [dtw()]
    with ``keep_internals=True``.

    Only single-variate time series can be plotted this way. In case of
    multivariate time series this function is called for each dimension
    separately.

    Parameters
    ---------------
    :param d: an alignment result, object of class `dtw`
    :param inner_figure: subgridspec to plot on - only internal usage
        in case of multivariate dtw results
    :param dim_num: number of dimension to put in the plot title - only
        in case of multivariate dtw results
    :param xts: query vector - if not provided there will be an attempt to
        retrieve it from dtw object
    :param yts: reference vector - if not provided there will be an attempt to
        retrieve it from dtw object
    :param match_indices: indices for which to draw a visual guide
    :param match_col: parameter passed to LineCollection used to customize
        lines style
    :param xlab: x-axis label
    :param ylab: y-axis label
    """

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import collections as mc

    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required. You should provide 'xts' and 'yts' params or set 'keep_internals' argument as 'True' in dtw call")

    nn = len(xts)
    mm = len(yts)
    nn1 = np.arange(nn)
    mm1 = np.arange(mm)

    if inner_figure is None:
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 2,
                               width_ratios=[1, 3],
                               height_ratios=[3, 1])

        axr = plt.subplot(gs[0])
        ax = plt.subplot(gs[1])
        axq = plt.subplot(gs[3])
    else:
        axr = plt.subplot(inner_figure[0])
        ax = plt.subplot(inner_figure[1])
        ax.set_title("Dimension " + str(dim_num), fontsize=15)
        axq = plt.subplot(inner_figure[3])

    axq.plot(nn1, xts)  # query, horizontal, bottom
    axq.set_xlabel(xlab)

    axr.plot(yts, mm1)  # ref, vertical
    axr.invert_xaxis()
    axr.set_ylabel(ylab)

    ax.plot(d.index1, d.index2)

    if match_indices is None:
        idx = []
    elif not hasattr(match_indices, "__len__"):
        idx = np.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = np.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(d.index1[i], 0),
                    (d.index1[i], d.index2[i])])
        col.append([(0, d.index2[i]),
                    (d.index1[i], d.index2[i])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    if inner_figure is None:
        plt.show()

def dtwPlotDensity(d: DTW,
                   axis: Axes = None,
                   normalize: bool=False,
                   xlab: str="Query index",
                   ylab: str="Reference index",
                   **kwargs) -> None:

    """Display the cumulative cost density with the warping path overimposed.
    The plot is based on the cumulative cost matrix. It displays the optimal
    alignment as a “ridge” in the global cost landscape.

    **Details**
    The alignment must have been constructed with the
    ``keep_internals=True`` parameter set.
    If ``normalize`` is ``True``, the *average* cost per step is plotted
    instead of the cumulative one. Step averaging depends on the
    [stepPattern()] used.

    Parameters
    ---------------
    :param d: an alignment result, object of class `dtw`
    :param axis: pyplot axis - used only in case of multivariate
        time series
    :param normalize: show per-step average cost instead of cumulative cost
    :param xlab: label for the query axis
    :param ylab: label for the reference axis
    :param kwargs : additional keyword parameters forwarded to plotting functions
    """

    try:
        cm = d.costMatrix
    except:
        raise ValueError("dtwPlotDensity requires dtw internals (set keep.internals=True on dtw() call)")

    if normalize:
        norm = d.stepPattern.hint
        row, col = np.indices(cm.shape)
        if norm == "NA":
            raise ValueError("Step pattern has no normalization")
        elif norm == "N":
            cm = cm / (row + 1)
        elif norm == "N+M":
            cm = cm / (row + col + 2)
        elif norm == "M":
            cm = cm / (col + 1)

    if axis is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        ax = axis

    ax.imshow(cm.T, origin="lower", cmap=plt.get_cmap("terrain"))
    co = ax.contour(cm.T, colors="black", linewidths=1)
    ax.clabel(co)

    ax.plot(d.index1, d.index2, color="blue", linewidth=2)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    if axis is None:
        plt.show()

class ShapeDTWPlot:

    """
    Parent class for subclasses implementing mechanisms allowing to
    plot all types of shape dtw results, including multivariate
    versions (dependent and independent). It contains a set of methods
    responsible for checking number of time series dimensions, preparing
    pyplot axis, etc.

    Attributes
    ---------------
    shape_dtw_results: ShapeDTW:
        object representing results of shape dtw

    Examples
    --------
    >> import numpy as np
    >> from shapedtw.shapedtw import *
    >> from shapedtw.dtwPlot import dtwPlot
    >>
    >> np.random.seed(10)
    >> ts_x = np.random.randn(20)
    >> ts_y = np.random.randn(20)
    >> shape_desc = CompoundDescriptor([SlopeDescriptor(2), PAADescriptor(2)])
    >> shape_dtw_res = shape_dtw(ts_x, ts_y, subsequence_width=2, shape_descriptor=shape_desc, keep_internals=True)
    >> dtwPlot(shape_dtw_res, plot_type = "alignment")
    >> dtwPlot(shape_dtw_tst, plot_type="twoway", xoffset=10)
    >> dtwPlot(shape_dtw_tst, plot_type="threeway")
    >> dtwPlot(dtw_tst, plot_type = "density")
    """

    def __init__(self, shape_dtw_results: ShapeDTW):
        """
        Constructs ShapeDTWPlot object

        Parameters
        ---------------
        :param shape_dtw_results: object representing results of shape dtw
        """
        self.shape_dtw_results = shape_dtw_results

    def _get_figure_nrow(self) -> int:
        """
        Get the number of rows in the plot's grid, assuming
        that there will be 2 columns at max.

        Returns
        ---------------
        :return: number of rows as int
        """
        dim_num = self.shape_dtw_results.ts_x.shape[1]
        res = 1 if dim_num < 2 else math.ceil(dim_num/2)
        return res

    def _get_figure_ncol(self) -> int:
        """
        Get the number of rows in the plot's grid. We use
        one col for univariate time series and two cols
        otherwise.

        Returns
        ---------------
        :return: number of cols as int
        """
        dim_num = self.shape_dtw_results.ts_x.shape[1]
        res = 1 if dim_num < 2 else 2
        return res

    @staticmethod
    def _get_ax_indices(dim_num: int, total_dim_num: int) -> Tuple[int]:
        """
        Get indices of subplot axis on which current
        dimension will be plotted.

        Parameters
        ---------------
        :param dim_num: number of dimension to plot
        :param total_dim_num: number of dimension in given time series

        Returns
        ---------------
        :return: tuple representing axis indices
        """
        row_ind = () if total_dim_num < 3 else (dim_num // 2,)
        col_ind = (dim_num % 2,)

        res = row_ind + col_ind
        return res

    @staticmethod
    def _clean_unnecessary_ax(axis_to_clean: np.ndarray, total_dim_num: int) -> None:
        """
        Clean last axis of subplot if number of multivariate time series
        dimensions is odd

        Parameters
        ---------------
        :param axis_to_clean: array of pyplot Axes
        :param total_dim_num: number of plotted time series dimensions
        """
        ax_ind = ShapeDTWPlot._get_ax_indices(total_dim_num, total_dim_num)
        axis_to_clean[ax_ind].remove()

    def _get_dtw_res_list(self) -> DTW:
        """
        Retrieve dtw results from shapedtw object

        Returns
        ---------------
        :return: dtw results as a DTW object
        """
        res = self.shape_dtw_results._dtw_results[0] \
            if isinstance(self.shape_dtw_results._dtw_results, list) \
            else self.shape_dtw_results._dtw_results
        return res

    def _dtw_plot_alignment(self, **kwargs) -> Axes:
        """
        Plot dtw alignment for univariate time series or
        single dimension of multivariate time series. For
        more details reference to [dtwPlotAlignment()] docs.

        Parameters
        ---------------
        :param kwargs: keyword parameters which will be passed to
            [dtwPlotAlignment()]

        Returns
        ---------------
        :return: pyplot Axes object
        """
        dtw_res = self._get_dtw_res_list()
        return dtwPlotAlignment(dtw_res, **kwargs)

    def _dtw_plot_twoway(self, **kwargs) -> None:
        """
        Render dtw twoway plot for univariate time series or
        single dimension of multivariate time series. For
        more details reference to [dtwPlotTwoWay()] docs.

        Parameters
        ---------------
        :param kwargs: keyword parameters which will be passed to
            [dtwPlotTwoWay()]
        """
        dtw_res = self._get_dtw_res_list()
        return dtwPlotTwoWay(
            dtw_res,
            xts=self.shape_dtw_results.ts_x,
            yts=self.shape_dtw_results.ts_y,
            **kwargs
        )

    def _dtw_plot_threeway(self, **kwargs):
        """
        Render dtw threeway plot for univariate time series or
        single dimension of multivariate time series. For
        more details reference to [dtwPlotThreeWay()] docs.

        Parameters
        ---------------
        :param kwargs: keyword parameters which will be passed to
            [dtwPlotThreeWay()]
        """
        dtw_res = self._get_dtw_res_list()
        return dtwPlotThreeWay(
            dtw_res,
            xts=self.shape_dtw_results.ts_x,
            yts=self.shape_dtw_results.ts_y,
            **kwargs
        )

    def _dtw_plot_density(self, **kwargs):
        """
        Render dtw density plot for univariate time series or
        single dimension of multivariate time series. For
        more details reference to [dtwPlotDensity()] docs.

        Parameters
        ---------------
        :param kwargs: keyword parameters which will be passed to
            [dtwPlotDensity()]
        """
        dtw_res = self._get_dtw_res_list()
        return dtwPlotDensity(
            dtw_res,
            **kwargs
        )

    def plot(self, plot_type: str, **kwargs):
        """
        Render desired plot based on ``plot_type`` parameter
        and provided kwargs

        Parameters
        ---------------
        :param plot_type: type of plot to render
        :param kwargs: keyword arguments which will be passed to
            the proper plotting function
        """
        if plot_type == "alignment":
            return self._dtw_plot_alignment(**kwargs)
        elif plot_type == "twoway":
            return self._dtw_plot_twoway(**kwargs)
        elif plot_type == "threeway":
            return self._dtw_plot_threeway(**kwargs)
        else:
            return self._dtw_plot_density(**kwargs)

class ShapeDTWPlotMultivariateDependent(ShapeDTWPlot):

    """
    Class allowing to render plots for shape dtw results in
    multivariate, dependent version. Due to the fact that in this
    variant all the dimensions share the same warping path then
    alignment and density plot look the same as in case of univariate
    version. However, we can see the difference in twoway and threeway
    plots, due to the fact that we want to visually inspect each
    dimension individually.

    Attributes
    ---------------
    shape_dtw_results: MultivariateShapeDTWDependent:
        multivariate dependent shape dtw results

    Examples
    --------
    >> import numpy as np
    >> from shapedtw.shapedtw import *
    >> from shapedtw.dtwPlot import dtwPlot
    >>
    >> np.random.seed(10)
    >> ts_x_multi = np.random.randn(20, 3)
    >> ts_y_multi = np.random.randn(20, 3)
    >> shape_desc = CompoundDescriptor([SlopeDescriptor(2), PAADescriptor(2)])
    >> shape_dtw_res_dependent = shape_dtw(
    >>      ts_x_multi,
    >>      ts_y_multi,
    >>      subsequence_width=2,
    >>      shape_descriptor=shape_desc,
    >>      multivariate_version="dependent",
    >>      keep_internals=True
    >> )
    >> dtwPlot(shape_dtw_res_dependent, plot_type = "alignment")
    >> dtwPlot(shape_dtw_res_dependent, plot_type="twoway", xoffset=10)
    >> dtwPlot(shape_dtw_res_dependent, plot_type="threeway")
    >> dtwPlot(shape_dtw_res_dependent, plot_type = "density")
    """

    def __init__(self, shape_dtw_results: MultivariateShapeDTWDependent):
        """
        Constructs ShapeDTWPlotMultivariateDependent object

        Parameters
        ---------------
        :param shape_dtw_results: multivariate dependent shape dtw results
        """
        super().__init__(shape_dtw_results)

    def _dtw_plot_twoway(self, fig_width=8, fig_height=5, **kwargs) -> None:
        """
        Render twoway plot adapted for the multivariate case. It constructs
        a grid on which each dimension is plotted separately. For more details
        see [dtwPlotTwoWay()] docs

        Parameters
        ---------------
        :param fig_width: width of the whole plot
        :param fig_height: height of the whole plot
        :param kwargs: additional keywords parameters which will be passed
            to the plotting function
        """
        total_dim_num = self.shape_dtw_results.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_twoway(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(ncols=fig_ncol, nrows=fig_nrow, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):
            ax_ind = self._get_ax_indices(dim_number, total_dim_num)

            dtwPlotTwoWay(
                self.shape_dtw_results._dtw_results,
                xts=self.shape_dtw_results.ts_x[:, dim_number],
                yts=self.shape_dtw_results.ts_y[:, dim_number],
                axis=ax[ax_ind],
                **kwargs
            )
            ax[ax_ind].set_title("Dimension " + str(dim_number+1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def _dtw_plot_threeway(self, fig_width=7, fig_height=7, **kwargs):
        """
        Render threeway plot adapted for the multivariate case. It constructs
        a grid on which each dimension is plotted separately. For more details
        see [dtwPlotThreeWay()] docs

        Parameters
        ---------------
        :param fig_width: width of the whole plot
        :param fig_height: height of the whole plot
        :param kwargs: additional keywords parameters which will be passed
            to the plotting function
        """
        total_dim_num = self.shape_dtw_results.ts_x.shape[1]

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
                self.shape_dtw_results._dtw_results,
                xts=self.shape_dtw_results.ts_x[:, dim_number],
                yts=self.shape_dtw_results.ts_y[:, dim_number],
                inner_figure=inner,
                dim_num=dim_number+1,
                **kwargs
            )

        plt.show()

class ShapeDTWPlotMultivariateIndependent(ShapeDTWPlot):
    """
    Class allowing to render plots for shape dtw results in
    multivariate, independent version. Due to the fact that in this
    variant each dimension has his own warping path assigned then
    all types of plots looks different from univariate case and
    require to construct grids with number of fields equal to the
    number of time series dimensions.

    Attributes
    ---------------
    shape_dtw_results: ShapeDTWPlotMultivariateIndependent:
        multivariate independent shape dtw results

    Examples
    --------
    >> import numpy as np
    >> from shapedtw.shapedtw import *
    >> from shapedtw.dtwPlot import dtwPlot
    >>
    >> np.random.seed(10)
    >> ts_x_multi = np.random.randn(20, 3)
    >> ts_y_multi = np.random.randn(20, 3)
    >> shape_desc = CompoundDescriptor([SlopeDescriptor(2), PAADescriptor(2)])
    >> shape_dtw_res_independent = shape_dtw(
    >>      ts_x_multi,
    >>      ts_y_multi,
    >>      subsequence_width=2,
    >>      shape_descriptor=shape_desc,
    >>      multivariate_version="independent",
    >>      keep_internals=True
    >> )
    >> dtwPlot(shape_dtw_res_independent, plot_type = "alignment")
    >> dtwPlot(shape_dtw_res_independent, plot_type="twoway", xoffset=10)
    >> dtwPlot(shape_dtw_res_independent, plot_type="threeway")
    >> dtwPlot(shape_dtw_res_independent, plot_type = "density")
    """
    def __init__(self, shape_dtw_results: MultivariateShapeDTWIndependent):
        """
        Constructs ShapeDTWPlotMultivariateIndependent object

        Parameters
        ---------------
        :param shape_dtw_results: multivariate independent shape dtw results
        """
        super().__init__(shape_dtw_results)

    def _dtw_plot_alignment(self, fig_width=6, fig_height=5, **kwargs):
        """
        Render alignment plot adapted for the multivariate case. It constructs
        a grid on which each dimension is plotted separately. For more details
        see [dtwPlotAlignment()] docs

        Parameters
        ---------------
        :param fig_width: width of the whole plot
        :param fig_height: height of the whole plot
        :param kwargs: additional keywords parameters which will be passed
            to the plotting function
        """

        total_dim_num = self.shape_dtw_results.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_alignment(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(nrows=fig_nrow, ncols=fig_ncol, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):
            ax_ind = self._get_ax_indices(dim_number, total_dim_num)
            dtwPlotAlignment(self.shape_dtw_results._dtw_results[dim_number], axis=ax[ax_ind], **kwargs)
            ax[ax_ind].set_title("Dimension " + str(dim_number + 1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.4)
        plt.show()

    def _dtw_plot_twoway(self, fig_width=8, fig_height=5, **kwargs):
        """
        Render twoway plot adapted for the multivariate case. It constructs
        a grid on which each dimension is plotted separately. For more details
        see [dtwPlotTwoWay()] docs

        Parameters
        ---------------
        :param fig_width: width of the whole plot
        :param fig_height: height of the whole plot
        :param kwargs: additional keywords parameters which will be passed
            to the plotting function
        """
        total_dim_num = self.shape_dtw_results.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_twoway(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(nrows=fig_nrow, ncols=fig_ncol, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):

            ax_ind = self._get_ax_indices(dim_number, total_dim_num)

            dtwPlotTwoWay(
                self.shape_dtw_results._dtw_results[dim_number],
                xts=self.shape_dtw_results.ts_x[:, dim_number],
                yts=self.shape_dtw_results.ts_y[:, dim_number],
                axis=ax[ax_ind],
                **kwargs
            )
            ax[ax_ind].set_title("Dimension " + str(dim_number + 1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.4)
        plt.show()


    def _dtw_plot_threeway(self, fig_width=7, fig_height=7, **kwargs):

        """
        Render threeway plot adapted for the multivariate case. It constructs
        a grid on which each dimension is plotted separately. For more details
        see [dtwPlotThreeWay()] docs

        Parameters
        ---------------
        :param fig_width: width of the whole plot
        :param fig_height: height of the whole plot
        :param kwargs: additional keywords parameters which will be passed
            to the plotting function
        """
        total_dim_num = self.shape_dtw_results.ts_x.shape[1]

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
                self.shape_dtw_results._dtw_results[dim_number],
                xts=self.shape_dtw_results.ts_x[:, dim_number],
                yts=self.shape_dtw_results.ts_y[:, dim_number],
                inner_figure=inner,
                dim_num=dim_number + 1,
                **kwargs
            )

        plt.show()

    def _dtw_plot_density(self, fig_width=5, fig_height=6, **kwargs):
        """
        Render density plot adapted for the multivariate case. It constructs
        a grid on which each dimension is plotted separately. For more details
        see [dtwPlotDensity()] docs

        Parameters
        ---------------
        :param fig_width: width of the whole plot
        :param fig_height: height of the whole plot
        :param kwargs: additional keywords parameters which will be passed
            to the plotting function
        """
        total_dim_num = self.shape_dtw_results.ts_x.shape[1]

        if total_dim_num == 1:
            return super()._dtw_plot_density(**kwargs)

        fig_nrow = self._get_figure_nrow()
        fig_ncol = self._get_figure_ncol()

        fig, ax = plt.subplots(nrows=fig_nrow, ncols=fig_ncol, figsize=(fig_width*fig_ncol, fig_nrow*fig_height))

        for dim_number in range(total_dim_num):
            ax_ind = self._get_ax_indices(dim_number, total_dim_num)
            dtwPlotDensity(
                self.shape_dtw_results._dtw_results[dim_number],
                axis=ax[ax_ind],
                **kwargs
            )
            ax[ax_ind].set_title("Dimension " + str(dim_number + 1), fontsize=15)

        if Utils.is_odd(total_dim_num):
            self._clean_unnecessary_ax(ax, total_dim_num)

        plt.subplots_adjust(hspace=0.1)
        plt.show()
