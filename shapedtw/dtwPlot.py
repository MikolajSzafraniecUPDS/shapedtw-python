## Code taken directly from the https://github.com/DynamicTimeWarping/dtw-python repo and slightly adapted
## by adding possibility to pass pyplot ax to the plotting functions as parameters. It had to be done in
## order to generate multiple plots at the same time for the purpose of multivariate DTW.

##
## Copyright (c) 2006-2019 of Toni Giorgino
##
## This file is part of the DTW package.
##
## DTW is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DTW is distributed in the hope that it will be useful, but WITHOUT
## ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
## or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
## License for more details.
##
## You should have received a copy of the GNU General Public License
## along with DTW.  If not, see <http://www.gnu.org/licenses/>.
##

"""DTW plotting functions"""

import numpy
import math
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from shapedtw.shapedtw import ShapeDTW, MultivariateShapeDTWDependent, MultivariateShapeDTWIndependent
from shapedtw.utils import Utils

def dtwPlot(x, type, axis = None, **kwargs):
    # IMPORT_RDOCSTRING plot.dtw
    """Plotting of dynamic time warp results
Methods for plotting dynamic time warp alignment objects returned by
[dtw()].
**Details**
``dtwPlot`` displays alignment contained in ``dtw`` objects.
Various plotting styles are available, passing strings to the ``type``
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
Parameters
----------
x,d :
    `dtw` object, usually result of call to [dtw()]
xlab :
    label for the query axis
ylab :
    label for the reference axis
type :
    general style for the plot, see below
plot_type :
    type of line to be drawn, used as the `type` argument in the underlying `plot` call
... :
    additional arguments, passed to plotting functions
"""
    # ENDIMPORT

    if isinstance(x, MultivariateShapeDTWDependent):
        return ShapeDTWPlotMultivariateDependent(x).plot(type, **kwargs)
    elif isinstance(x, MultivariateShapeDTWIndependent):
        return ShapeDTWPlotMultivariateIndependent(x).plot(type, **kwargs)
    elif isinstance(x, ShapeDTW):
        return ShapeDTWPlot(x).plot(type, **kwargs)

    if type == "alignment":
        return dtwPlotAlignment(x, axis=axis, **kwargs)
    elif type == "twoway":
        return dtwPlotTwoWay(x, axis=axis, **kwargs)
    elif type == "threeway":
        return dtwPlotThreeWay(x, axis=axis, **kwargs)
    elif type == "density":
        return dtwPlotDensity(x, axis=axis, **kwargs)


def dtwPlotAlignment(d, axis=None, xlab="Query index", ylab="Reference index", **kwargs):

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


def dtwPlotTwoWay(d, axis = None,
                  xts=None, yts=None,
                  xoffset = None, yoffset=None,
                  offset=0,
                  ts_type="l",
                  match_indices=None,
                  match_col="gray",
                  xlab="Index",
                  ylab="Query value",
                  **kwargs):
    # IMPORT_RDOCSTRING dtwPlotTwoWay
    """Plotting of dynamic time warp results: pointwise comparison
Display the query and reference time series and their alignment,
arranged for visual inspection.
**Details**
The two vectors are displayed via the [matplot()] functions; their
appearance can be customized via the ``type`` and ``pch`` arguments
(constants or vectors of two elements). If ``offset`` is set, the
reference is shifted vertically by the given amount; this will be
reflected by the *right-hand* axis.
Argument ``match_indices`` is used to draw a visual guide to matches; if
a vector is given, guides are drawn for the corresponding indices in the
warping curve (match lines). If integer, it is used as the number of
guides to be plotted. The corresponding style is customized via the
``match_col`` and ``match_lty`` arguments.
If ``xts`` and ``yts`` are not supplied, they will be recovered from
``d``, as long as it was created with the two-argument call of [dtw()]
with ``keep_internals=True``. Only single-variate time series can be
plotted this way.
Parameters
----------
d :
    an alignment result, object of class `dtw`
xts :
    query vector
yts :
    reference vector
xlab,ylab :
    axis labels
offset :
    displacement between the timeseries, summed to reference
match_col,match_lty :
    color and line type of the match guide lines
match_indices :
    indices for which to draw a visual guide
ts_type,pch :
    graphical parameters for timeseries plotting, passed to `matplot`
... :
    additional arguments, passed to `matplot`
Notes
-----
When ``offset`` is set values on the left axis only apply to the query.
"""
    # ENDIMPORT

    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required")

    if xoffset is not None:
        xts = xts + xoffset

    if yoffset is not None:
        yts = yts + yoffset

    # ytso = yts + offset
    # offset = -offset

    maxlen = max(len(xts), len(yts))
    times = numpy.arange(maxlen)
    xts = numpy.pad(xts, (0, maxlen - len(xts)), "constant", constant_values=numpy.nan)
    yts = numpy.pad(yts, (0, maxlen - len(yts)), "constant", constant_values=numpy.nan)

    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
    # if offset != 0:
    #     ax2 = ax.twinx()
    #     ax2.tick_params('y', colors='b')
    # else:
    #     ax2 = ax

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    ax.plot(times, xts, color='k', **kwargs)
    ax.plot(times, yts, **kwargs)

    ql, qh = ax.get_ylim()
    rl, rh = ax.get_ylim()

    # if offset > 0:
    #     ax.set_ylim(ql - offset, qh)
    #     ax2.set_ylim(rl, rh + offset)
    # elif offset < 0:
    #     ax.set_ylim(ql, qh - offset)
    #     ax2.set_ylim(rl + offset, rh)

    # https://stackoverflow.com/questions/21352580/matplotlib-plotting-numerous-disconnected-line-segments-with-different-colors
    if match_indices is None:
        idx = numpy.linspace(0, len(d.index1) - 1)
    elif not hasattr(match_indices, "__len__"):
        idx = numpy.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = numpy.array(idx).astype(int)

    col = []
    for i in idx:
        col.append([(d.index1[i], xts[d.index1[i]]),
                    (d.index2[i], -offset + yts[d.index2[i]])])

    lc = mc.LineCollection(col, linewidths=1, linestyles=":", colors=match_col)
    ax.add_collection(lc)

    if axis is None:
        plt.show()
    # return ax

def dtwPlotThreeWay(d, inner_figure = None,
                    dim_num=None,
                    xts=None, yts=None,
                    match_indices=None,
                    match_col="gray",
                    xlab="Query index",
                    ylab="Reference index", **kwargs):
    # IMPORT_RDOCSTRING dtwPlotThreeWay
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
``match_col`` and ``match_lty`` arguments.
If ``xts`` and ``yts`` are not supplied, they will be recovered from
``d``, as long as it was created with the two-argument call of [dtw()]
with ``keep_internals=True``. Only single-variate time series can be
plotted.
Parameters
----------
d :
    an alignment result, object of class `dtw`
xts :
    query vector
yts :
    reference vector
xlab :
    label for the query axis
ylab :
    label for the reference axis
main :
    main title
type_align :
    line style for warping curve plot
type_ts :
    line style for timeseries plot
match_indices :
    indices for which to draw a visual guide
margin :
    outer figure margin
inner_margin :
    inner figure margin
title_margin :
    space on the top of figure
... :
    additional arguments, used for the warping curve
"""
    # ENDIMPORT
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib import collections as mc

    if xts is None or yts is None:
        try:
            xts = d.query
            yts = d.reference
        except:
            raise ValueError("Original timeseries are required")

    nn = len(xts)
    mm = len(yts)
    nn1 = numpy.arange(nn)
    mm1 = numpy.arange(mm)

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

    # axr = plt.subplot(gs[0])
    # ax = plt.subplot(gs[1])
    # axq = plt.subplot(gs[3])

    axq.plot(nn1, xts)  # query, horizontal, bottom
    axq.set_xlabel(xlab)

    axr.plot(yts, mm1)  # ref, vertical
    axr.invert_xaxis()
    axr.set_ylabel(ylab)

    ax.plot(d.index1, d.index2)

    if match_indices is None:
        idx = []
    elif not hasattr(match_indices, "__len__"):
        idx = numpy.linspace(0, len(d.index1) - 1, num=match_indices)
    else:
        idx = match_indices
    idx = numpy.array(idx).astype(int)

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
    # return ax

def dtwPlotDensity(d, axis = None,
                   normalize=False,
                   xlab="Query index",
                   ylab="Reference index", **kwargs):
    # IMPORT_RDOCSTRING dtwPlotDensity
    """Display the cumulative cost density with the warping path overimposed
The plot is based on the cumulative cost matrix. It displays the optimal
alignment as a “ridge” in the global cost landscape.
**Details**
The alignment must have been constructed with the
``keep_internals=True`` parameter set.
If ``normalize`` is ``True``, the *average* cost per step is plotted
instead of the cumulative one. Step averaging depends on the
[stepPattern()] used.
Parameters
----------
d :
    an alignment result, object of class `dtw`
normalize :
    show per-step average cost instead of cumulative cost
xlab :
    label for the query axis
ylab :
    label for the reference axis
... :
    additional parameters forwarded to plotting functions
Examples
--------
>>> from dtw import *
A study of the "Itakura" parallelogram
A widely held misconception is that the "Itakura parallelogram" (as
described in the original article) is a global constraint.  Instead,
it arises from local slope restrictions. Anyway, an "itakuraWindow",
is provided in this package. A comparison between the two follows.
The local constraint: three sides of the parallelogram are seen
>>> (query, reference) = dtw_test_data.sin_cos()
>>> ita = dtw(query, reference, keep_internals=True, step_pattern=typeIIIc)
>>> dtwPlotDensity(ita)				     # doctest: +SKIP
Symmetric step with global parallelogram-shaped constraint. Note how
long (>2 steps) horizontal stretches are allowed within the window.
>>> ita = dtw(query, reference, keep_internals=True, window_type=itakuraWindow)
>>> dtwPlotDensity(ita)				     # doctest: +SKIP
"""
    # ENDIMPORT

    try:
        cm = d.costMatrix
    except:
        raise ValueError("dtwPlotDensity requires dtw internals (set keep.internals=True on dtw() call)")

    if normalize:
        norm = d.stepPattern.hint
        row, col = numpy.indices(cm.shape)
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

    def __init__(self, shape_dtw_results: ShapeDTW):
        self.shape_dtw_results = shape_dtw_results

    def _get_figure_nrow(self):
        dim_num = self.shape_dtw_results.ts_x.shape[1]
        res = 1 if dim_num < 2 else math.ceil(dim_num/2)
        return res

    def _get_figure_ncol(self):
        dim_num = self.shape_dtw_results.ts_x.shape[1]
        res = 1 if dim_num < 2 else 2
        return res

    @staticmethod
    def _get_ax_indices(dim_num: int, total_dim_num: int):
        row_ind = () if total_dim_num < 3 else (dim_num // 2,)
        col_ind = (dim_num % 2,)

        res = row_ind + col_ind
        return res

    @staticmethod
    def _clean_unnecessary_ax(axis_to_clean, total_dim_num: int):
        ax_ind = ShapeDTWPlot._get_ax_indices(total_dim_num, total_dim_num)
        axis_to_clean[ax_ind].remove()

    def _get_dtw_res_list(self):
        res = self.shape_dtw_results._dtw_results[0] \
            if isinstance(self.shape_dtw_results._dtw_results, list) \
            else self.shape_dtw_results._dtw_results
        return res

    def _dtw_plot_alignment(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotAlignment(dtw_res, **kwargs)

    def _dtw_plot_twoway(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotTwoWay(
            dtw_res,
            xts=self.shape_dtw_results.ts_x,
            yts=self.shape_dtw_results.ts_y,
            **kwargs
        )

    def _dtw_plot_threeway(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotThreeWay(
            dtw_res,
            xts=self.shape_dtw_results.ts_x,
            yts=self.shape_dtw_results.ts_y,
            **kwargs
        )

    def _dtw_plot_density(self, **kwargs):
        dtw_res = self._get_dtw_res_list()
        return dtwPlotDensity(
            dtw_res,
            **kwargs
        )

    def plot(self, plot_type, **kwargs):
        if plot_type == "alignment":
            return self._dtw_plot_alignment(**kwargs)
        elif plot_type == "twoway":
            return self._dtw_plot_twoway(**kwargs)
        elif plot_type == "threeway":
            return self._dtw_plot_threeway(**kwargs)
        else:
            return self._dtw_plot_density(**kwargs)

class ShapeDTWPlotMultivariateDependent(ShapeDTWPlot):

    def __init__(self, shape_dtw_results: MultivariateShapeDTWDependent):
        super().__init__(shape_dtw_results)

    def _dtw_plot_twoway(self, fig_width=8, fig_height=5, **kwargs):

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

        # if Utils.is_odd(total_dim_num):
        #     self._clean_unnecessary_ax(outer_fig, total_dim_num)

        plt.show()

class ShapeDTWPlotMultivariateIndependent(ShapeDTWPlot):

    def __init__(self, shape_dtw_results: MultivariateShapeDTWIndependent):
        super().__init__(shape_dtw_results)

    def _dtw_plot_alignment(self, fig_width=6, fig_height=5, **kwargs):

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

        # if Utils.is_odd(total_dim_num):
        #     self._clean_unnecessary_ax(outer_fig, total_dim_num)

        plt.show()

    def _dtw_plot_density(self, fig_width=5, fig_height=6, **kwargs):

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