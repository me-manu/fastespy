from __future__ import absolute_import, division, print_function
from fastespy.functions import TimeLine
from fastespy.analysis import derivative_filtered
import matplotlib
import numpy as np
matplotlib.use("agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import ListedColormap


def new_cmap(cmap_name, step=25, cmin=0., cmax=0.7):
    """Create a new color map that goes smoothly to white"""

    cm = plt.cm.get_cmap(cmap_name, 256)
    newcolors = cm(np.linspace(cmin, cmax, 256 - step))
    vals = np.zeros((step, 4))

    for i in range(newcolors.shape[1]):
        vals[:, i] = np.linspace(1., newcolors[0, i], step)

    return ListedColormap(np.vstack([vals, newcolors]))

def plot_time_line(t, v, dv=0., function=None, func_kwargs=None, ax=None, **plot_kwargs):
    """
    Plot the time line data of a trigger window and optionally the fit

    :param t: array-like
        time values in micro s
    :param v:
        voltage values in mV
    :param dv:
        voltage uncertainties in mV
    :param fname: str
        filename for plot
    :param function: str
        function identifier for `fastespy.fitting.TimeLine` class
    :param func_kwargs: dict, optional
        result dict from fitting
    :param plot_kwargs: dict, optional
        additional kwargs for plotting
    :return: matplotlib ax object
    """

    data_col = plot_kwargs.pop('data_col', plt.cm.tab10(0.))
    data_ls = plot_kwargs.pop('data_ls', '-')
    data_label = plot_kwargs.pop('data_label', 'data')
    func_col = plot_kwargs.pop('data_col', plt.cm.tab10(0.1))
    func_ls = plot_kwargs.pop('func_ls', '--')
    func_label = plot_kwargs.pop('func_label', 'fit')

    if ax is None:
        ax = plt.gca()

    if np.any(dv) > 0.:
        ax.fill_between(t, v - dv,
                    y2=v + dv,
                    alpha=0.2, color=data_col)

    ax.plot(t, v, color=data_col, ls=data_ls, label=data_label, **plot_kwargs)
    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"Voltage (mV)")

    if function is not None and func_kwargs is not None:
        func = TimeLine(numcomp=func_kwargs['numcomp'], function=function)

        ax.plot(t, func(t, **func_kwargs['fitarg']),
                ls=func_ls, color=func_col, label=func_label, **plot_kwargs)

        chi2dof = func_kwargs['chi2'] / func_kwargs['dof']
        string = ''
        for k in func_kwargs['value'].keys():
            if 'tr' in k or 'td' in k:
                ttype = 'rise' if 'tr' in k else 'decay'
                it = int(k.split('_')[-1])
                string += "$t_{{\\mathrm{{{1:s}}}, {0:n}}} = ({2:.2f} \\pm {3:.2f})\\mu$s\n".format(
                    it+1,
                    ttype,
                    func_kwargs['value'][k],
                    func_kwargs['error'][k])
        string += "$\\chi^2 / \\mathrm{{d.o.f.}} = {0:.2f}$".format(chi2dof)

        leg = ax.legend(title=string, fontsize='x-small')
    else:
        leg = ax.legend(fontsize='x-small')

    plt.setp(leg.get_title(), fontsize='x-small')
    return ax


def plot_time_line_derivative(t, v, fSample, ax=None,
                              fmax = 1.e6, norder = 3,
                              yunit='mV',
                              plot_unfiltered=False,
                              **plot_kwargs):
    """
    Plot the time line data of a trigger window and optionally the fit

    :param t: array-like
        time values in micro s
    :param v:
        voltage values in mV
    :param fSample: float
        sampling frequency
    :param fmax:
        maximum frequency above which filter is applied, see scipy.signal.butter function
    :param norder:
        filter order
    :param plot_unfiltered: bool, optional
        if true, plot unfiltered derivative
    :param plot_kwargs: dict, optional
        additional kwargs for plotting
    :return: matplotlib ax object
    """
    label = plot_kwargs.pop('label', 'Derivative')

    # dv and dv_filter are in units of v * fSample,
    # which is in Hz = 1 / s
    # therefore divide by 1e6 to get it in mu s
    dv, dv_filter = derivative_filtered(v,
                                        fSample=fSample,
                                        fmax=fmax,
                                        norder=norder)
    if ax is None:
        ax = plt.gca()

    if plot_unfiltered:
        ax.plot(t, dv / 1e6, label=label, **plot_kwargs)

    if norder > 0:
        ax.plot(t, dv_filter / 1e6, label=label +' filtered', **plot_kwargs)

    ax.set_xlabel(r"Time ($\mu$s)")
    ax.set_ylabel(r"$dU/dt$ ({0:s}$\,\mu\mathrm{{s}}^{{-1}}$)".format(yunit))
    ax.legend(fontsize='x-small')

    return ax


def plot_time_line_and_derivative(
                                  t, v,
                                  dv=0.,
                                  function=None,
                                  func_kwargs=None,
                                  fig=None,
                                  fSample=None,
                                  fmax = 1.e6,
                                  norder = 3,
                                  kwargs_timeline={},
                                  kwargs_derivative={}
                                  ):
    gs = gridspec.GridSpec(3, 1)
    kwargs_derivative.setdefault('lw', 1.)
    if fig is None:
        fig = plt.figure(figsize=(6, 6), tight_layout=True)

    ax_t_vs_v = fig.add_subplot(gs[:2, 0])
    ax_dvdt = fig.add_subplot(gs[2, 0])

    plot_time_line(t, v, dv=dv, function=function, func_kwargs=func_kwargs, ax=ax_t_vs_v, **kwargs_timeline)

    plot_time_line_derivative(t, v, fSample, ax=ax_dvdt,
                              fmax=fmax, norder=norder,
                              **kwargs_derivative)

    ax_t_vs_v.set_xlabel('')
    ax_t_vs_v.tick_params(labelbottom=False)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)

    v = ax_t_vs_v.get_xlim()
    ax_dvdt.set_xlim(v)

    ax_t_vs_v.xaxis.set_minor_locator(MultipleLocator(5))
    ax_dvdt.xaxis.set_minor_locator(MultipleLocator(5))
    ax_t_vs_v.yaxis.set_minor_locator(MultipleLocator(5))
    ax_dvdt.yaxis.set_minor_locator(MultipleLocator(5))

    ax_t_vs_v.grid(which='both', lw=0.5, ls='-', color='0.8')
    ax_dvdt.grid(which='both', lw=0.5, ls='-', color='0.8')

    return fig, ax_t_vs_v, ax_dvdt


def plot_2d_hist(x,y,
                 fig=None, ax_2d=None, ax_x=None, ax_y=None,
                 bins_x=100, bins_y=100,
                 mesh_kwargs={},
                 add_cbar=False,
                 add_contours=True,
                 axes_lims=[0.01,0.99],
                 quantiles=[0.05,0.95],
                 hist_2d_kwargs={},
                 contour_kwargs={},
                 hist_x_kwargs={},
                 hist_y_kwargs={}):
    """
    Create a 2d histogram with projected histograms

    Returns
    -------
    fig, ax for 2d hist, ax for x hist, ax for y hist, bins for x, bins for y

    Notes
    -----
    Adapted from https://matplotlib.org/examples/pylab_examples/scatter_hist.html
    """
    mesh_kwargs.setdefault('cmap', plt.cm.Blues)

    if fig is None:
        fig = plt.figure(1, figsize=(8, 8))

    if ax_2d is None or ax_x is None or ax_y is None:
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        ax_2d = plt.axes(rect_scatter)
        ax_x = plt.axes(rect_histx)
        ax_y = plt.axes(rect_histy)

    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)

    # the 2d histogram
    n, b1, b2 = np.histogram2d(x, y, bins=[bins_x, bins_y], **hist_2d_kwargs)
    b1cen = 0.5 * (b1[1:] + b1[:-1])
    b2cen = 0.5 * (b2[1:] + b2[:-1])
    b11, b22 = np.meshgrid(b1, b2, indexing='ij')

    c = ax_2d.pcolormesh(b11, b22, n, **mesh_kwargs)

    if add_contours:
        levels = contour_kwargs.pop('levels',
                                    np.linspace(n.min(), n.max(), 7)[1:-1])
        contours = ax_2d.contour(b1cen, b2cen, n.T, levels, **contour_kwargs)
        ax_2d.clabel(contours, **contour_kwargs)

    if add_cbar:
        plt.colorbar(c)

    nx, _, _ = ax_x.hist(x, bins=bins_x, **hist_x_kwargs)
    ny, _, _ = ax_y.hist(y, bins=bins_y, orientation='horizontal', **hist_y_kwargs)

    # add quantiles to hist plots
    # first compute cdf of histograms
    cdf_x = np.cumsum(nx)
    cdf_x = (cdf_x - cdf_x[0]) / (cdf_x[-1] - cdf_x[0])

    cdf_y = np.cumsum(ny)
    cdf_y = (cdf_y - cdf_y[0]) / (cdf_y[-1] - cdf_y[0])

    # compute quantiles
    if quantiles is not None:
        q_x = np.interp(quantiles, xp=cdf_x, fp=b1cen)
        q_y = np.interp(quantiles, xp=cdf_y, fp=b2cen)
        np.set_printoptions(precision=2)
        print("x={0} quantile values are {1}".format(quantiles, q_x))
        print("y={0} quantile values are {1}".format(quantiles, q_y))
        for i in range(2):
            ax_x.axvline(q_x[i], ls=':', color='k')
            ax_y.axhline(q_y[i], ls=':', color='k')


    # compute axes lims
    if axes_lims is not None:
        xlims = np.interp(axes_lims, xp=cdf_x, fp=b1cen)
        ylims = np.interp(axes_lims, xp=cdf_y, fp=b2cen)

        ax_x.set_xlim(xlims)
        ax_y.set_ylim(ylims)
        ax_2d.set_xlim(xlims)
        ax_2d.set_ylim(ylims)

    else:
        ax_x.set_xlim(ax_2d.get_xlim())
        ax_y.set_ylim(ax_2d.get_ylim())

    return fig, ax_2d, ax_x, ax_y, bins_x, bins_y

def plot_scatter_w_hist(x,y,
                        fig=None, ax_2d=None, ax_x=None, ax_y=None,
                        bins_x=100, bins_y=100,
                        scatter_kwargs={},
                        add_cbar=False,
                        hist_x_kwargs={},
                        hist_y_kwargs={}):
    """
    Create a scatter plot with projected histograms

    Returns
    -------
    fig, ax for 2d hist, ax for x hist, ax for y hist, bins for x, bins for y

    Notes
    -----
    Adapted from https://matplotlib.org/examples/pylab_examples/scatter_hist.html
    """
    if fig is None:
        fig = plt.figure(1, figsize=(8, 8))

    if ax_2d is None or ax_x is None or ax_y is None:
        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom_h, width, 0.2]
        rect_histy = [left_h, bottom, 0.2, height]

        ax_2d = plt.axes(rect_scatter)
        ax_x = plt.axes(rect_histx)
        ax_y = plt.axes(rect_histy)

    ax_x.tick_params(labelbottom=False)
    ax_y.tick_params(labelleft=False)

    ax_2d.scatter(x, y, **scatter_kwargs)
    if add_cbar:
        plt.colorbar(c)

    ax_x.hist(x, bins=bins_x, **hist_x_kwargs)
    ax_y.hist(y, bins=bins_y, orientation='horizontal', **hist_y_kwargs)

    ax_x.set_xlim(ax_2d.get_xlim())
    ax_y.set_ylim(ax_2d.get_ylim())

    return fig, ax_2d, ax_x, ax_y, bins_x, bins_y
