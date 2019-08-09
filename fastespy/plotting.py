from __future__ import absolute_import, division, print_function
from fastespy.fitting import TimeLine
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


def plot_time_line(t, v, dv = 0., function=None, func_kwargs=None, **plot_kwargs):
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
    ax = plot_kwargs.pop('ax', plt.gca())

    ax.fill_between(t, v - dv,
                    y2=v + dv,
                    alpha=0.2, color=data_col,
                    **plot_kwargs)

    ax.plot(t, v, color=plt.cm.tab10(0.), ls=data_ls, label=data_label)
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

        leg = plt.legend(title=string, fontsize='x-small')
    else:
        leg = plt.legend(fontsize='x-small')

    plt.setp(leg.get_title(), fontsize='x-small')
    return ax
