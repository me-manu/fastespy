from __future__ import absolute_import, division, print_function
import argparse
import os
import numpy as np
import logging
import itertools
from fastespy.plotting import plot_2d_hist, new_cmap, plot_scatter_w_hist
from fastespy.analysis import init_logging
from fastespy import functions
import glob
import matplotlib.pyplot as plt

if __name__ == '__main__':
    usage = "usage: %(prog)s -i indir -o outdir "
    description = "Plot some results on pulse fit results"

    parser = argparse.ArgumentParser(usage=usage, description=description)
    parser.add_argument('-i', '--indir', required=True, help='Path with cobmined npy files for all triggers')
    parser.add_argument('-o', '--outdir', required=True, help='Path to output file')
    args = parser.parse_args()
    init_logging("INFO", color=True)

    # First, we get the data path where
    # the collected results from the pulse shape fit that were read
    # in with `read_result_pulse_shape_triggered.py` are stored.

    file_list = glob.glob(os.path.join(args.indir, "*.npy"))
    if not len(file_list):
        raise ValueError("No files found")

    for i, result_file in enumerate(file_list):
        logging.info("Reading {0:s}".format(result_file))
        r = np.load(result_file, allow_pickle=True).flat[0]
        if not i:
            result = r
        else:
            for k, v in result.items():
                result[k] = np.append(v, r[k])

    # print some stats
    logging.info("In total, the files contain {0:n} events".format(result[k].size))
    logging.info("Of which {0:n} are events taken with light".format(np.sum(result['type'] == 0)))
    logging.info("Of which {0:n} are events taken w/o light and no fiber coupled to TES (intrinsic)".format(
        np.sum(result['type'] == 1)))
    logging.info("Of which {0:n} are events taken w/o light and a fiber coupled to TES (extrinsic)".format(
        np.sum(result['type'] == 2)))

    # plot the average pulse
    mask = result['type'] == 0  # light data only
    mask &= result['fit_ok']  # fit converged
    mask &= result['chi2_dof'] <= 1.5

    logging.info("{0:n} / {1:n} fit with light included".format(mask.sum(), (result['type'] == 0.).sum()))

    pars = dict(tr=result['tr'][mask].mean(),
                td=result['td'][mask].mean(),
                A=result['amplitude'][mask].mean(),
                c=0.,
                t0=10.
                )
    x = np.linspace(0., 50., 500)
    if r['function'] == 'tesresponse':
        f = functions.tesresponse
    elif r['function'] == 'tesresponse_simple':
        f = functions.tesresponse_simple
    elif r['function'] == 'expflare':
        f = functions.expflare

    plt.plot(x, f(x, **pars))

    for k, v in pars.items():
        if k == 'c' or k == 't0': continue
        logging.info("Mean value for {0:5s}: {1:.3f} +/- {2:.3f}".format(
            k, v, np.sqrt(result[k if not k == 'A' else 'amplitude'][mask].var())))

    plt.xlabel("Time ($\mu$s)")
    plt.ylabel("Voltage (mV)")
    plt.savefig(os.path.join(args.outdir, "avg_pulse.png"), dpi=150)
    plt.close("all")

    # plot 2d histograms
    cmap_name = 'nipy_spectral_r'
    cmap = new_cmap(cmap_name)

    label = dict(
        tr=r'Rise time $(\mu\mathrm{s})$',
        td=r'Decay time $(\mu\mathrm{s})$',
        amplitude=r'Amplitude (mV)',
        integral=r'Pulse Integral (mV $\mu$s)',
        chi2_dof=r'$\chi^2/$d.o.f.'
    )

    # Plot 2d histograms for photon pulses only in order to see spread of values
    for i, (x, y) in enumerate(itertools.combinations(list(result.keys()), 2)):
        if x == 'chi2' or y == 'chi2' or x == 'dof' or y == 'dof' \
                or x == 'type' or y == 'type' or x == 'fit_ok' or y == 'fit_ok' \
                or x == 'function' or y == 'function':
            continue
        logging.info(f'Plotting 2d histogram for {x} vs {y}')
        fig = plt.figure(i + 1, figsize=(7, 6))
        for t in range(1):
            m = result['type'] == t
            m &= (result['fit_ok'] & (result['chi2_dof'] < 1.5))
            fig, ax_2d, ax_x, ax_y, bins_x, bins_y = plot_2d_hist(result[x][m], result[y][m],
                                                                  fig=fig,
                                                                  bins_x=50,
                                                                  bins_y=50,
                                                                  add_contours=True,
                                                                  add_cbar=False,
                                                                  quantiles=[0.05, 0.95],
                                                                  axes_lims=[0.01, 0.99],
                                                                  hist_2d_kwargs={"density": True},
                                                                  contour_kwargs={"colors": "k"},
                                                                  mesh_kwargs={"cmap": cmap},
                                                                  hist_x_kwargs={"density": True, "color": cmap(1.)},
                                                                  hist_y_kwargs={"density": True, "color": cmap(1.)}
                                                                  )
        ax_2d.set_xlabel(label[x])
        ax_2d.set_ylabel(label[y])
        ax_2d.grid(which='both', ls='-', lw='0.5', color='0.7')
        ax_x.grid(which='both', ls='-', lw='0.5', color='0.7')
        ax_y.grid(which='both', ls='-', lw='0.5', color='0.7')
        fig.subplots_adjust(left=0.2)
        fig.savefig(os.path.join(args.outdir, f"scatter_2d_light_hist_{x}_{y}.png"), dpi=150)
        plt.close("all")

    # From the above histograms, define axis limits / bins
    # for scatter plots.These roughly correspond to to two times the 95 % quantiles.
    n_bins = 50
    bins = dict(
        tr=np.linspace(0., 5., n_bins),
        td=np.linspace(0., 10., n_bins),
        integral=np.linspace(-300., -50, n_bins),
        amplitude=np.linspace(10., 40., n_bins),
        chi2_dof=np.linspace(0.5, 3., n_bins)
    )

    # Now we plot scatter plots: light and intrinsic, light vs all.
    label_points = ['light', 'intrinsic', 'extrinsic']

    for i, (x, y) in enumerate(itertools.combinations(list(result.keys()), 2)):
        if x == 'chi2' or y == 'chi2' or x == 'dof' or y == 'dof' \
                or x == 'type' or y == 'type' or x == 'fit_ok' or y == 'fit_ok' \
                or x == 'function' or y == 'function':
            continue
        logging.info(f'Plotting 2d scatter plots for {x} vs {y}')
        fig = plt.figure(i + 1, figsize=(7, 6))
        for t in range(2):
            m = result['type'] == t
            fig, ax_2d, ax_x, ax_y, bins_x, bins_y = plot_scatter_w_hist(result[x][m], result[y][m],
                                                                         bins_x=bins[x],
                                                                         bins_y=bins[y],
                                                                         fig=fig,
                                                                         # bins_x=100,
                                                                         # bins_y=100,
                                                                         scatter_kwargs={"marker": 'o',
                                                                                         "alpha": 0.5,
                                                                                         "label": label_points[t]
                                                                                         },
                                                                         hist_x_kwargs={"density": True, "alpha": 0.5},
                                                                         hist_y_kwargs={"density": True, "alpha": 0.5}
                                                                         )
        ax_2d.set_xlabel(label[x])
        ax_2d.set_ylabel(label[y])
        ax_2d.legend()
        ax_2d.grid(which='both', ls='-', lw='0.5', color='0.7')
        ax_x.grid(which='both', ls='-', lw='0.5', color='0.7')
        ax_y.grid(which='both', ls='-', lw='0.5', color='0.7')

        ax_2d.set_xlim(bins[x].min(), bins[x].max())
        ax_2d.set_ylim(bins[y].min(), bins[y].max())
        ax_x.set_xlim(ax_2d.get_xlim())
        ax_y.set_ylim(ax_2d.get_ylim())

        fig.subplots_adjust(left=0.2)
        fig.savefig(os.path.join(args.outdir, f'scatter_2d_hist_{x}_{y}_light+intrinsic.png'), dpi=150)
        plt.close("all")

    for i, (x, y) in enumerate(itertools.combinations(list(result.keys()), 2)):
        if x == 'chi2' or y == 'chi2' or x == 'dof' or y == 'dof' \
                or x == 'type' or y == 'type' or x == 'fit_ok' or y == 'fit_ok' \
                or x == 'function' or y == 'function':
            continue
        logging.info(f'Plotting 2d scatter plots for {x} vs {y}')
        fig = plt.figure(i + 1, figsize=(7, 6))
        for t in range(3):
            m = result['type'] == t
            fig, ax_2d, ax_x, ax_y, bins_x, bins_y = plot_scatter_w_hist(result[x][m], result[y][m],
                                                                         bins_x=bins[x],
                                                                         bins_y=bins[y],
                                                                         fig=fig,
                                                                         # bins_x=100,
                                                                         # bins_y=100,
                                                                         scatter_kwargs={"marker": 'o',
                                                                                         "alpha": 0.5,
                                                                                         "label": label_points[t]
                                                                                         },
                                                                         hist_x_kwargs={"density": True,
                                                                                        "alpha": 0.5},
                                                                         hist_y_kwargs={"density": True,
                                                                                        "alpha": 0.5}
                                                                         )
        ax_2d.set_xlabel(label[x])
        ax_2d.set_ylabel(label[y])
        ax_2d.legend()
        ax_2d.grid(which='both', ls='-', lw='0.5', color='0.7')
        ax_x.grid(which='both', ls='-', lw='0.5', color='0.7')
        ax_y.grid(which='both', ls='-', lw='0.5', color='0.7')

        ax_2d.set_xlim(bins[x].min(), bins[x].max())
        ax_2d.set_ylim(bins[y].min(), bins[y].max())
        ax_x.set_xlim(ax_2d.get_xlim())
        ax_y.set_ylim(ax_2d.get_ylim())

        fig.subplots_adjust(left=0.2)
        fig.savefig(os.path.join(args.outdir, f'scatter_2d_hist_{x}_{y}_light+all.png'), dpi=150)
        plt.close("all")

