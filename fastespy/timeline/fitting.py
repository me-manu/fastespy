from __future__ import absolute_import, division, print_function
import numpy as np
import iminuit as minuit
import time
import functools
import logging
from .processing import build_trigger_windows
from scipy import optimize as op
from collections.abc import OrderedDict
from copy import deepcopy
from scipy.special import gammainc
from .models import TimeLine

pvalue = lambda dof, chisq: 1. - gammainc(.5 * dof, .5 * chisq)

def setDefault(func = None, passed_kwargs = {}):
    """
    Read in default keywords of the simulation and pass to function
    """
    if func is None:
        return functools.partial(setDefault, passed_kwargs = passed_kwargs)
    @functools.wraps(func)
    def init(*args, **kwargs):
        for k in passed_kwargs.keys():
            kwargs.setdefault(k,passed_kwargs[k])
        return func(*args, **kwargs)
    return init

minuit_def = {
    'verbosity': 0,
    'int_steps': 1e-4,
    'strategy': 2,
    'tol': 1e-5,
    'up': 1., # it's a chi 2 fit
    'max_tol_increase': 3000.,
    'tol_increase': 1000.,
    'ncall': 10000,
    'steps': 40,
    'scan_bound': (0.,10.),
    'pedantic': True,
    'precision': None,
    'scipy': False,
    'pinit': {'A' : 1.,
              'c': 1.,
              't0': 1.,
              'tr': 1.,
              'td': 1.},
    'fix': {'A' : False,
            'c': False,
            't0': False,
            'tr': False,
            'td': False},
    'islog': {'A' : False,
              'c': False,
              't0': False,
              'tr': False,
              'td': False},
    'limits': {'A' : [0.,100.],
               'c': [0.,5.],
               't0': [0.,100.],
               'tr': [0.,100.],
               'td': [0.,100.]}
}
# --- miniuit defaults ------------------------------------- #

class FitTimeLine(object):
    def __init__(self, t, v, dv, fSample):
        """
        Initialize the fitting class

        :param t: array-like
            time values in micro s
        :param v: array-like
            voltage values in mV
        :param dv: array-like
            uncertainties of voltage values in mV
        :param fSample: float
            Sampling frequency in Hz
        """
        self._t = t
        self._v = v
        self._dv = dv
        self._f = None
        self._fSample = fSample
        return

    @property
    def t(self):
        return self._t

    @property
    def v(self):
        return self._v

    @property
    def dv(self):
        return self._dv

    @property
    def f(self):
        return self._f

    @property
    def fSample(self):
        return self._fSample

    @property
    def m(self):
        return self._m

    @t.setter
    def t(self, t):
        self._t = t
        return

    @v.setter
    def v(self, v):
        self._v = v
        return

    @fSample.setter
    def fSample(self, fSample):
        self._fSample = fSample
        return

    @dv.setter
    def dv(self, dv):
        self._dv = dv
        return

    def calcObjFunc(self,*args):
        return self.__calcObjFunc(*args)

    def __calcObjFunc(self,*args):
        """
        objective function passed to iMinuit
        """
        params = {}
        for i,p in enumerate(self.parnames):
            if self.par_islog[p]:
                params[p] = np.power(10.,args[i])
            else:
                params[p] = args[i]
        return self.returnObjFunc(params)

    def __wrapObjFunc(self,args):
        """
        objective function passed to scipy.optimize
        """
        params = {}
        for i,p in enumerate(self.parnames):
            if not self.fitarg['fix_{0:s}'.format(p)]:
                if self.par_islog[p]:
                    params[p] = np.power(10.,args[i])
                else:
                    params[p] = args[i]
            else:
                if self.par_islog[p]:
                    params[p] = np.power(10.,self.fitarg[p])
                else:
                    params[p] = self.fitarg[p]
        return self.returnObjFunc(params)

    def returnObjFunc(self,params):
        """Calculate the objective function"""
        f = self._f(self._t, **params)
        chi2 = ((self._v - f)**2. / self._dv**2.).sum()
        return chi2

    @setDefault(passed_kwargs = minuit_def)
    def fill_fitarg(self, **kwargs):
        """Helper function to fill the dictionary for minuit fitting"""
        # set the fit arguments
        fitarg = {}
        fitarg.update(kwargs['pinit'])
        for k in kwargs['limits'].keys():
            fitarg['limit_{0:s}'.format(k)] = kwargs['limits'][k]
            fitarg['fix_{0:s}'.format(k)] = kwargs['fix'][k]
            fitarg['error_{0:s}'.format(k)] = kwargs['pinit'][k] * kwargs['int_steps']

        fitarg = OrderedDict(sorted(fitarg.items()))
        # get the names of the parameters
        self.parnames = kwargs['pinit'].keys()
        self.par_islog = kwargs['islog']
        return fitarg

    @setDefault(passed_kwargs = minuit_def)
    def run_migrad(self,fitarg,**kwargs):
        """
        Helper function to initialize migrad and run the fit.
        Initial parameters are optionally estimated with scipy optimize.
        """
        self.fitarg = fitarg

        values, bounds = [],[]
        for k in self.parnames:
            values.append(fitarg[k])
            bounds.append(fitarg['limit_{0:s}'.format(k)])

        logging.debug(self.parnames)
        logging.debug(values)

        logging.debug(self.__wrapObjFunc(values))

        if kwargs['scipy']:
            self.res = op.minimize(self.__wrapObjFunc,
                                   values,
                                   bounds = bounds,
                                   method='TNC',
                                   #method='Powell',
                                   options={'maxiter': kwargs['ncall']} #'xtol': 1e-20, 'eps' : 1e-20, 'disp': True}
                                   #tol=None, callback=None,
                                   #options={'disp': False, 'minfev': 0, 'scale': None,
                                   #'rescale': -1, 'offset': None, 'gtol': -1,
                                   #'eps': 1e-08, 'eta': -1, 'maxiter': kwargs['ncall'],
                                   #'maxCGit': -1, 'mesg_num': None, 'ftol': -1, 'xtol': -1, 'stepmx': 0,
                                   #'accuracy': 0}
                                   )
            logging.debug(self.res)
            for i,k in enumerate(self.parnames):
                fitarg[k] = self.res.x[i]

            logging.debug(fitarg)
        cmd_string = "lambda {0}: self.__calcObjFunc({0})".format(
            (", ".join(self.parnames), ", ".join(self.parnames)))

        string_args = ", ".join(self.parnames)
        global f # needs to be global for eval to find it
        f = lambda *args: self.__calcObjFunc(*args)

        cmd_string = "lambda %s: f(%s)" % (string_args, string_args)
        logging.debug(cmd_string)

        # work around so that the parameters get names for minuit
        self._minimize_f = eval(cmd_string, globals(), locals())

        self._m = minuit.Minuit(self._minimize_f,
                               print_level =kwargs['verbosity'],
                               errordef = kwargs['up'],
                               pedantic = kwargs['pedantic'],
                               **fitarg)

        self._m.tol = kwargs['tol']
        self._m.strategy = kwargs['strategy']

        logging.debug("tol {0:.2e}, strategy: {1:n}".format(
            self._m.tol,self._m.strategy))

        self._m.migrad(ncall = kwargs['ncall']) #, precision = kwargs['precision'])
        return

    def __print_failed_fit(self):
        """print output if migrad failed"""
        if not self._m.migrad_ok():
            fmin = self._m.get_fmin()
            logging.warning(
                '*** migrad minimum not ok! Printing output of get_fmin'
            )
            logging.warning('{0:s}:\t{1}'.format('*** has_accurate_covar',
                                                 fmin.has_accurate_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_covariance',
                                                 fmin.has_covariance))
            logging.warning('{0:s}:\t{1}'.format('*** has_made_posdef_covar',
                                                 fmin.has_made_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_posdef_covar',
                                                 fmin.has_posdef_covar))
            logging.warning('{0:s}:\t{1}'.format('*** has_reached_call_limit',
                                                 fmin.has_reached_call_limit))
            logging.warning('{0:s}:\t{1}'.format('*** has_valid_parameters',
                                                 fmin.has_valid_parameters))
            logging.warning('{0:s}:\t{1}'.format('*** hesse_failed',
                                                 fmin.hesse_failed))
            logging.warning('{0:s}:\t{1}'.format('*** is_above_max_edm',
                                                 fmin.is_above_max_edm))
            logging.warning('{0:s}:\t{1}'.format('*** is_valid',
                                                 fmin.is_valid))
        return


    def __repeat_migrad(self, **kwargs):
        """Repeat fit if fit was above edm"""
        fmin = self._m.get_fmin()
        if not self._m.migrad_ok() and fmin['is_above_max_edm']:
            logging.warning(
                'Migrad did not converge, is above max edm. Increasing tol.'
            )
            tol = self._m.tol
            self._m.tol *= self._m.edm /(self._m.tol * self._m.errordef ) * kwargs['tol_increase']

            logging.info('New tolerance : {0}'.format(self._m.tol))
            if self._m.tol >= kwargs['max_tol_increase']:
                logging.warning(
                    'New tolerance to large for required precision'
                )
            else:
                self._m.migrad(
                    ncall = kwargs['ncall'])#,
                #precision = kwargs['precision']
                #)
                logging.info(
                    'Migrad status after second try: {0}'.format(
                        self._m.migrad_ok()
                    )
                )
                self._m.tol = tol
        return

    @setDefault(passed_kwargs = minuit_def)
    def fit(self,tmin = None, tmax = None, function = 'tesresponse',
            minos = 1., parscan = 'none', fmax = 1e6, norder = 3,
            dvdt_thr = -25., maxcomp = 3, v_thr=0.,
            **kwargs):
        """
        Fit the time series

        {options}

        :param tmin: float
            Minimum time in time series. If None, don't use a limit. Default: None
        :param tmax:
            Maximum time in time series. If None, don't use a limit. Default: None
        :param function: str
            either "tesresponse" or "expflare". Determines the function that is fit to the data.
            Default: "tesresponse"
        :param minos: float
            Confidence level in sigma for which minos errors are calculated. Default: 1.
        :param parscan: str
            either 'none' or name of parameter. If name of parameter, the likelihood is profiled
            for this parameter and the profile likelihood is returned. Default: 'none'
        :param fmax: float
            Maximum frequency for filter (used to find trigger time)
        :param norder: int
            Order of filter (used to find trigger time)
        :param dvdt_thr: float
            threshold to find trigger from derivative in mV / micro sec. Default: -25.
        :param v_thr: float
            threshold to find trigger from time series in V. Default: 0.
        :param maxcomp: int
            Maximum pulse components allowed in one trigger window.
        :param kwargs:
            Additional key word arguments passed to minuit.
        :return:
            Dictionary with results
        """

        if np.isscalar(tmin) and np.isscalar(tmax):
            m = (self.t >= tmin) & (self.t < tmax)
        elif np.isscalar(tmax):
            m = (self.t < tmax)
        elif np.isscalar(tmin):
            m = (self.t >= tmin)
        if np.isscalar(tmin) or np.isscalar(tmax) == float:
            self._t = self._t[m]
            self._v = self._v[m]
            self._dv = self._dv[m]

        t0s, _, _ = build_trigger_windows(self._t / 1e6, self._v / 1e3,
                                          self._fSample,
                                          thr=dvdt_thr,
                                          thr_v=v_thr,
                                          fmax=fmax,
                                          norder=norder)

        t0s = np.array(t0s) * 1e6  # convert back to micro s
        ntrigger = t0s.size
        logging.info("Within window, found {0:n} point(s) satisfying trigger criteria".format(ntrigger))

        t1 = time.time()

        aic, chi2, nPars, vals, errs, fitargs, dof = [], [], [], [], [], [], []

        # loop through trigger times
        #for i, t0 in enumerate(t0s):
        i = 0
        while i < ntrigger and i < maxcomp:
            f = TimeLine(numcomp=i+1, function=function)

            kwargs['pinit']['t0_{0:03n}'.format(i)] = t0s[i]
            kwargs['limits']['t0_{0:03n}'.format(i)] = [self._t.min(), self._t.max()]

            if i > 0:
                if self._m.migrad_ok():
                    kwargs['pinit'].update({k : self._m.values[k] for k in kwargs['pinit'].keys() \
                                            if k in self._m.values.keys()})
                for k in ['td', 'tr', 'A']:
                    kwargs['pinit']['{0:s}_{1:03n}'.format(k, i)] = kwargs['pinit']['{0:s}_000'.format(k)]
                    kwargs['limits']['{0:s}_{1:03n}'.format(k, i)] = kwargs['limits']['{0:s}_000'.format(k)]

            kwargs['fix'] = {k: False for k in kwargs['pinit'].keys()}
            kwargs['islog'] = {k: False for k in kwargs['pinit'].keys()}

            self._f = lambda t, **params : f(t, **params)

            fitarg = self.fill_fitarg(**kwargs)
            logging.debug(fitarg)

            self.run_migrad(fitarg, **kwargs)

            npar = np.sum([np.invert(self._m.fitarg[k]) for k in self._m.fitarg.keys() if 'fix' in k])

            try:
                self._m.hesse()
                logging.debug("Hesse matrix calculation finished")
            except RuntimeError as e:
                logging.warning(
                    "*** Hesse matrix calculation failed: {0}".format(e)
                )

            logging.debug(self._m.fval)
            self.__repeat_migrad(**kwargs)
            logging.debug(self._m.fval)

            fmin = self._m.get_fmin()


            if not fmin.hesse_failed:
                try:
                    self.corr = self._m.np_matrix(correlation=True)
                except:
                    self.corr = -1

            logging.debug(self._m.values)

            if self._m.migrad_ok():

                if parscan in self.parnames:
                    parscan, llh, bf, ok = self.llhscan(parscan,
                                                        bounds = kwargs['scan_bound'],
                                                        steps = kwargs['steps'],
                                                        log = False
                                                        )
                    self._m.fitarg['fix_{0:s}'.format(parscan)] = False

                    if np.min(llh) < self._m.fval:
                        idx = np.argmin(llh)
                        if ok[idx]:
                            logging.warning("New minimum found in objective function scan!")
                            fitarg = deepcopy(self._m.fitarg)
                            for k in self.parnames:
                                fitarg[k] = bf[idx][k]
                                fitarg['fix_{0:s}'.format(parscan)] = True
                                kwargs['scipy'] = False
                                self.run_migrad(fitarg, **kwargs)

                if minos:
                    for k in self._m.values.keys():
                        if kwargs['fix'][k]:
                            continue
                        self._m.minos(k,minos)
                    logging.debug("Minos finished")

            else:
                self.__print_failed_fit()

                # terminate if we have tested all components
                if i == t0s.size - 1 or i == maxcomp - 1:
                    return dict(chi2 = self._m.fval,
                                value = dict(self._m.values),
                                error = dict(self._m.errors),
                                aic = 2. * npar + self._m.fval,
                                dof = self._t.size - npar,
                                npar = npar,
                                fitarg = self._m.fitarg,
                                numcomp = i + 1,
                                fit_ok = False
                                )

            vals.append(dict(self._m.values))
            errs.append(dict(self._m.errors))
            fitargs.append(self._m.fitarg)
            nPars.append(npar)
            aic.append(2. * npar + self._m.fval)
            dof.append(self._t.size - nPars[-1])
            chi2.append(self._m.fval)

            i+=1

            # bad fit, add additional trigger time
            #if chi2[-1] / dof[-1] > 2.5 and i >= ntrigger:
            if pvalue(dof[-1], chi2[-1]) < 0.01 and i >= ntrigger:
                logging.info("bad fit and all trigger times added, adding additional component")
                idresmax = np.argmax(np.abs((self._v - self._f(self._t, **self._m.fitarg))/self._dv))
                t0s = np.append(t0s, self._t[idresmax])

        # select best fit
        ibest = np.argmin(aic)
        logging.info('fit took: {0}s'.format(time.time() - t1))
        logging.info("Best AIC = {0:.2f} for {1:n} components".format(aic[ibest], ibest + 1))

        for k in vals[ibest].keys():
            if kwargs['fix'][k]:
                err = np.nan
            else:
                err = fitargs[ibest]['error_{0:s}'.format(k)]
            val = fitargs[ibest]['{0:s}'.format(k)]
            logging.info('best fit {0:s}: {1:.5e} +/- {2:.5e}'.format(k,val,err))

        result = dict(chi2 = chi2[ibest],
                      value = vals[ibest],
                      error = errs[ibest],
                      aic = aic[ibest],
                      dof = dof[ibest],
                      npar = nPars[ibest],
                      fitarg = fitargs[ibest],
                      numcomp = ibest + 1,
                      fit_ok = True
                      )
        if minos:
            result['minos'] = self._m.merrors
        if parscan in self.parnames:
            result['parscan'] = (r, llh)

        return result

    def llhscan(self, parname, bounds, steps, log = False):
        """
        Perform a manual scan of the objective function for one parameter
        (inspired by mnprofile)

        Parameters
        ----------
        parname: str
            parameter that is scanned

        bounds: list or tuple
            scan bounds for parameter

        steps: int
            number of scanning steps

        {options}

        log: bool
            if true, use logarithmic scale

        Returns
        -------
        tuple of 4 lists containing the scan values, likelihood values,
        best fit values at each scanning step, migrad_ok status
        """
        llh, pars, ok = [],[],[]
        if log:
            values = np.logscape(np.log10(bounds[0]),np.log10(bounds[1]), steps)
        else:
            values = np.linspace(bounds[0], bounds[1], steps)

        for i,v in enumerate(values):
            fitarg = deepcopy(self._m.fitarg)
            fitarg[parname] = v
            fitarg['fix_{0:s}'.format(parname)] = True

            string_args = ", ".join(self.parnames)
            global f # needs to be global for eval to find it
            f = lambda *args: self.__calcObjFunc(*args)

            cmd_string = "lambda %s: f(%s)" % (string_args, string_args)

            minimize_f = eval(cmd_string, globals(), locals())

            m = minuit.Minuit(minimize_f,
                              print_level=0, forced_parameters=self._m.parameters,
                              pedantic=False, **fitarg)
            m.migrad()
            llh.append(m.fval)
            pars.append(m.values)
            ok.append(m.migrad_ok())

        return values, np.array(llh), pars, ok