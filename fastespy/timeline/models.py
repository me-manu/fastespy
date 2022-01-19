from __future__ import absolute_import, division, print_function
import numpy as np
from scipy.integrate import simps


def expflare(t, **kwargs):
    """
    Exponential flare function adopted for TES time line

    f(t) = c - 2. * A / (exp((t0 - t) / tr) + exp((t - t0) / td)) +

    :param t: array-like
        array with times
    :param kwargs:
        Function parameters: t0, tr, td, A, c, all floats

    :return: array
        array with function values
    """
    if np.isscalar(t):
        t = np.array([t])
    elif type(t) == list:
        t = np.array(t)

    rise_exp = (kwargs['t0'] - t) / kwargs['tr']
    decay_exp = (t - kwargs['t0']) / kwargs['td']

    exp_rise = np.exp(rise_exp)
    exp_decay = np.exp(decay_exp)

    result = kwargs['c'] - 2. * kwargs['A'] / (exp_rise + exp_decay)
    return result


def calc_xi(td, tr):
    """
    Calculate the "xi" parameter of tes function

    if xi < 0 then tr is rise time and td is decay time
    otherwise tr and td have switched places sind tes function is
    symmetrix in tr and td
    """
    return np.power(td / tr, td / (tr - td)) - np.power(td / tr, tr / (tr - td))


def tesresponse(t,**kwargs):
    """
    TES response function in small signal limit,
    See Christoph Weinsheimer's PhD thesis, Chapter 10

    I(t) = c - A / xi  * (exp(-(t - t0) / tr) - exp(-(t - t0) / td)) for t > t0
        and c otherwise
    xi = (td / tr)^(td / (tr - td)) - (td / tr)^(tr / (tr - td))

    :param t: array-like
        array with times
    :param kwargs:
        Function parameters: t0, tr, td, A, c

    :return: array
        array with function values
    """
    if np.isscalar(t):
        t = np.array([t])
    elif type(t) == list:
        t = np.array(t)

    xi = calc_xi(td=kwargs['td'], tr=kwargs['tr'])

    m = t > kwargs['t0']

    rise_exp = -(t - kwargs['t0']) / kwargs['tr']
    decay_exp = -(t - kwargs['t0']) / kwargs['td']

    exp_rise = np.exp(rise_exp)
    exp_decay = np.exp(decay_exp)

    result = np.zeros_like(t)
    result[m] = kwargs['c'] - (exp_rise - exp_decay)[m] * kwargs['A'] / xi
    result[~m] = np.full((~m).sum(), kwargs['c'])
    return result


def tesresponse_simple(t, tol=1e-6, **kwargs):
    if np.isscalar(t):
        t = np.array([t])
    elif type(t) == list:
        t = np.array(t)

    m = t > kwargs['t0']

    rise_exp = -(t - kwargs['t0']) / kwargs['tr']
    decay_exp = -(t - kwargs['t0']) / kwargs['td']

    exp_rise = np.exp(rise_exp)
    exp_decay = np.exp(decay_exp)

    result = np.full(t.size, kwargs['c']).astype(np.float64)
    if np.abs(kwargs['tr'] - kwargs['td']) < tol:
        result[m] -= np.abs(kwargs['A'] * ((t - kwargs['t0']) * exp_rise)[m])
    else:
        result[m] -= np.abs((exp_rise - exp_decay)[m] * kwargs['A'])

    return result


# the flare function for several components
class TimeLine(object):
    """
    Time line function with signal response for several peaks within a time window

    f(t) = sum_i response_i(t)

    """
    def __init__(self, numcomp, function='tesresponse'):
        self._numcomp = numcomp
        self._result = np.array([])
        self._result_quite = 0.
        if function == 'tesresponse':
            self._f = tesresponse
        if function == 'tesresponse_simple':
            self._f = tesresponse_simple
        elif function == 'expflare':
            self._f = expflare
        else:
            raise ValueError("Unknown response function chosen")
        return

    @property
    def numcomp(self):
        return self._numcomp

    @property
    def result(self):
        return self._result

    @property
    def result_quite(self):
        return self._result_quite

    @numcomp.setter
    def numcomp(self, numcomp):
        if numcomp < 1:
            raise Exception("There must at least be one component")
        self._numcomp = numcomp
        return

    @property
    def f(self):
        return self._f

    @f.setter
    def f(self, function):
        if function == 'tesresponse':
            self._f = tesresponse
        elif function == 'expflare':
            self._f = expflare
        else:
            raise ValueError("Unknown response function chosen")
        return

    @classmethod
    def read_fit_result(cls, fit_result):
        """
        Initialize class from a fit result dictionary

        :param fit_result:
            dict with fit results loaded from npy file
        :return:
        """
        return cls(fit_result['result']['numcomp'],
                   function=fit_result['parameters']['function'])

    def __call__(self, t, **kwargs):
        """
        Evaluate the function

        :param t: array-like
            time values
        :param kwargs:
            The signal resonse parameters
        :return: array with function values
        """
        kwargs.setdefault('force_zero', True)
        if np.isscalar(t):
            t = np.array([t])
        elif type(t) == list:
            t = np.array(t)

        self._result = np.zeros((self._numcomp, t.size))
        for i in list(range(self._numcomp)):
            self._result[i] = self._f(t,
                                      t0 = kwargs['t0_{0:03n}'.format(i)],
                                      tr = kwargs['tr_{0:03n}'.format(i)],
                                      td = kwargs['td_{0:03n}'.format(i)],
                                      c = 0.,
                                      A = kwargs['A_{0:03n}'.format(i)]
                                      )

        return kwargs['c'] + self._result.sum(axis=0)

    def integral(self, tmin, tmax, tstep=100, **kwargs):
        """
        Calculate integral of each component

        :param tmin: float
            minimum time
        :param tmax: float
            maximum time
        :param tstep:
            number of steps for integration
        :param kwargs:
            function parameters
        :return: array
            integrated signal responses for each component
        """
        tt = []
        for _ in list(range(self._numcomp)):
            tt.append(np.linspace(tmin, tmax, tstep))
        tt = np.array(tt)
        # set self._result in __call__ function
        _ = self.__call__(tt[0], **kwargs)
        return simps(self._result, tt, axis=1)
