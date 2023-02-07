#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a Python implementation of damping identification
method using Morlet wave, based on [1]_:

.. [1] J. Slavič, M. Boltežar, Damping identification with the Morlet-
wave, Mechanical Systems and Signal Processing. 25 (2011) 1632–1645.
_doi: 10.1016/j.ymssp.2011.01.008.
.. _doi: https://doi.org/10.1016/j.ymssp.2011.01.008

Check web page of EU project `NOSTRADAMUS`_ for more info.
.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php

This code is based on the `MorletDamping`_ code developed by WANG Longqi.
.. _MorletDamping: https://github.com/wanglongqi/MorletDamping

Created on Wed 09 Feb 2022 06:45:45 PM CET

@author: TOMAC Ivan, WANG Longqi, SLAVIČ Janko
..meta::
    :keywords: damping, morlet-wave, identification
"""

import numpy as np
from scipy.optimize import ridder
from scipy.optimize import newton
from scipy.optimize import minimize_scalar
from scipy.special import erf
from warnings import warn

class MorletWave(object):
    
    def __init__(self, free_response, fs):
        """
        Initiates the MorletWave object

        :param free_response: analyses signal
        :param fs:  frequency of sampling
        :return:
        """
        self.free_response = free_response
        self.fs = fs

    def identify_damping(self, w, n_1=8, n_2=12, k=30, find_exact_freq=True,
                         root_finding='closed-form', damping_ratio_init='auto'):
        """
        Identify damping at circular frequency `w` (rad/s)

        :param w: circular frequency at which to identify damping
        :param find_exact_freq:  if True, parameter `w` is used as the initial circular
                frequency to find the local extreme
        :param root_finding: root finding method: 
                `closed-form` - for closed form solution
                `Newton` - for the exact-form solution which requires root finding using Newton method
                `Ridder` - for the exact-form solution which requires root finding using Ridder method
        :param damping_ratio_init: initial guess for damping ratio, if `auto` then the closed-form 
                                   solution will be used for initial guess
        :return: damping_ratio
        """
        if n_1>n_2:
            raise Exception('`n_1` should be smaller than `n_2`.')
        if find_exact_freq:
            w = self.find_natural_frequency(omega=w, n=n_1, k=k)

        M = np.abs(self.morlet_integrate(w, n=n_1, k=k)) \
          / np.abs(self.morlet_integrate(w, n=n_2, k=k))
        if root_finding == 'closed-form':
            damping_ratio = self.closed_form_mwdi(M_numerical=M, n_1=n_1, n_2=n_2, k=k)
        else:
            damping_ratio = self.exact_mwdi(M_numerical=M, n_1=n_1, n_2=n_2, k=k, 
                            root_finding=root_finding, damping_ratio_init=damping_ratio_init)

        if self.theoretical_condition_satisfied(k, n_1, damping_ratio):
            k_suggested = self.get_k_suggestion(damping_ratio, n_1)
            if k < k_suggested[0] and self.theoretical_condition_satisfied(k_suggested[0], n_1, damping_ratio):
                warn(f'Low k({k}) value used, possible options: {k_suggested} or {self.get_k_suggestion(damping_ratio)}.', Warning)
            return damping_ratio
        else:
            raise Exception(f'Parameter `k` should be below {n_1**2/(8*np.pi*damping_ratio)}, see Eq. (21) in [1].')

    def morlet_integrate(self, omega, n, k):
        """
        Integration with a Morlet wave at circular freq `omega`, time-spread
        parameter `n` and number of oscillation `k`.
        
        :param w: circular frequency (rad/s)
        :param n: time-spread parameter
        :param k: number of MW's oscillations
        :return: a wavelet coefficient
        """

        kernel = np.conj(self.morlet_wave(omega, n, k))
        npoints = kernel.size

        return np.trapz(self.free_response[:npoints] * kernel, dx=1/self.fs) # eq (15)

    def morlet_wave(self, omega, n, k):
        """
        Function generates Morlet-Wave basic wavelet function on the circular 
        frequency `omega` with `k` cycles and `n` time spread.
        
        :param omega: circular frequency (rad/s)
        :param n: time-spread parameter
        :param k: number of oscillations
        :return:
        """
        eta = 2 * k * np.pi * np.sqrt(2) / n
        s = eta / omega
        T = 2 * k * np.pi / omega
        N = int(self.fs * T) + 1
        
        t = np.arange(N) / self.fs
        t -= 0.5 * T
        t /= s
    
        return np.pi**-0.25 * s**-0.5 * np.exp(-0.5*t**2 + 1j*eta*t)

    def find_natural_frequency(self, omega, n, k):
        """
        Finds local maximum of the Morlet integral at `w`, `n` and `k`.

        :param omega: circular frequency (rad/s)
        :param n: time-spread parameter
        :param k: number of oscillations for the damping identification
        :return:
        """
        # delta = omega * n / (4 * k * np.pi)
        delta = frequency_spread_mw(omega, n, k)
        lwr = omega - delta
        upr = lwr + 2*delta

        def func(omega, n, k):
            return -np.abs(self.morlet_integrate(omega=omega, n=n, k=k))

        mnm = minimize_scalar(func, bounds=(lwr, upr), args=(n, k), \
                        method='bounded', options={'maxiter': 40, 'disp': 0})
        return mnm.x
    
    def frequency_correction(self, damping_ratio, n, k):
        """
        Calculates frequency correction [1]_ caused by the wavelet transform `n`, `k` and
        the damping `d`. Identified frequency needs to be corrected using the returned
        factor in following way: `omega = omega_identified * correction`

        .. [1] J. Slavič, I. Simonovski, M. Boltežar, Damping identification using a 
        continuous wavelet transform: application to real data, Journal of Sound
        and Vibration, 262 (2003) 291-307, _doi: 10.1016/S0022-460X(02)01032-5.
        .. _doi: https://doi.org/10.1016/S0022-460X(02)01032-5.

        :param damping_ratio: damping ratio
        :param n: time-spread parameter
        :param k: number of oscillations for the damping identification
        """
        correction = (np.pi*k/n**2) * (-8*k*np.pi + (n**2 * damping_ratio)/np.sqrt(1 - damping_ratio**2) \
                        + np.sqrt((-n**4*damping_ratio**2 \
                            + 64*k**2*np.pi**2 * (-1 + damping_ratio**2) \
                            + 16*n**2 * (-1 + 2*damping_ratio**2 + k*np.pi*damping_ratio*np.sqrt(1 - damping_ratio**2))) \
                            / (-1 + damping_ratio**2)))
        return correction**-1

    def get_k_suggestion(self, damping_ratio, n=None):
        """
        Get `k` where frequency correction is not required

        Function calculates `k` value for which frequency correction is negligible.
        The expression for frequency correction is analytically solved for `k` value
        where correction factor is equal to 1.

        :param damping_ratio: damping ratio
        :param n: time-spread parameter; if not given the simplified solution for
                low damping ratio is used
        :return: returns tuple (k_1, k_2) where `k_i` is the suggested `k`.
        """   
        if not n:
            return int(1 / (np.pi*damping_ratio))

        k_lo = ((n**2 - n * np.sqrt(n**2 - 16)) * np.sqrt(1 - damping_ratio**2)) \
             / (16 * np.pi * damping_ratio)
        k_hi = ((n**2 + n * np.sqrt(n**2 - 16)) * np.sqrt(1 - damping_ratio**2)) \
             / (16 * np.pi * damping_ratio)

        return int(k_lo), int(k_hi)

    def theoretical_condition_satisfied(self, k, n_1, damping_ratio):
        """
        Checks if the theoretical condition defined with Eq. (21) in [1] is satisfied.

        :param k: number of oscillations for the damping identification
        :param n_1: time-spread parameter
        :param damping_ratio: damping ratio
        """
        return k <= n_1**2/(8*np.pi*damping_ratio)

    def exact_mwdi_goal_function(self, damping_ratio, M_numerical, n_1, n_2, k):
        """
        The goal function of the exact approach 

        This is the implementation of Eq 19 in the [1] and identifies the difference
        between analytically expressed ratio of absolute value
        of two wavelet coefficients expressed with: `d`, `k`, `n_1`, `n_2` and
        numerically calculated ratio `M_numerical`, which is obtained by integrating
        a free response of mechanical system with the Morlet wave function using the
        same parameters: `k`, `n_1` and `n_2` as used for calculation of analytical
        ratio.

        :param damping_ratio: damping ratio
        :param M_numerical: numerical ratio of abs valued of two wavelet coefficients
        :param n_1: time-spread parameter of nominators wavelet coeff.
        :param n_2: time-spread parameter of denominators wavelet coeff.
        :param k: number of oscillations for the damping identification
        :return: M_analytical - M_numerical
        """
        const = 2 * k * np.pi * damping_ratio / np.sqrt(1 - damping_ratio**2)
        n = np.array([n_1, n_2], dtype=object)
        g_1 = 0.25 * n
        g_2_0 = const / n[0]
        g_2_1 = const / n[1]
        err = erf(g_1[0] - g_2_0) + erf(g_1[0] + g_2_0)
        err /=erf(g_1[1] - g_2_1) + erf(g_1[1] + g_2_1)
        g_2_0 /= n[0]
        g_2_1 /= n[1]
        M_analytical = np.sqrt(n_2 / n_1) \
                     * np.exp(g_2_0 * g_2_1 * (n_2**2 - n_1**2)) \
                     * err
        return M_analytical - M_numerical

    def exact_mwdi(self, M_numerical, n_1, n_2, k, damping_ratio_init='auto', root_finding='Newton'):
        """
        The exact approach by using the root finding algorithm

        Using the Eq (19) [1] and based on `k`, `n_1`, `n_2` it finds the damping_ratio
        which results in `M_numerical`=`M_analytical`

        :param M_numerical: numerical ratio of abs valued of two wavelet coefficients
        :param n_1: time-spread parameter of nominators wavelet coeff.
        :param n_2: time-spread parameter of denominators wavelet coeff.
        :param k: number of oscillations for the damping identification        
        :param damping_ratio_init: initial search value, if 'auto', the closed_form solution is used
        :param root_finding: root finding algorithm to use: `Newton`, `Ridder`
        :return: damping_ratio
        """
        if damping_ratio_init=='auto':
            try:
                damping_ratio_init = self.closed_form_mwdi(M_numerical=M_numerical, n_1=n_1, n_2=n_2, k=k)
            except Exception:
                damping_ratio_init = 2e-3
        
        if root_finding=='Newton':
            damping_ratio, _ = newton(self.exact_mwdi_goal_function, x0=damping_ratio_init, \
                                        args=(M_numerical, n_1, n_2, k), full_output=True)
        elif root_finding=='Ridder':
            x0 = (0, 10*damping_ratio_init)
            damping_ratio, _ = ridder(self.exact_mwdi_goal_function, a=x0[0], b=x0[1], \
                                        args=(M_numerical, n_1, n_2, k), full_output=True, disp=False)

        return damping_ratio

    def closed_form_mwdi(self, M_numerical, n_1, n_2, k):
        """
        The closed-form approach by using the root finding algorithm

        Using the Eq (20) [1] and based on `k`, `n_1`, `n_2` it finds the damping_ratio
        which results in `M_numerical`=`M_analytical`

        :param M_numerical: numerical ratio of abs valued of two wavelet coefficients
        :param n_1: time-spread parameter of nominators wavelet coeff.
        :param n_2: time-spread parameter of denominators wavelet coeff.
        :param k: number of oscillations for the damping identification        
        :return: damping_ratio
        """
        const = np.sqrt(n_1 / n_2) * M_numerical
        if const < 1:
            raise Exception('Invalid set of parameters, try increasing `k` value or increase `n_1/n_2`.')
        damping_ratio = n_1 * n_2 / 2 / np.pi \
                        / np.sqrt(k * k * (n_2 * n_2 - n_1 * n_1)) \
                        * np.sqrt(np.log(const))
        return damping_ratio

def frequency_spread_mw(omega, n, k):
    """
    Frequency spread of the Morlet-wave function, Eq.(12).

    :param omega: circular frequency
    :param n: time spread parameter
    :param k: number of oscillations
    :return: frequency spread in rad/s
    """
    return 0.25 * omega * n / (np.pi * k)

if __name__ == "__main__":
    fs = 100
    t = np.arange(0, 6, 1. / fs)
    w = 2 * np.pi * 10
    damping_ratio = 0.01
    free_response = np.cos(w*t+0.33) * np.exp(-damping_ratio*w*t)
    k = 50

#    Close form
    identifier = MorletWave(free_response=free_response, fs=fs)
    print(f'damping_ratio={damping_ratio:5.4f}, identified={identifier.identify_damping(w=w):5.4f}')
#    Exact
    identifier = MorletWave(free_response=free_response, fs=fs)
    print(f'damping_ratio={damping_ratio:5.4f}, identified={identifier.identify_damping(w, n_1=5, n_2=10, k=k, root_finding="Newton"):5.4f}')

