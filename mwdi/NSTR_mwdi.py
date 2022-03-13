#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a Python implementation of damping identification
method using Morlet wave, based on [1]_:

.. [1] J. Slavič, M. Boltežar, Damping identification with the Morlet-
wave, Mechanical Systems and Signal Processing. 25 (2011) 1632–1645.
doi:10.1016/j.ymssp.2011.01.008.

Check web page of EU project `NOSTRADAMUS`_ for more info.
.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php

This code is based on the `MorletDamping`_ code developped by WANG Longqi.
.. _MorletDamping: https://github.com/wanglongqi/MorletDamping

Created on Wed 09 Feb 2022 06:45:45 PM CET

@author: TOMAC Ivan, SLAVIČ Janko
..meta::
    :keywords: damping, morlet-wave, identification
"""
import numpy as np
from scipy.optimize import ridder
from scipy.optimize import minimize_scalar
from scipy.special import erf

class MorletWave(object):
    _root_finding = "exact"

    def __init__(self, sig, fs, k, n1=5, n2=10):
        """
        :param sig: analysed signal
        :param fs:  frequency of sampling
        :param k: number of oscillations for the damping identification
        :param n1: time-spread parameter
        :param n2: time-spread parameter
        :return:
        """
        self.sig = sig
        self.fs = float(fs)
        self.k = float(k)
        self.n1 = float(n1)
        self.n2 = float(n2)

    def identify_damping(self, w, verb=False):
        """
        Identify damping at circular frequency `w` (rad/s)

        """
        M = self.morlet_integrate(w, self.n1) \
          / self.morlet_integrate(w, self.n2)
        if self._root_finding == "close":
            dmp = self.n1 * self.n2 / 2 / np.pi \
                / np.sqrt(self.k * self.k * (self.n2 * self.n2 - self.n1 * self.n1)) \
                * np.sqrt(np.log(np.sqrt(self.n1 / self.n2) * M))
        else:
            self.x0 = (0, 0.01)
            # eq (19):
            eqn = lambda x: -M + np.exp((2 * np.pi * self.k * x / (self.n1*self.n2))**2 \
                        * (self.n2**2 - self.n1**2)) * np.sqrt(self.n2/self.n1) \
                        * (erf(2 * np.pi * self.k * x / self.n1 + self.n1 / 4) \
                            - erf(2 * np.pi * self.k * x / self.n1 - self.n1 / 4)) \
                        / (erf(2 * np.pi * self.k * x / self.n2 + self.n2 / 4) \
                            - erf(2 * np.pi * self.k * x / self.n2 - self.n2 / 4))

            try:
                dmp, r = ridder(eqn, a=self.x0[0], b=self.x0[1], maxiter=20, \
                                full_output=True, disp=False)
                if not r.converged:
                    dmp = np.NaN
                    if verb:
                        print('maximum iterations limit reached!')
            except RuntimeWarning:
                if verb:
                    print('Ridder raised Warning.')
            except ValueError:
                dmp = np.NaN

        if dmp <= self.n1**2/(8*np.pi*self.k):
            return dmp
        else:
            if verb:
                print('Damping theoretical limit not satisfied!')
            return np.NaN

    def set_root_finding(self, method):
        self._root_finding = method

    def morlet_integrate(self, w, n):
        """
        Perform the numerical integration with a Morlet wave at circular freq `w` and time-spread
        parameter `n`. Returns negated absolute value of the integral.

        :param n: time-spread parameter
        :param w: circular frequency (rad/s)
        :return:
        """
        eta = 2 * np.sqrt(2) * np.pi * self.k / n # eq (14)
        s = eta / w
        T = 2 * self.k * np.pi / w # eq (12)
        if T > (self.sig.size / self.fs):
            # print("err: ", w)
            raise ValueError("Signal is too short, %d points are needed" % np.around(T * self.fs))
            return np.nan
        npoints = int(np.around(T * self.fs))
        t = np.arange(npoints) / self.fs
        # From now on `t` is `t - T/2`
        t -= T/2
        kernel = np.exp(-t * t / s / s / 2) * np.exp(-1j * eta * t / s)
        kernel *= 1 / (np.pi ** 0.25 * np.sqrt(s))

        return -np.abs(np.trapz(self.sig[:npoints] * kernel, dx=1/float(self.fs))) # eq (15)

    def identify_natural_frequency(self, w, n):
        """
        Numerically searches for the maximum of morlet-integral in the small range
        arround circular frequency `w`. Search range is set to the frequency spread
        of the morlet-wave function parameters `k` and `n2`.

        :param w: circular frequency (rad/s)
        :return:
        """
        delta = w * self.n2 / (2 * self.k)
        lwr = w - 0.5 * delta
        upr = lwr + delta
        mnm = minimize_scalar(self.morlet_integrate, bounds=(lwr, upr), args=(n),\
                        method='bounded', options={'maxiter': 40, 'disp': 0})
        return mnm.x

if __name__ == "__main__":
    fs1 = 100
    t1 = np.arange(0, 6, 1. / fs1)
    w1 = 2 * np.pi * 10
    sig1 = np.cos(w1 * t1) * np.exp(-0.02 * w1 * t1)
    k1 = 40

#    Close form
    identifier = MorletWave(sig1, fs1, k1, 10, 20)
    identifier.set_root_finding(method="close")
    print(identifier.identify_damping(w1))
#    Exact
    identifier = MorletWave(sig1, fs1, k1, 5, 10)
    print(identifier.identify_damping(w1))
