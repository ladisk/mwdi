MWDI - Morlet-Wave damping identification 
------------------------------------------
This is a Python implementation of damping identification method using Morlet wave, based on [1]_.

Check web page of EU project `NOSTRADAMUS`_ for more info.

.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php

This code is based on the `MorletDamping`_ code developped by WANG Longqi.

.. _MorletDamping: https://github.com/wanglongqi/MorletDamping

Usage
-----
The package uses the Morlet wave method to identify structural damping from a single free response (impulse response function) of the mechanical system. The method requires three parameters, to set :code:`n_1`, :code:`n_2` and :code:`k`, that are set to default values. For more details how to select parameters check [1]_ and [2]_.

:code:`identifier = mw.MorletDamping(free_response=response, fs=sampling_frequency, k=30, n_1=5, n_2=10)`

To identify damping a natural circular frequency must be supplied:

:code:`identifier.identify_damping(w=natural_frequency)`

The package has a method to identify natural frequency. User must provide estimated natural frequency.

:code:`identified_nat_freq = identifier.find_natural_frequency(w=estimated_nat_freq, n=5)`

Simple example
---------------

A simple example on how to generate random signals on PSD basis:

.. code-block:: python

   import mwdi as mw
   import numpy as np

   # set time domain
   fs = 5000 # sampling frequency [Hz]
   N = 5000 # number of data points of time signal
   time = np.arange(N) / fs # time vector

   # generate a free response of a SDOF damped mechanical system
   w_n = 2*np.pi * 100 # undamped natural frequency
   d = 0.01 # damping ratio
   x = 1 # amplitude
   phi = 0.3 # phase
   response = x * np.exp(-d * w_n * time) * np.cos(w_n * np.sqrt(1 - d**2) * time - phi)

   # set MWDI object identifier
   identifier = mw.MorletWave(free_response=response, fs=fs, k=40)

   # identify natural frequency
   w_ident = identifier.find_natural_frequency(w=99*2*np.pi, n=5)
   print(w_ident / 2 / np.pi)

   # identify damping
   dmp = identifier.identify_damping(w_ident)
   print(dmp)

References
----------

.. [1] J. Slavič, M. Boltežar, Damping identification with the Morlet-wave, Mechanical Systems and Signal Processing, 25 (2011) 1632–1645, doi: `10.1016/j.ymssp.2011.01.008`_.

.. _10.1016/j.ymssp.2011.01.008: https://doi.org/10.1016/j.ymssp.2011.01.008

.. [2] I. Tomac, J. Slavič, Damping identification based on a high-speed camera. Mechanical Systems and Signal Processing, 166 (2022) 108485–108497, doi: `10.1016/j.ymssp.2021.108485`_.

.. _10.1016/j.ymssp.2021.108485: https://doi.org/10.1016/j.ymssp.2021.108485

.. .. |DOI| |Build Status| |Docs Status|

.. .. |Docs Status| image:: https://readthedocs.org/projects/pyexsi/badge/
..    :target: https://pyexsi.readthedocs.io
   
.. .. |Build Status| image:: https://travis-ci.com/ladisk/pyExSi.svg?branch=main
..    :target: https://travis-ci.com/ladisk/pyExSi
   
.. .. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4431844.svg
..    :target: https://doi.org/10.5281/zenodo.4431844