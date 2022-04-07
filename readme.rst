MWDI - Morlet-Wave Damping Identification 
------------------------------------------
This is the Python implementation of the Morlet-Wave damping identification method, see [1]_ and [2]_ for details.

This package is based on the `MorletDamping`_ code developed by WANG Longqi and was created within the 
MSCA IF project `NOSTRADAMUS`_.


Simple example
---------------

A simple example how to identify damping using MWDI method:

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
   identifier = mw.MorletWave(free_response=response, fs=fs)

   # identify damping
   dmp = identifier.identify_damping(w=w_n, root_finding='Newton')
   print(dmp)

References
----------
.. [1] J\. Slavič, M. Boltežar, Damping identification with the Morlet-wave, Mechanical Systems and Signal Processing, 25 (2011) 1632–1645, doi: `10.1016/j.ymssp.2011.01.008`_.
.. [2] I\. Tomac, J. Slavič, Damping identification based on a high-speed camera. Mechanical Systems and Signal Processing, 166 (2022) 108485–108497, doi: `10.1016/j.ymssp.2021.108485`_.

.. _NOSTRADAMUS: http://ladisk.si/?what=incfl&flnm=nostradamus.php
.. _MorletDamping: https://github.com/wanglongqi/MorletDamping
.. _10.1016/j.ymssp.2011.01.008: https://doi.org/10.1016/j.ymssp.2011.01.008
.. _10.1016/j.ymssp.2021.108485: https://doi.org/10.1016/j.ymssp.2021.108485

|Build Status|

.. |Build Status| image:: https://travis-ci.com/ladisk/mwdi.svg?branch=main
   :target: https://travis-ci.com/ladisk/mwdi
   

