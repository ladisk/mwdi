import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import mwdi

from scipy.special import erf

def test_version():
    """ check if MWDI exposes a version attribute """
    assert hasattr(mwdi, '__version__')
    assert isinstance(mwdi.__version__, str)

def get_free_response(fs=5000, n=5000, fr=100, damping_ratio=0.01, phase=0.3, amplitude=1.0):
    """
    Calculates a time response of free SDOF damped mechanical system.

    :param fs: sample frequency (S/s)
    :param n: number of samples that will response contain (S)
    :param fr: undamped natural frequency (Hz)
    :param damping_ratio: damping ratio of the system (-)
    :param phase: phase of the response (rad)
    :param amplitude: amplitude of the response
    """
    time = np.arange(0, n) / fs
    w_n = 2 * np.pi * fr
    w_d = w_n * np.sqrt(1 - damping_ratio**2)
    free_response = amplitude * np.cos(w_d * time + phase) * np.exp(-damping_ratio * w_n * time)
    return free_response

def morlet_integral_analytical(k, n, w_n, damping_ratio, phase, amplitude):
    """
    Calculates amplitude and phase of analytically derived integral between free SDOF
    response and the Morlet-Wave basic wavelet function, on a finite interval `[0, T]`.
    
    :param k: number of oscillations
    :param n: time-spread parameter
    :param w_n: undamped natural frequency of the SDOF system (rad/s)
    :param damping_ratio: damping ratio of the system (-)
    :param phase: phase of the response (rad)
    :param amplitude: amplitude of the response
    :return: amplitude, phase (rad/s) of analytical wavelet coefficient
    """
    w_d = w_n * np.sqrt(1 - damping_ratio**2)
    norm = (0.5 * np.pi)**0.75 * amplitude * np.sqrt(k / (n * w_d))
    A = 2 * k * np.pi * damping_ratio * w_n / (n * w_d)
    B = 0.25 * n
    G = erf(A + B) - erf(A - B)
    I_abs = np.exp(A * (A - 2*B))
    I_phase = np.angle(np.exp(1j * (np.pi*k - phase)))
    return norm * I_abs * G, I_phase

def test_sythetic(fs=5000, n=5000, fr=100, damping_ratio=0.01, phase=0.3, amplitude=1):
    signal = get_free_response(fs=fs, n=n, fr=fr, damping_ratio=damping_ratio, 
                               phase=phase, amplitude=amplitude)
    n_1 = 5
    n_2 = 10
    k = 20
    identifier = mwdi.MorletWave(free_response=signal, fs=fs, k=k, n_1=n_1, n_2=n_2)
    w_n = 2*np.pi*fr
    w_d = w_n * np.sqrt(1 - damping_ratio**2)

    ###### Test Morlet Wave integral ######
    mw_integral_num = np.abs(identifier.morlet_integrate(w=w_d, n=n_1))
    mw_integral_anl, _ = morlet_integral_analytical(k=k, n=n_1, w_n=w_n, 
                                                    damping_ratio=damping_ratio,
                                                    phase=phase, amplitude=amplitude)

    np.testing.assert_allclose(mw_integral_num, mw_integral_anl, 1e-3)
    print(f'\nTest Morlet integral:\n\tanalytical={mw_integral_anl},'
          f' numerical={mw_integral_num}')

    ###### Test Identification of natural frequency ######
    w_ident = identifier.find_natural_frequency(w=w_d, n=n_1)
    corr = identifier.frequency_correction(n=n_1, d=damping_ratio)

    np.testing.assert_allclose(w_ident/corr, w_d, 0.5e-3)
    print(f'\nTest find frequency:\n\tw={w_d}, w_corr={w_ident/corr}, w_ident={w_ident}')

    ###### Test identification of damping ratio ######
    # damping_ratio_ident = identifier.identify_damping(w=w)
    # np.testing.assert_allclose(damping_ratio_ident, damping_ratio) 
    # print(f'damping_ratio={damping_ratio}, damping_ratio_ident:{damping_ratio_ident}')



if __name__ == "__main__":
    test_version()
    test_sythetic()

if __name__ == '__mains__':
    np.testing.run_module_suite()
