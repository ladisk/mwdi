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
    free_response = amplitude * np.cos(w_d * time - phase) * np.exp(-damping_ratio * w_n * time)
    return free_response

def morlet_integral_analytical(damping_ratio, n, k, w_n, phase, amplitude):
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
    k = 40
    identifier = mwdi.MorletWave(free_response=signal, fs=fs)
    w_n = 2*np.pi*fr
    w_d = w_n * np.sqrt(1 - damping_ratio**2)

    ###### Test Morlet Wave integral ######
    mw_integral_num = np.abs(identifier.morlet_integrate(w=w_d, n=n_1, k=k))
    mw_integral_anl, _ = morlet_integral_analytical(damping_ratio=damping_ratio,
                                                    n=n_1, k=k, w_n=w_n,
                                                    phase=phase, amplitude=amplitude)

    np.testing.assert_allclose(mw_integral_num, mw_integral_anl, 9.6e-4)
    print(f'\nTest Morlet integral:\n\tanalytical={mw_integral_anl},'
          f' numerical={mw_integral_num}')

    ###### Test Identification of natural frequency ######
    w_ident1 = identifier.find_natural_frequency(w=w_d, n=n_1, k=k)
    w_ident2 = identifier.find_natural_frequency(w=w_d, n=n_2, k=k)
    corr1 = identifier.frequency_correction(damping_ratio=damping_ratio, n=n_1, k=k)
    corr2 = identifier.frequency_correction(damping_ratio=damping_ratio, n=n_2, k=k)

    np.testing.assert_allclose(w_ident1*corr1, w_d, 1.7e-4)
    print(f'\nTest find frequency for n_1:\n\tw={w_d}, w_corr={w_ident1*corr1}, w_ident={w_ident1}')
    np.testing.assert_allclose(w_ident2*corr2, w_d, 1e-5)
    print(f'\nTest find frequency for n_2:\n\tw={w_d}, w_corr={w_ident2*corr2}, w_ident={w_ident2}')

    #### Test identification of damping ratio ######
    damping_ratio_ident = identifier.identify_damping(w=w_d, n_1=n_1, n_2=n_2, k=k, root_finding='Newton')
    np.testing.assert_allclose(damping_ratio_ident, damping_ratio, 4.6e-3) 
    print(f'\nTest damping ratio:\n\tdamping_ratio={damping_ratio}, damping_ratio_ident:{damping_ratio_ident}')

def test_multi_sine(fs=5000, n=5000):
    fr = [100, 300, 600, 800]
    damping_ratio = [0.01, 0.015, 0.02, 0.005]
    phase = [0.3, 1.0, 1.4, 2.2]
    amplitude = [1, .5, 0.2, 0.1]

    signal = np.zeros(n)
    for f_, d_, p_, a_ in zip(fr, damping_ratio,phase,amplitude):
        signal += get_free_response(fs=fs, n=n, fr=f_, damping_ratio=d_, phase=p_, amplitude=a_)

    n_1 = 7
    n_2 = 12
    k = [20, 30, 40, 80]
    identifier = mwdi.MorletWave(free_response=signal, fs=fs, n_1=n_1, n_2=n_2)

    for fr_, d_, k_ in zip(fr, damping_ratio, k):
        w_d = 2*np.pi*fr_
        damping_ratio_ident = identifier.identify_damping(w=w_d, k=k_, root_finding='Newton')
        np.testing.assert_allclose(damping_ratio_ident, d_, 2e-1) 
        print(f'\nTest damping ratio:\n\tdamping_ratio={d_}, damping_ratio_ident:{damping_ratio_ident}')


if __name__ == "__main__":
    #test_version()
    #test_sythetic()
    test_multi_sine()

if __name__ == '__mains__':
    np.testing.run_module_suite()
