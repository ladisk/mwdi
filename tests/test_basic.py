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

def get_free_response(fs=1000, n=1000, fr=100, damping_ratio=0.01, phase=0.3, amplitude=1.0):
    time = np.arange(0, n) / fs
    w = 2 * np.pi * fr
    free_response = amplitude*np.cos(w * time+phase) * np.exp(-damping_ratio * w * time)
    return free_response


def test_sythetic(fs=1000, n=1000, fr=100, damping_ratio=0.01, phase=0.3, amplitude=1):
    signal = get_free_response(fs=fs, n=n, fr=fr, damping_ratio=damping_ratio, 
                               phase=phase, amplitude=amplitude)
    n_1 = 10
    n_2 = 20
    k = 10
    identifier = mwdi.MorletWave(free_response=signal, fs=fs, k=k, n_1=n_1, n_2=n_2)
    w = 2*np.pi*fr
    
    w_ident = identifier.find_natural_frequency(w=w, k=k, n=n_1)
    np.testing.assert_equal(w_ident, w)
    print(f'w={w}, w_ident={w_ident}')

    damping_ratio_ident = identifier.identify_damping(w=w)
    np.testing.assert_equal(damping_ratio_ident, damping_ratio) 
    print(f'damping_ratio={damping_ratio}, damping_ratio_ident:{damping_ratio_ident}')

def morlet_integral_analytical(k=10, n=10, w_n=100*np.pi, damping_ratio=0.01, amplitude=1, phase=0.3):
    w_d = w_n * np.sqrt(1 - damping_ratio**2)
    const = (np.pi/2)**(3/4) * amplitude * np.sqrt(k / (n * w_d))
    A = 2*np.pi*k*damping_ratio*w_n / (n*w_d)
    B = 0.25 * n
    error_function = erf(A + B) - erf(A - B)
    integral = np.exp(4*(np.pi*damping_ratio*k*w_n/(n*w_d))**2 - np.pi*damping_ratio*k*w_n/w_d + 1j*(np.pi*k - phase))
    return const * integral * error_function

def morlet_integral_analytical_paper(k=10, n=10, w=100*np.pi, damping_ratio=0.01, amplitude=1):
#     const = amplitude * (2*np.pi**3)**0.25 * np.sqrt(k/(n*w)) # from papaer (17)
#     const = amplitude * (0.5*np.pi)**0.75 * np.sqrt(k/(n*w)) # from paper (A.19)
    const = amplitude * (np.pi/2)**(3/4) * np.sqrt(k/(n*w))
    A = 2*np.pi*k*damping_ratio/n
    B = 0.25 * n
    error_function = erf(A + B) - erf(A - B)
    integral = np.exp(np.pi*k*damping_ratio*(4*np.pi*k*damping_ratio - n**2)/n**2)
    return const * integral * error_function


if __name__ == "__main__":
    test_version()
    test_sythetic()

if __name__ == '__mains__':
    np.testing.run_module_suite()
