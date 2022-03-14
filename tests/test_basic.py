import sys, os

my_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, my_path + '/../')

import numpy as np
import mwdi


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




if __name__ == "__main__":
    test_version()
    test_sythetic()

if __name__ == '__mains__':
    np.testing.run_module_suite()
