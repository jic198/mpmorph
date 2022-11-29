import numpy as np
import collections

__author__ = 'Eric Sivonxay'


def recursive_update(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, collections.abc.Mapping):
            tmp = recursive_update(orig_dict.get(key, {}), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict


def autocorrFFT(x):
    """ Calculates the position autocorrelation function using the fast Fourier transform.
    """
    N=len(x)
    F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
    PSD = F * F.conjugate()
    res = np.fft.ifft(PSD)
    res= (res[:N]).real   #now we have the autocorrelation in convention B
    n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
    return res/n #this is the autocorrelation in convention A


def msd_fft(r):
    """ Calculates mean square displacement of the array r using the fast Fourier transform.
    """
    n = len(r)
    d = np.square(r).sum(axis=1)
    d = np.append(d, 0)
    s2 = sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
    q = 2 * d.sum()
    s1 = np.zeros(n)
    for m in range(n):
        q = q - d[m-1] - d[n-m]
        s1[m] = q / (n - m)
    return s1 - 2 * s2