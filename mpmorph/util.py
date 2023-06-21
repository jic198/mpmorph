import numpy as np
import collections

__author__ = 'Jianli Cheng, Eric Sivonxay'


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


def get_diffusivity(structure, frac_coords, lattices, step_skip, time_step, t_range):
    dt = np.arange(len(frac_coords)) * time_step * step_skip
    if len(t_range) < 2:
        t_range.append(dt[-1])
    diffs = {}
    for ele, msd in get_msd(structure, frac_coords, lattices, step_skip, time_step).items():
        x = np.array([])
        y = np.array([])
        for i, v in enumerate(dt):
            if t_range[0] < v < t_range[1]:
                x = np.append(x, v)
                y = np.append(y, msd[i])
        a = np.ones((len(x), 2))
        a[:, 0] = x
        (m, c), _, _, _ = np.linalg.lstsq(a, y, rcond=None)
        diffs[ele] = m / 60 / len(structure.indices_from_symbol(ele))
    return diffs


def get_msd(structure, frac_coords, lattices, step_skip, time_step):
    frac_coords = np.concatenate(frac_coords, axis=1)
    dp = frac_coords[:, 1:] - frac_coords[:, :-1]
    dp = dp - np.round(dp)
    f_disp = np.cumsum(dp, axis=1)
    c_disp = []
    for i in f_disp:
        c_disp.append([np.dot(d, m) for d, m in zip(i, lattices[1:])])
    c_disp = np.array(c_disp)
    wts = [site.species.weight for site in structure]
    dc = []
    for i in range(len(frac_coords)):
        frame = c_disp[:, i, :]
        center = np.sum([v * wts[i] for i, v in enumerate(frame)], axis=0)
        dc.append(frame - center / sum(wts))
    dc = np.array(dc)
    dt = np.arange(len(frac_coords)) * time_step * step_skip
    msds = {}
    for ele in structure.composition.elements:
        ele = str(ele)
        indices = structure.indices_from_symbol(ele)
        sp_disp = dc[:, indices, :]
        msd = np.zeros(len(dt))
        n_atoms = len(indices)
        for atom_num in range(n_atoms):
            msd_temp = msd_fft(sp_disp[:, atom_num, :])
            msd += msd_temp
        msds[ele] = msd
    return msds