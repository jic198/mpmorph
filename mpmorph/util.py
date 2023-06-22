import re
import numpy as np
import collections
from monty.io import zopen
from scipy.optimize import curve_fit
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsBox

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


def get_diffusivity(structures, step_skip, time_step, t_range):
    dt = np.arange(len(structures)) * time_step * step_skip
    if len(t_range) < 2:
        t_range.append(dt[-1])
    diffs = {}
    for ele, msd in get_msd(structures, step_skip, time_step).items():
        x = np.array([])
        y = np.array([])
        for i, v in enumerate(dt):
            if t_range[0] < v < t_range[1]:
                x = np.append(x, v)
                y = np.append(y, msd[i])
        a = np.ones((len(x), 2))
        a[:, 0] = x
        (m, c), _, _, _ = np.linalg.lstsq(a, y, rcond=None)
        diffs[ele] = m / 60 / len(structures[0].indices_from_symbol(ele))
    return diffs


def get_msd(structures, step_skip, time_step):
    p, l = [], []
    for s in structures:
        p.append(np.array(s.frac_coords)[:, None])
        l.append(s.lattice.matrix)
    p.insert(0, p[0])
    l.insert(0, l[0])
    p = np.concatenate(p, axis=1)
    dp = p[:, 1:] - p[:, :-1]
    dp = dp - np.round(dp)
    f_disp = np.cumsum(dp, axis=1)
    c_disp = []
    for i in f_disp:
        c_disp.append([np.dot(d, m) for d, m in zip(i, l[1:])])
    c_disp = np.array(c_disp)
    wts = [site.species.weight for site in structures[0]]
    dc = []
    for i in range(len(structures)):
        frame = c_disp[:, i, :]
        center = np.sum([v * wts[i] for i, v in enumerate(frame)], axis=0)
        dc.append(frame - center / sum(wts))
    dc = np.array(dc)
    dt = np.arange(len(structures)) * time_step * step_skip
    msds = {}
    for ele in structures[0].composition.elements:
        ele = str(ele)
        indices = structures[0].indices_from_symbol(ele)
        sp_disp = dc[:, indices, :]
        msd = np.zeros(len(dt))
        n_atoms = len(indices)
        for atom_num in range(n_atoms):
            msd_temp = msd_fft(sp_disp[:, atom_num, :])
            msd += msd_temp
        msds[ele] = msd
    return msds


def linear(x, k, b):
    return k * x + b


def fit_arrhenius(temps, diffusivities, weight):
    t_1 = 1 / np.array(temps)
    logd = np.log(diffusivities)
    [slope, intercept], cov = curve_fit(linear, t_1, logd, sigma=weight)
    return slope, intercept, cov


def get_extrapolated_diffusivity(temps, diffusivities, weight, t):
    slope, intercept, cov = fit_arrhenius(temps, diffusivities, weight)
    slope_sigma = np.sqrt(np.diag(cov))[0]
    intercept_sigma = np.sqrt(np.diag(cov))[1]
    log_d = slope * (1 / t) + intercept
    log_d_sigma = ((slope_sigma * (1 / t)) ** 2 + intercept_sigma ** 2) ** 0.5
    d_min = np.exp(log_d - log_d_sigma)
    d_max = np.exp(log_d + log_d_sigma)
    return np.exp(log_d), [d_min, d_max]


def get_structure_from_lammps(filename, element_profile):
    with zopen(filename, 'rt') as f:
        lines = f.read()
    bounds_pattern = re.compile('pp\n(.*)\nITEM', re.S)
    bounds = bounds_pattern.findall(lines)[0].split('\n')[:3] 
    bounds = np.array([b.split() for b in bounds], dtype=np.float64)
    orth_bd = bounds[:,:2]
    lattice = LammpsBox(bounds=orth_bd).to_lattice()
    info_pattern = re.compile('xs ys zs(.*)', re.S)
    infos = info_pattern.findall(lines)[0].split('\n')[1:-1]
    infos = np.array([info.split() for info in infos], dtype=np.float64)
    coords = infos[:, -3:]
    species = [element_profile[i] for i in infos[:, 1].astype(np.int32)]
    return Structure(lattice=lattice, species=species, coords=coords)