import numpy as np
import matplotlib.pyplot as plt

import h5py
from cycler import cycler

from scipy.signal import welch
from scipy.signal import butter, sosfilt

def load_plotting_setting():
    cmap = plt.colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0.2, 0.95, 5))

    default_cycler = cycler(color=colors)
    
    params = {'figure.figsize': (7, 5),
              'axes.prop_cycle': default_cycler,
              'axes.titlesize': 14,
              'legend.fontsize': 12,
              'axes.labelsize': 14,
              'axes.titlesize': 14,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top': True,
              'ytick.right': True
              }
    plt.rcParams.update(params)

def get_timestreams(file, channels=['d'], attrs=['delta_t']):
    timestreams, attrs_ret = [], []

    with h5py.File(file, 'r') as f:
        for attr in attrs:
            attrs_ret.append(f['data'].attrs[attr])

        for channel in channels:
            _ = f['data'][f'channel_{channel}'][:] * f['data'][f'channel_{channel}'].attrs['adc2mv'] / 1000
            timestreams.append(_)
        f.close()

    return attrs_ret, timestreams

def load_timestreams(file, channels=['C']):
    timestreams = []
    delta_t = None
    if file[-4:] == '.mat':
        import scipy.io as sio
        data = sio.loadmat(file)
        delta_t = data['Tinterval'][0,0]

        for c in channels:
            timestreams.append(data[c][:,0])

    if file[-5:] == '.hdf5':
        f = h5py.File(file, 'r')
        for c in channels:
            # Convert mv to V
            adc2mv = f['data'][f'channel_{c.lower()}'].attrs['adc2mv']
            timestreams.append(f['data'][f'channel_{c.lower()}'][:] * adc2mv / 1000)

        if delta_t is None:
                delta_t = f['data'].attrs['delta_t']
        f.close()
            
    return delta_t, timestreams

def get_psd(dt=None, tt=None, zz=None, nperseg=None):
    if dt is not None:
        fs = int(np.round(1 / dt))
    elif tt is not None:
        fs = int(np.ceil(1 / (tt[1] - tt[0])))
    else:
        raise SyntaxError('Need to supply either `dt` or `tt`.')
    
    if nperseg is None:
        nperseg = fs / 10
    ff, pp = welch(zz, fs=fs, nperseg=nperseg)
    return ff, pp

def get_area_driven_peak(ffd, ppd, passband=(88700, 89300), noise_floor=None, plot=False):
    """Integrate power in PSD over passband"""
    if noise_floor is None:
        noise_idx = np.logical_and(ffd > 100000, ffd < 105000)
        noise_floor = np.mean(ppd[noise_idx])
    
    all_idx = np.logical_and(ffd > passband[0], ffd < passband[1])
    area_all = np.trapz(ppd[all_idx]-noise_floor, ffd[all_idx]*2*np.pi)
    v2_drive = area_all / (2 * np.pi)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(ffd[all_idx], ppd[all_idx])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Spectral density ($V^2 / Hz$)')
        ax.set_yscale('log')
        plt.show()

    return v2_drive

def get_c_mv(data_files_ordered, vp2p, omegad, passband, charge=3, n_chunk=10, efield=106, return_psds=False):
    m = 2000 * (83e-9**3) * (4 / 3) * np.pi  # sphere mass
    
    ffss, ppss = [], []
    for file in data_files_ordered:
        dtt, nn = load_timestreams(file, ['D'])
        zz = nn[0]

        size_per_chunk = int(zz.size / n_chunk)
        ffs, pps = [], []

        for i in range(n_chunk):
            ff, pp = get_psd(dt=dtt, zz=zz[i*size_per_chunk : (i+1)*size_per_chunk], nperseg=2**16)
            ffs.append(ff)
            pps.append(pp)

        ffss.append(ffs)
        ppss.append(pps)
        
    c_cals = []
    for i, vpp in enumerate(vp2p):
        fd0 = (vpp / 2) * efield * charge * 1.6e-19

        c_cal = []
        for j, ff in enumerate(ffss[i]):
            pp = ppss[i][j]
            v2_drive = get_area_driven_peak(ff, pp, passband=passband, noise_floor=0, plot=False)

            idx_band = np.logical_and(ff > 30000, ff < 80000)
            omega0 = 2 * np.pi * ff[idx_band][np.argmax(pp[idx_band])]
            z2_drive = (fd0**2 / 2) / ((m * (omega0**2 - omegad**2))**2)

            c_cal.append(v2_drive / z2_drive)
        c_cals.append(c_cal)
    
    c_mvs = np.sqrt(1 / np.asarray(c_cals))

    if return_psds:
        return c_mvs, ffss, ppss

    return c_mvs


def demodulate(_sig, _lo, f_samp, f_lp, sos_filt=None):
    mixed_sig = _sig * _lo

    if sos_filt is None:
        sos_filt = butter(N=8, Wn=f_lp, btype='lowpass', output='sos', fs=f_samp)
    filtered = sosfilt(sos_filt, mixed_sig)
    
    return filtered

def get_eb_comp(dt, zz, gg, sos_filt=None):
    fs = int(1 / dt)  # Sample frequency

    ee = demodulate(zz, gg, fs, 10, sos_filt)
    bb = demodulate(zz, np.gradient(gg), fs, 10, sos_filt)

    return ee, bb

def normalized_drive(sig_drive):
    """Normalized the drive signal to have amplitude 1"""
    mean = np.mean(sig_drive[0:10000])
    amp = np.max(np.abs(sig_drive[0:10000] - mean))
    return (sig_drive-mean) / amp

def get_diff_newton_nocorr(bb_0, bb_1, lockin_2_newton):
    _diff = (bb_0 - bb_1) * lockin_2_newton

    return _diff[1000000:]

def get_diff_newton(ee_0, bb_0, ee_1, bb_1, lockin_2_newton):
    amp = 0.5 * (np.mean(ee_0[1000000:]) + np.mean(ee_1[1000000:]))
    _diff = (bb_0/np.mean(ee_0[1000000:]) - bb_1/np.mean(ee_1[1000000:])) * lockin_2_newton * amp

    # _diff = (bb_0/ee_0 - bb_1/ee_1) * lockin_2_newton * amp

    return _diff[1000000:]