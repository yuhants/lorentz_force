import numpy as np
from scipy.signal import butter, sosfilt

import analysis_utils as utils

lockin_2_newton = -3.6076810164602596e-11

def normalized_drive(sig_drive):
    """Normalized the drive signal to have amplitude 1"""
    mean = np.mean(sig_drive[0:10000])
    amp = np.max(np.abs(sig_drive[0:10000] - mean))
    return (sig_drive-mean) / amp

def get_diff_newton(ee_0, bb_0, ee_1, bb_1, lockin_2_newton):
    amp = 0.5 * (np.mean(ee_0[1000000:]) + np.mean(ee_1[1000000:]))
    _diff = (bb_0/ee_0 - bb_1/ee_1) * lockin_2_newton * amp

    return _diff[1000000:]

def demodulate(_sig, _lo, f_samp, f_lp, sos_filt=None):
    mixed_sig = _sig * _lo

    if sos_filt is None:
        sos_filt = butter(N=8, Wn=f_lp, btype='lowpass', output='sos', fs=f_samp)
    filtered = sosfilt(sos_filt, mixed_sig)
    
    return filtered

def get_eb_comp(dt, zz, gg, sos_filt=None):
    fs   = int(1 / dt)  # Sample frequency
    f_lp = 10           # Low-pass frequency (Hz)  

    ee = demodulate(zz, gg, fs, f_lp, sos_filt)
    bb = demodulate(zz, np.gradient(gg), fs, f_lp, sos_filt)

    return ee, bb

if __name__ == '__main__':
    nstart = 2000
    nfiles = 100000

    fs   = 1250000
    f_lp = 10
    sos_filt = butter(N=8, Wn=f_lp, btype='lowpass', output='sos', fs=fs)

    be = np.linspace(-1e-15, 1e-15, 250)
    bc = 0.5 * (be[1:] + be[:-1])

    hist_diff = np.zeros(bc.size, dtype=np.int64)
    hist_diff_nob_0 = np.zeros(bc.size, dtype=np.int64)
    hist_diff_nob_1 = np.zeros(bc.size, dtype=np.int64)

    for i in range(nstart, nfiles):
        if i % 100 == 0:
            print(i)

        _file_0 = rf"E:\lorentz_force\sphere_20250708\20250808_2e-8mbar_midfreq\20250808_m350e_276khz_350vpp_withb_{i}.hdf5"
        _file_1 = rf"E:\lorentz_force\sphere_20250708\20250808_2e-8mbar_midfreq\20250808_m350e_276khz_350vpp_withb_flipped_{i}.hdf5"
        _file_n = rf"E:\lorentz_force\sphere_20250708\20250808_2e-8mbar_midfreq\20250808_m350e_276khz_350vpp_nob_{i}.hdf5"

        # Read in timestreams (positive, negative, no B field)
        attrs, tts = utils.get_timestreams(file=_file_0, channels=['d', 'g'], attrs=['delta_t'])
        dt, zz_0, gg_0 = attrs[0], tts[0], tts[1]

        attrs, tts = utils.get_timestreams(file=_file_1, channels=['d', 'g'], attrs=['delta_t'])
        dt, zz_1, gg_1 = attrs[0], tts[0], tts[1]

        attrs, tts = utils.get_timestreams(file=_file_n, channels=['d', 'g'], attrs=['delta_t'])
        dt, zz_n, gg_n = attrs[0], tts[0], tts[1]

        gg_normalized_0 = normalized_drive(gg_0)
        gg_normalized_1 = normalized_drive(gg_1)
        gg_normalized_n = normalized_drive(gg_n)

        # Calculate the in-phase and out-of-phase signal using the normalized drive
        ee_0, bb_0 = get_eb_comp(dt, zz_0, gg_normalized_0, sos_filt)
        ee_1, bb_1 = get_eb_comp(dt, zz_1, gg_normalized_1, sos_filt)
        ee_n, bb_n = get_eb_comp(dt, zz_n, gg_normalized_n, sos_filt)

        _diff = get_diff_newton(ee_0, bb_0, ee_1, bb_1, lockin_2_newton)
        hh, _ = np.histogram(_diff, bins=be)
        hist_diff += hh

        _diff_0n = get_diff_newton(ee_0, bb_0, ee_n, bb_n, lockin_2_newton)
        hh_0n, _ = np.histogram(_diff_0n, bins=be)
        hist_diff_nob_0 += hh_0n

        _diff_1n = get_diff_newton(ee_1, bb_1, ee_n, bb_n, lockin_2_newton)
        hh_1n, _ = np.histogram(_diff_1n, bins=be)
        hist_diff_nob_1 += hh_1n

        if (i+1) % 500 == 0:
            # Save histograms and re-initiate
            print(f'Saving file {f"hist_diff_{i-499}_{i}.npz"}')
            np.savez(f'hists_processed/hist_diff_{i-499}_{i}.npz', be=be, hist=hist_diff)
            np.savez(f'hists_processed/hist_diff_nob_pos_{i-499}_{i}.npz', be=be, hist=hist_diff_nob_0)
            np.savez(f'hists_processed/hist_diff_nob_neg_{i-499}_{i}.npz', be=be, hist=hist_diff_nob_1)

            hist_diff = np.zeros(bc.size, dtype=np.int64)
            hist_diff_nob_0 = np.zeros(bc.size, dtype=np.int64)
            hist_diff_nob_1 = np.zeros(bc.size, dtype=np.int64)