"""
generate amplitude features
"""
import numpy as np
from scipy.signal import hilbert, medfilt
from numpy.lib.stride_tricks import sliding_window_view


def calculate_feature(x, feature='edo', fs=64):
    """calculate the amplitude feature for 1D signal"""

    if feature == "envelope":
        feat_x = np.abs(hilbert(x))
        # include a 1 second median filter
        feat_x = medfilt(feat_x, 2 * fs - 1)

    elif feature == "line_length":
        # assuming step size is 1 sample and window is 1 second:
        l_win = fs
        l_win_h = l_win // 2
        x_pad = np.pad(x, (l_win_h, l_win_h), mode='edge')
        eeg_wins = sliding_window_view(x_pad, l_win)

        # Calculate the length of every 1-second overlapping segment
        line_length = np.sum(np.abs(np.diff(eeg_wins, axis=1)), axis=1)
        if len(line_length) > len(x):
            line_length = line_length[:len(x)]

        feat_x = line_length

    elif feature == "edo":
        # envelope derivative operator
        feat_x = _gen_edo(x)
        feat_x = medfilt(feat_x, 2 * fs - 1)

    elif feature == "peak_to_peak":
        # peak-to-peak amplitude
        l_win = fs
        l_win_h = l_win // 2
        x_pad = np.pad(x, (l_win_h, l_win_h), mode='edge')
        eeg_wins = sliding_window_view(x_pad, l_win)

        peak_to_peak = np.sum(np.abs(eeg_wins) > 12.5, axis=1) / l_win

        if len(peak_to_peak) > len(x):
            peak_to_peak = peak_to_peak[:len(x)]
        elif len(peak_to_peak) < len(x):
            peak_to_peak = np.pad(peak_to_peak, (0, len(x) - len(peak_to_peak)), mode='edge')

        feat_x = medfilt(peak_to_peak, 2 * fs - 1)

    else:
        raise ValueError("unkown method " + feature)

    # for all features, invert as want as positive is IBI and not bursts:
    feat_x = -feat_x.astype(np.float32)

    return feat_x


def _gen_edo(x):
    """Generate EDO Γ[x(n)] from simple formula in the time domain."""
    # 1. check if odd length and if so make even:
    N_start = len(x)
    if (N_start % 2) != 0:
        x = np.hstack((x, 0))

    N = len(x)
    nl = np.arange(1, N - 1)
    xx = np.zeros(N)

    # 2. calculate the Hilbert transform
    h = np.imag(hilbert(x))

    # 3. implement with the central finite difference equation
    xx[nl] = ((x[nl+1] ** 2) + (x[nl-1] ** 2) +
              (h[nl+1] ** 2) + (h[nl-1] ** 2)) / 4 - ((x[nl+1] * x[nl-1] +
                                                       h[nl+1] * h[nl-1]) / 2)

    # trim and zero-pad and the ends:
    x_edo = np.pad(xx[2:(len(xx) - 2)], (2, 2), 'constant', constant_values=(0, 0))

    return(x_edo[0:N_start])
