import numpy as np
import utils
import meta_profile
import full_aquisition
import matplotlib.pyplot as plt
import os
import re
from scipy import optimize


def get_metadata(path):
    folder = os.path.dirname(path)
    fname = os.path.basename(path)
    data = np.loadtxt(os.path.join(folder, "waveform.csv"))
    ind = int(re.match(r"acq(\d+).csv", fname).groups()[0]) - 1
    return (data[ind, 0], data[ind, 1])


def get_normalized_data(path):
    folder = os.path.dirname(path)
    t = meta_profile.get_aquisition_reltime(path)
    data = utils.read_data_file(path)
    meta = meta_profile.get_meta_profile(folder)
    data[:, 1] -= np.interp(data[:, 0] + t, meta[:, 0], meta[:, 1])
    return data


def get_aligned_data(path, normalize=True):
    if normalize:
        data = get_normalized_data(path)
    else:
        data = utils.read_data_file(path)
    _, t1 = get_ramp_corner_times(path)
    start_ind = utils.get_index_zero_crossings(data[:, 0] - t1)[0] + 1
    mid = (np.max(data[:, 1]) + np.min(data[:, 1])) / 2
    data[:, 1] *= meta_profile.get_crossing_directions(data[:, 1] - mid)[0]
    data = data[start_ind:, :]
    data[:, 1] -= data[0, 1]
    data[:, 0] -= t1
    return data


def get_ramp_corner_times(path):
    data = get_normalized_data(path)
    m, _ = get_metadata(path)
    m *= meta_profile.get_crossing_directions(data[:, 2])[0]

    def f(x, x1, x2, y2):
        b = y2 - m * x2
        y1 = y2 - m * (x2 - x1)
        return np.piecewise(
            x, [x <= x1, x1 < x, x2 <= x], [y1, lambda x: m * x + b, y2]
        )

    p0 = [-0.1, 0.1, data[-1, 2]]
    out = optimize.curve_fit(f, data[:, 0], data[:, 2], p0)
    return (out[0][0], out[0][1])

def get_spline_data(path):
        data = get_aligned_data(path, False)

        data_filt = np.copy(data)
        filt = signal.butter(8, 1 / 5)
        data_filt[:, 1] = signal.filtfilt(*filt, data_filt[:, 1])

        weights = np.abs(data[:, 1] - np.mean(data_filt[:, 1]))
        weights /= np.mean(weights)
        spl = interpolate.UnivariateSpline(
            data_filt[:, 0], data_filt[:, 1], w=weights, s=2e-15, k=3
        )
        return spl


if __name__ == "__main__":
    path = "data/sn0001/07-14-2021/acq0004.csv"

    data = get_aligned_data(path, False)

    plt.plot(data[:, 0], data[:, 1])
    plt.show()
