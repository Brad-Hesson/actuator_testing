import numpy as np
import utils
import meta_profile
import full_aquisition
import matplotlib.pyplot as plt
import os
import re
from scipy import interpolate, optimize, signal, ndimage, stats


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
    data = utils.read_data_file(path)
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

    weights = np.abs(data[:, 1] - np.mean(data[:, 1]))
    weights /= np.mean(weights)
    spl = interpolate.UnivariateSpline(data[:, 0], data[:, 1], w=weights, k=4)
    spl.set_smoothing_factor(8e-15)
    return spl


def logtimize(path):
    xs = utils.read_data_file(path)[:, 0]
    spline = get_spline_data(path)
    ys = spline.derivative(1)(xs)

    def f(x):
        lxs = np.log(xs + x[0])
        lys = np.log(ys + x[1])
        mask = ~np.isnan(lxs) & ~np.isnan(lys)
        slope, intercept, r_value, p_value, std_err = stats.linregress(lxs[mask], lys[mask])
        print(slope, intercept)
        plt.plot(np.log(xs + x[0]), np.log(ys + x[1]))
        plt.plot(np.log(xs + x[0]), slope*np.log(xs + x[0])+intercept)
        plt.show()
        return std_err

    res = optimize.minimize(f, [0,0])
    print(res)


if __name__ == "__main__":
    plt.style.use(["dark_background", "seaborn-deep"])
    folder = "data/sn0001/07-15-2021"
    paths = utils.get_data_files_in_dir(folder)
    last_metadata = get_metadata(paths[0])
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    for path in paths:
        metadata = get_metadata(path)
        if last_metadata != metadata:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.tight_layout()
            plt.show()
            ax1 = plt.subplot(2, 1, 1)
            ax2 = plt.subplot(2, 1, 2)
        last_metadata = metadata

        data = utils.read_data_file(path)
        mid = (np.max(data[:, 1]) + np.min(data[:, 1])) / 2
        direction = meta_profile.get_crossing_directions(data[:, 1] - mid)[0]
        color = "green" if direction > 0 else "red"

        data = get_aligned_data(path, False)

        spl = get_spline_data(path)

        ax1.set_title(path)
        ax1.plot(data[:, 0], data[:, 1], color=color, alpha=0.4)
        ax1.plot(data[:, 0], spl(data[:, 0]), color=color)

        xs = data[:, 0]
        ys = spl.derivative(1)(xs)
        ax2.plot(np.log(xs + 0.5), np.log(ys+1e-11), color=color)
        # ax2.set_ylim(-1e-10, 1e-9)

        # def f(x, a, w, b, c):
        #     return a * np.power(x + w, -b) + c

        # xs = np.linspace(20, data[-1, 0], 1600)
        # try:
        #     out = optimize.curve_fit(
        #         f,
        #         xs,
        #         spl.derivative(1)(xs),
        #     )
        #     a = out[0][0]
        #     w = out[0][1]
        #     b = out[0][2]
        #     c = out[0][3]

        #     plt.plot(xs, f(xs, *out[0]), "--")
        # except RuntimeError:
        #     pass
