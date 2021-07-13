import numpy as np
import matplotlib.pyplot as plt
from utils import get_files_in_dir, read_data_file
from meta_profile import (
    get_aquisition_datetime,
    get_meta_profile,
    get_aquisition_time_vector,
)


def get_full_aquisition(folder):
    ts = get_aquisition_time_vector(folder)
    fs = get_files_in_dir(folder)

    all_data = np.empty((0, 2))
    num = len(fs)
    for f, t, i in zip(fs, ts, range(len(fs))):
        print("Full aquisition construction: %6.2f%% complete" % (100 * (i + 1) / num))
        data = read_data_file(f)[:, [0, 1]]
        data[:, 0] += t
        all_data = np.vstack((all_data, data))
    return all_data


def get_normalized_full_aquisition(folder):
    data = get_full_aquisition(folder)
    vs = get_meta_profile(folder)
    data[:, 1] -= np.interp(data[:, 0], vs[:, 0], vs[:, 1])
    return data


if __name__ == "__main__":
    folder = "data/sn0001"
    folder = r"data\old\4plate_v1\P10min_A100m_S1p"

    data = get_full_aquisition(folder)
    vs = get_meta_profile(folder)

    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(vs[:, 0], vs[:, 1])

    plt.subplot(2, 1, 2)
    data[:, 1] -= np.interp(data[:, 0], vs[:, 0], vs[:, 1])
    plt.plot(data[:, 0], data[:, 1] * 1000 * 1000 * 1000)
    plt.show()
