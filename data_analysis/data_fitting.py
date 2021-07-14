import numpy as np
import utils
import meta_profile
import matplotlib.pyplot as plt
import os


def get_normalized_data(path):
    folder = os.path.dirname(path)
    t = meta_profile.get_aquisition_reltime(path)
    data = utils.read_data_file(path)
    meta = meta_profile.get_meta_profile(folder)
    data[:, 1] -= np.interp(data[:, 0] + t, meta[:, 0], meta[:, 1])
    return data[:, [0, 1]]


if __name__ == "__main__":
    path = "data/sn0001/07-12-2021/acq0100.csv"

    data = get_normalized_data(path)

    plt.plot(data[:, 0], data[:, 1])
    plt.show()
