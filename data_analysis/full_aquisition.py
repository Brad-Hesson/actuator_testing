import numpy as np
import matplotlib.pyplot as plt
import utils
import meta_profile


@utils.cache_result(ttl=60 * 10)
def get_full_aquisition(folder):
    ts = meta_profile.get_aquisition_time_vector(folder)
    fs = utils.get_data_files_in_dir(folder)

    all_data = np.empty((0, 2))
    num = len(fs)
    for f, t, i in zip(fs, ts, range(len(fs))):
        print("Full aquisition construction: %6.2f%% complete" % (100 * (i + 1) / num))
        data = utils.read_data_file(f)[:, [0, 1]]
        data[:, 0] += t
        all_data = np.vstack((all_data, data))
    return all_data


@utils.cache_result(ttl=60 * 10)
def get_normalized_full_aquisition(folder):
    data = get_full_aquisition(folder)
    vs = meta_profile.get_meta_profile(folder)
    data[:, 1] -= np.interp(data[:, 0], vs[:, 0], vs[:, 1])
    return data


if __name__ == "__main__":
    folder = "data/sn0001/07-12-2021"
    # folder = r"data\old\4plate_v1\P10min_A100m_S1p"

    data = get_full_aquisition(folder)
    vs = meta_profile.get_meta_profile(folder)

    plt.subplot(2, 1, 1)
    plt.plot(data[:, 0], data[:, 1])
    plt.plot(vs[:, 0], vs[:, 1])

    data = get_normalized_full_aquisition(folder)

    plt.subplot(2, 1, 2)
    plt.plot(data[:, 0], data[:, 1] * 1000 * 1000 * 1000)
    plt.show()
