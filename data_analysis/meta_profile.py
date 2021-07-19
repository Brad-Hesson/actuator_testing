import numpy as np
import matplotlib.pyplot as plt
import utils
import re
from datetime import datetime
import os


def get_crossing_directions(d):
    shape = np.shape(d)
    assert len(shape) == 1
    zcis = utils.get_index_zero_crossings(d)
    return [np.sign(d[zci + 1] - d[zci]) for zci in zcis]


@utils.cache_result()
def get_aquisition_datetime(path):
    pat = re.compile("#Date Time: (.*)$")
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m is not None:
                return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")


def get_folder_aquisition_datetime(folder):
    fs = utils.get_data_files_in_dir(folder)
    return get_aquisition_datetime(fs[0])


def get_aquisition_reltime(path):
    folder = os.path.dirname(path)
    t0 = get_folder_aquisition_datetime(folder)
    return (get_aquisition_datetime(path) - t0).total_seconds()


@utils.cache_result(ttl=60)
def get_aquisition_time_vector(folder):
    fs = utils.get_data_files_in_dir(folder)
    t0 = get_folder_aquisition_datetime(folder)
    ts = []
    num = len(fs)
    for i, f in enumerate(fs):
        print(
            "Aquisition time vector construction: %6.2f%% complete"
            % (100 * (i + 1) / num)
        )
        ts += [(get_aquisition_datetime(f) - t0).total_seconds()]
    return ts


def mutual_mean(ds):
    for d in ds:
        shape = np.shape(d)
        assert len(shape) == 2
        assert shape[1] == 2
    new_ds = utils.mututal_interp(ds)
    return sum(new_ds) / len(new_ds)


@utils.cache_result(ttl=60)
def get_meta_profiles(folder):
    fnames = utils.get_data_files_in_dir(folder)

    vus = np.empty((0, 2), dtype=np.float64)
    vds = np.empty((0, 2), dtype=np.float64)
    first_acq_dt = get_aquisition_datetime(fnames[0])
    num = len(fnames)
    for i, fname in enumerate(fnames):
        print("Meta profile construction: %6.2f%% complete" %
              (100 * (i + 1) / num))
        data = utils.read_data_file(fname)
        acq_dt = get_aquisition_datetime(fname)

        time = data[:, 0]
        signal = data[:, 1]
        drive = data[:, 2]

        zc = utils.get_interp_zero_crossings(time, drive)[0]
        sig_at_zc = np.interp(zc, time, signal)
        acq_time = (acq_dt - first_acq_dt).total_seconds()
        row = np.array([acq_time, sig_at_zc])

        mid = (np.max(signal) + np.min(signal)) / 2
        if get_crossing_directions(signal - mid)[0] > 0:
            vus = np.vstack((vus, row))
        else:
            vds = np.vstack((vds, row))

    return (vus, vds)


@utils.cache_result(ttl=60)
def get_meta_profile(folder):
    vus, vds = get_meta_profiles(folder)
    return mutual_mean((vus, vds))


if __name__ == "__main__":
    folder = "data/sn0001/07-16-2021"

    vs = get_meta_profile(folder)
    vus, vds = get_meta_profiles(folder)

    plt.subplot(2, 1, 1)
    plt.plot(vds[:, 0] / 60 / 60, vds[:, 1] * 1e9)
    plt.plot(vus[:, 0] / 60 / 60, vus[:, 1] * 1e9)
    plt.plot(vs[:, 0] / 60 / 60, vs[:, 1] * 1e9)

    mvus, mvds = utils.mututal_interp((vus, vds))
    mvds[:, 0] *= 0
    mvus[:, 1] -= mvds[:, 1]
    ds = mvus

    plt.subplot(2, 1, 2)
    plt.plot(ds[:, 0] / 60 / 60, ds[:, 1] * 1e9)
    plt.show()
