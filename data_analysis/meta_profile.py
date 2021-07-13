import numpy as np
import matplotlib.pyplot as plt
import utils
import re
from datetime import datetime


def get_crossing_direction(d):
    shape = np.shape(d)
    assert len(shape) == 1
    zci = utils.get_index_zero_crossing(d)
    return np.sign(d[zci + 1] - d[zci])


@utils.cache_result
def get_aquisition_datetime(path):
    pat = re.compile("#Date Time: (.*)$")
    with open(path) as f:
        for line in f:
            m = pat.match(line)
            if m is not None:
                return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")

def mutual_mean(ds):
    for d in ds:
        shape = np.shape(d)
        assert len(shape) == 2
        assert shape[1] == 2
    new_ds = utils.mututal_interp(ds)
    return sum(new_ds) / len(new_ds)


#@utils.cache_result
def get_meta_profiles(folder):
    fnames = utils.get_files_in_dir(folder, True)

    vus = np.empty((0, 2), dtype=np.float64)
    vds = np.empty((0, 2), dtype=np.float64)
    first_acq_dt = get_aquisition_datetime(fnames[0])
    for fname in fnames:
        print(fname)
        data = utils.read_data_file(fname)
        acq_dt = get_aquisition_datetime(fname)

        time = data[:, 0]
        signal = data[:, 1]
        drive = data[:, 2]

        zc = utils.get_interp_zero_crossing(time, drive)
        sig_at_zc = np.interp(zc, time, signal)
        acq_time = (acq_dt - first_acq_dt).total_seconds()
        row = np.array([acq_time, sig_at_zc])

        if get_crossing_direction(drive) > 0:
            vus = np.vstack((vus, row))
        else:
            vds = np.vstack((vds, row))

    return (vus, vds)

def get_meta_profile(folder):
    vus, vds = get_meta_profiles(folder)
    return mutual_mean((vus, vds))


if __name__ == "__main__":
    folder = "data/sn0001"

    vus, vds = get_meta_profiles(folder)
    mvus, mvds = utils.mututal_interp((vus, vds))
    vs = mutual_mean((vus, vds))
    mvds[:, 0] *= 0
    ds = mvus - mvds

    plt.subplot(2,1,1)
    plt.plot(vds[:, 0] / 60 / 60, vds[:, 1])
    plt.plot(vus[:, 0] / 60 / 60, vus[:, 1])
    plt.plot(vs[:, 0] / 60 / 60, vs[:, 1])

    plt.subplot(2,1,2)
    plt.plot(ds[:, 0] / 60 / 60, ds[:, 1])
    plt.show()
