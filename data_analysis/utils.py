import os
import numpy as np
import re

def get_files_in_dir(path, full_path=False):
    iter = os.walk(path)
    path, _, fnames = next(iter)
    if full_path:
        return [os.path.normpath(os.path.join(path, fname)) for fname in fnames]
    else:
        return [os.path.normpath(fname) for fname in fnames]

def read_data_file(path):
    pat = re.compile("^((-?\d+(\.\d+)?(e-?\d+)?),?)+$")
    data_start = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if pat.match(line) is not None:
                data_start = i
                break
    return np.loadtxt(path, delimiter=',', skiprows=data_start, dtype=np.float64)

if __name__ == "__main__":
    DEBUG = False
    folder = "data/sn0001"
    pred = lambda p: '.csv' in p
    files = filter(pred, get_files_in_dir(folder, True))
    data = read_data_file(list(files)[0])
    print(data[0,2])
