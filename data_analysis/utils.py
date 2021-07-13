import os
import numpy as np
import re
import shelve
import inspect


def cache_result(func):
    def wrapper(*args, **kwargs):
        args_key = repr(args) + repr(kwargs)
        f_name = func.__name__
        s_flag = 'c'
        try:
            with shelve.open("./__cache_store__/" + f_name, "c") as sh:
                if sh["function_hash"] == inspect.getsource(func):
                    return sh[args_key]
                else:
                    s_flag = 'n'
        except FileNotFoundError:
            os.mkdir("./__cache_store__")
        except KeyError:
            pass
        out = func(*args, **kwargs)
        with shelve.open("./__cache_store__/" + f_name, s_flag) as sh:
            sh["function_hash"] = inspect.getsource(func)
            sh[args_key] = out
        return out

    return wrapper


def get_files_in_dir(path, full_path=False):
    iter = os.walk(path)
    path, _, fnames = next(iter)
    if full_path:
        return [os.path.normpath(os.path.join(path, fname)) for fname in fnames]
    else:
        return [os.path.normpath(fname) for fname in fnames]


@cache_result
def read_data_file(path):
    pat = re.compile("^((-?\d+(\.\d+)?(e-?\d+)?),?)+$")
    data_start = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if pat.match(line) is not None:
                data_start = i
                break
    return np.loadtxt(path, delimiter=",", skiprows=data_start, dtype=np.float64)


def get_index_zero_crossing(y):
    return np.nonzero(np.diff(np.sign(y)))[0][0]


def get_interp_zero_crossing(x, y):
    zci = get_index_zero_crossing(y)
    m = (y[zci + 1] - y[zci]) / (x[zci + 1] - x[zci])
    return (m * x[zci] - y[zci]) / m


def mututal_interp(ds):
    for d in ds:
        shape = np.shape(d)
        assert len(shape) == 2
        assert shape[1] == 2
    xs = np.sort(np.concatenate([d[:, 0] for d in ds]))
    ys_s = [np.interp(xs, d[:, 0], d[:, 1]) for d in ds]
    return [np.transpose(np.vstack((xs, ys))) for ys in ys_s]




if __name__ == "__main__":
    @cache_result
    def fun(i):
        print("actually ran")
        return i
    
    print(fun(1))
    print(fun(2))