import os
import numpy as np
import re
import shelve
import inspect
from datetime import datetime, timedelta


def cache_result(ttl=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            f_name = func.__name__

            # if we are inside a non-ttl cached function and we ourselves
            # are a ttl cache function, there is potential for invalid data.
            assert not (
                cache_result.nested is not None and ttl is not None
            ), "ttl cached function `%s` inside of non-ttl cached function `%s`" % (
                f_name,
                cache_result.nested,
            )
            args_key = repr(args) + repr(kwargs)
            sh_flag = "c"
            try:
                with shelve.open("./__cache_store__/" + f_name, "c") as sh:
                    if sh["function_hash"] == inspect.getsource(func):
                        if ttl is None or sh[args_key + "ttl"] > datetime.now():
                            # if the key exists in the shelf, the function has not been modified
                            # since the last cache, and if a ttl was provided and has not yet
                            # passed, then just return the cached value.
                            return sh[args_key]
                    else:
                        # if the function has been modified since the last cache, mark the
                        # open flag to clear the shelf completely because any/all of the
                        # cached values may now be invalid.
                        sh_flag = "n"
            except FileNotFoundError:
                # if the cache directory does not exist, create it.
                os.mkdir("./__cache_store__")
            except KeyError:
                # if the function cache does not exist, the args key does not exist in
                # the cache file, or a ttl was provided but none exists in the cache, then
                # continue to run the function and store the result.
                pass
            if cache_result.nested is not None:
                # if we are currently running in a higher level cached function, simply
                # run the function and return.  This is because the higher level cache
                # will store the value, and we don't want to store intermediate results.
                return func(*args, **kwargs)
            else:
                # if we are not in a higher level cache function and we do not have a
                # tll, set the nested flag to notify lower level functions that they are nested.
                cache_result.nested = f_name if ttl is None else cache_result.nested
                out = func(*args, **kwargs)
                cache_result.nested = None if ttl is None else cache_result.nested
                with shelve.open("./__cache_store__/" + f_name, sh_flag) as sh:
                    sh["function_hash"] = inspect.getsource(func)
                    sh[args_key] = out
                    if ttl is not None:
                        sh[args_key + "ttl"] = datetime.now() + timedelta(seconds=ttl)
                return out

        return wrapper

    return decorator


cache_result.nested = None


def get_files_in_dir(folder, full_path=True):
    iter = os.walk(folder)
    path, _, fnames = next(iter)
    if full_path:
        return [os.path.normpath(os.path.join(path, fname)) for fname in fnames]
    else:
        return [os.path.normpath(fname) for fname in fnames]


def get_data_files_in_dir(folder, full_path=True):
    files = get_files_in_dir(folder, full_path=False)
    if full_path:
        return [os.path.join(folder, f) for f in files if "acq" in f and ".csv" in f]
    else:
        return [f for f in files if "acq" in f and ".csv" in f]


@cache_result()
def read_data_file(path):
    pat = re.compile(r"^((-?\d+(\.\d+)?(e-?\d+)?),?)+$")
    data_start = 0
    with open(path) as f:
        for i, line in enumerate(f):
            if pat.match(line) is not None:
                data_start = i
                break
    return np.loadtxt(path, delimiter=",", skiprows=data_start, dtype=np.float64)


def get_index_zero_crossings(y):
    return np.nonzero(np.diff(np.sign(y)))[0]


def get_interp_zero_crossings(x, y):
    zcis = get_index_zero_crossings(y)
    zcs = []
    for zci in zcis:
        m = (y[zci + 1] - y[zci]) / (x[zci + 1] - x[zci])
        zcs += [(m * x[zci] - y[zci]) / m]
    return zcs


def mututal_interp(ds):
    for d in ds:
        shape = np.shape(d)
        assert len(shape) == 2
        assert shape[1] == 2
    xs = np.sort(np.concatenate([d[:, 0] for d in ds]))
    ys_s = [np.interp(xs, d[:, 0], d[:, 1]) for d in ds]
    return [np.transpose(np.vstack((xs, ys))) for ys in ys_s]


if __name__ == "__main__":
    folder = "data/sn0001/07-12-2021"
    _ = [print(f) for f in get_data_files_in_dir(folder, False)]
