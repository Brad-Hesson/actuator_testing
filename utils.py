import os

def files(path, full_path):
    iter = os.walk(path)
    path, _, fnames = next(iter)
    if full_path:
        return [os.path.normpath(os.path.join(path, fname)) for fname in fnames]
    else:
        return [os.path.normpath(fname) for fname in fnames]

if __name__ == "__main__":
    DEBUG = False
    for f in files("data/sn0001", True):
        print(f)