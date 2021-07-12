import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from os import walk

def get_last_file(folder):
    f = []
    for (dirpath, dirnames, filenames) in walk(folder):
        for filename in filenames:
            if '.csv' in filename:
                f += [filename]
        break
    fname = f[-1]
    path = folder + '/' + fname
    return path

def rsquared(ys,fs):
    mean = np.mean(ys)
    SStot = np.sum(np.square(ys-mean))
    SSres = np.sum(np.square(ys-fs))
    return 1-(SSres/SStot)

def compute_rsquared_data(F, p0, xs, ys):
    N = 200
    rs = []
    fails = 0
    total = 0
    [p0, _] = curve_fit(F, xs, ys, p0=p0, maxfev=1000)
    for i in range(1,N+1):
        total += 1
        string = ""
        string += "Progress: %6.2f%%   " % (i/N*100)
        string += "Failed: %6.2f%%   "%(fails/total*100)
        string += "P0: %s" % (str(p0))
        print(string)
        end = int(len(xs)*i/N)
        try:
            [cvec, _] = curve_fit(F, xs[:end], ys[:end], p0=p0, maxfev=2000)
        except:
            fails += 1
            cvec = p0*0
        rs += [rsquared(ys, F(xs,*cvec))]
    return rs

def get_data_start(dat):
    ds = dat[:,2]
    return np.argmax(np.abs(ds - ds[-1]) < 0.1)

if __name__ == '__main__':
    # constants
    SAVE = False
    folder = "4plate_v1/P10min_A200m_S1p"

    # get the last file from the folder
    path = get_last_file(folder)
    print(path)

    # load data from file
    dat = np.loadtxt(path, delimiter=',', dtype=np.float64, skiprows=21)

    # crop the data to just the creep section
    start = get_data_start(dat)
    xs = dat[start:,0]
    ys = dat[start:,1]

    # plot the starting index location and cropped data
    plt.figure(1, figsize=(20,5))
    plt.subplot(1,2,1)
    plt.plot(dat[start-10:start+10,0],dat[start-10:start+10,2])
    plt.axvline(dat[start,0])
    plt.subplot(1,2,2)
    plt.plot(xs, ys)
    # show the plot
    plt.show()

    # setup list to hold different rsquared datasets
    rs_s = []

    # define the fit funtion and initial constants
    def F(x, a, w, b, c): return a*np.power(x+w,b)+c
    p0 = [1, 30, -1, 0]

    # get the predictability
    rs = compute_rsquared_data(F, p0, xs, ys)
    rs_s += [rs]

    # plot the predictability
    plt.figure(1, figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title("Dataset: %s" % folder, fontsize=20, loc='center', pad=20)
    plt.plot(np.linspace(0,100,len(rs)), rs)
    plt.ylim(-0.1,1.1)
    print("R^2: %.5f"%(rs[-1]))

    # get the vector of constants
    [cvec, _] = curve_fit(F, xs, ys, p0=p0, maxfev=1000)

    # plot the fit
    plt.subplot(1,2,2)
    plt.title("Fit Function: a*np.power(x+w,b)+c", fontsize=20, loc='center', pad=20)
    plt.plot(xs, ys)
    plt.plot(xs, F(xs,*cvec))
    # show the plot
    plt.show()

    # save the plot if the save flag is set
    plt.tight_layout()
    if SAVE: plt.savefig("Power_Fit/%s.jpg" % folder, dpi=300)



    # define the fit funtion and initial constants
    def F(x, a, w, c): return a*np.log(x+w)+c
    p0=[1, 100, 0]

    # get the predictability
    rs = compute_rsquared_data(F, p0, xs, ys)
    rs_s += [rs]

    # plot the predictability
    plt.figure(1, figsize=(20,5))
    plt.subplot(1,2,1)
    plt.title("Dataset: %s" % folder, fontsize=20, loc='center', pad=20)
    plt.plot(np.linspace(0,100,len(rs)), rs)
    plt.ylim(-0.1,1.1)
    print("R^2: %.5f"%(rs[-1]))

    # get the vector of constants
    [cvec, _] = curve_fit(F, xs, ys, p0=p0, maxfev=1000)

    # plot the fit
    plt.subplot(1,2,2)
    plt.title("Fit Function: a*np.log(x+w)+c", fontsize=20, loc='center', pad=20)
    plt.plot(xs, ys)
    plt.plot(xs, F(xs,*cvec))
    # show the plot
    plt.show()

    # save the plot if the save flag is set
    plt.tight_layout()
    if SAVE: plt.savefig("Log_Fit/%s.jpg" % folder, dpi=300)


    # plot the predictability comparison
    plt.figure(1)
    plt.ylim(-0.1, 1.1)
    plt.title("R^2 Predictability\n%s" % folder, fontsize=15, loc='center', pad=10)
    for rs in rs_s:
        plt.plot(np.linspace(0,100,len(rs)), rs)
    plt.legend(("Power Law","Logarithmic"))
    # show the plot
    plt.show()

    # save the plot if the save flag is set
    plt.tight_layout()
    if SAVE: plt.savefig("Predictability_Compare/%s.jpg" % folder, dpi=300)