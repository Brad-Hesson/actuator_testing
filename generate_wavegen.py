from numpy import linspace, array, append, flip, copy, savetxt
from itertools import product
import matplotlib.pyplot as plt
from scipy.io import wavfile


dt = 0.01  # seconds
ramp_rates = linspace(10, 0.1, 21)  # volts/second  !!number must be odd!!
displacements = linspace(5, 90, 21) / 20  # volts  !!number must be odd!!
hold_time = 5 * 60  # seconds


displacements = append(copy(displacements[::2]), flip(copy(displacements[1::2])))
print(displacements)
data = array([displacements[0] / 2] * int(hold_time / dt))
direction = -1
num = len(ramp_rates) * len(displacements)
i = 1
for (ramp, disp) in product(ramp_rates, displacements):
    print("Progress: %5.2f%%" % (i / num * 100))
    i += 1
    # print("Ramp: %.2f v/s  Disp: %.2f v" % (ramp, disp))
    start_v = data[-1]
    data = append(
        data, linspace(start_v, start_v + direction * disp, int(disp / ramp / dt))
    )
    data = append(data, linspace(data[-1], data[-1], int(hold_time / dt)))
    direction *= -1

wavfile.write("wavegen.wav", int(1 / dt), data)
print("Wav File Written")

xs = linspace(0, len(data)*dt/60/60, len(data))
ys = data*20
plt.scatter(xs,ys, marker='.', s=0.8)
plt.axhline(50, linewidth=1, color="black")
plt.axhline(-50, linewidth=1, color="black")
plt.show()
