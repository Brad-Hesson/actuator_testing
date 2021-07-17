import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.io import wavfile


dt = 0.01  # seconds
ramp_rates = np.linspace(5, 0.1, 5)  # volts/second  !!number must be odd!!
displacements = np.linspace(95, 95, 1) / 20  # volts  !!number must be odd!!
hold_time = 10.5 * 60  # seconds
repeat = 25

displacements = np.append(
    np.copy(displacements[::2]), np.flip(np.copy(displacements[1::2]))
)

displacements = np.array([m for x in displacements for m in [x]*repeat])

print(displacements)
data = np.array([displacements[0] / 2] * int(hold_time / dt))
direction = -1
num = len(ramp_rates) * len(displacements)
i = 1
for (ramp, disp) in itertools.product(ramp_rates, displacements):
    print(
        "Progress: %5.2f%%  Ramp: %5.2f v/s  Disp: %5.2f v"
        % (i / num * 100, ramp, disp)
    )
    i += 1
    start_v = data[-1]
    data = np.append(
        data, np.linspace(start_v, start_v + direction * disp, int(disp / ramp / dt))
    )
    data = np.append(data, np.linspace(data[-1], data[-1], int(hold_time / dt)))
    direction *= -1

path = "waveforms/waveform"
metadata = np.array(list(itertools.product(ramp_rates * 20, displacements * 20)))
np.savetxt(path + ".csv", metadata)
wavfile.write(path + ".wav", int(1 / dt), data)
print("Wav File Written")

import display_waveform

display_waveform.display_wav_file(path + ".wav")
