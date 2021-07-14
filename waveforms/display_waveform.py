import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def display_wav_file(path):
    rate, data = wavfile.read(path)

    dt = 1 / rate

    xs = np.linspace(0, len(data) * dt / 60 / 60, len(data))
    ys = data * 20
    plt.scatter(xs, ys, marker=".", s=0.8)
    plt.axhline(50, linewidth=1, color="black")
    plt.axhline(-50, linewidth=1, color="black")
    plt.show()


if __name__ == "__main__":
    display_wav_file("waveforms/waveform.wav")
