import numpy as np
import matplotlib.pyplot as plt
from scipy import *
from matplotlib import ticker
from scipy import signal
from IPython.display import Audio
from math import pi
import sounddevice as sd
import random
from contextlib import closing
from thinkdsp import *


def plot_play(sound_data, time_data, sample_rate, ampl=1):
    print(f"{len(sound_data)} data points(measurements)")
    plt.plot(time_data, sound_data)
    plt.grid()
    plt.xlabel(r"Time(seconds)")
    plt.ylabel(f'Volume(Amplitude) = {ampl}')
    plt.show()
    sd.play(sound_data, sample_rate)


def make_wave(freq= 440, sample_rate=8000, t=0.1, ampl=1):
    T = 1/sample_rate # Sample lenght/period = time duration of 1 sample;
                        # we need this to generate the "duration timeline"
    # N = sample_rate * t # data points per lenght;
    omega = 2*pi*freq
    timeline = np.arange(0, t, T)
    freq_datapoints = np.sin(omega*timeline)*ampl
    return [freq_datapoints, timeline]


def fades(wave_data, sample_rate, denom=20, duration=0.1):
    n = len(wave_data[0])
    p1 = n // denom
    p2 = int(duration * sample_rate)
    p = min(p1, p2)

    f1 = np.linspace(0, 1, p)
    f2 = np.ones(n - 2*p)
    f3 = np.linspace(1, 0, p)

    window = np.concatenate((f1, f2, f3))
    faded_data = [wave_data[0]*window, wave_data[1]]
    return faded_data


def generate_function(t):
    x = 1 * np.sin(2 * pi * (2 * t - 0))
    x += 0.5 * np.sin(2 * pi * (6 * t - 0.1))
    x += 0.1 * np.sin(2 * pi * (20 * t - 0.2))
    return x


def sampling_reconstruction(Fs, Fs_fine):
    duration = 3
    num_fine = int(Fs_fine * duration)
    t_fine = np.arange(num_fine) / Fs_fine
    f = generate_function(t_fine)

    num_samples = int(Fs * duration)
    t_sample = np.arange(num_samples) / Fs
    x = generate_function(t_sample)

    f_sinc = np.zeros(len(t_fine))
    for n in range(0, len(t_sample)):
        f_sinc += x[n] * np.sinc(Fs * t_fine - n)

    return f, t_fine, x, t_sample, f_sinc

