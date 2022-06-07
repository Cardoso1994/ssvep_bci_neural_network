#!/usr/bin/env python3

"""
Test for a Neural Network implementation for the BETA Database BCI.

Developer: Marco Antonio Cardoso Moreno (marcoacardosom@gmail.com)

[1] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
        benchmark database toward ssvep-bci application,” Frontiers in
        Neuroscience, vol. 14, p. 627, 2020.
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.io import loadmat

TOTAL_SUBJECTS = 15  # only testing right now for first 15 subjects
NUM_BLOCKS = 4
NUM_CHARS = 40  # number of symbols in screen keyboard
FS = 250  # sampling rate [Hz]
SIGNAL_LEN = 0.4  # [s]
VISUAL_LATENCY = 0.13  # don't know what this is
VISUAL_CUE = 0.5  # time where the target is highlighted before stimulus
SAMPLE_LEN = FS * SIGNAL_LEN
NUM_CHNNLS = 64
MAX_EPOCHS = 100

S = loadmat("../data/S1_own.mat")
eeg_data = S['eeg']
chan = S['chan']
print(eeg_data.shape)
print(chan.shape)
print(eeg_data)

fft_eeg = scipy.fft.rfft(eeg_data[0, :, 0, 0])
freqs = scipy.fft.rfftfreq(eeg_data[0, :, 0, 0].shape[0], d=1/250)

plt.figure(1)
plt.plot(eeg_data[0, :, 0, 0])

plt.figure('fourier')
plt.plot(freqs, np.abs(fft_eeg))
plt.xlabel("Frequency [Hz]")

plt.show()
