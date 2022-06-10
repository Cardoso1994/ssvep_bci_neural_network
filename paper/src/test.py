#!/usr/bin/env python3
"""
Test for a Neural Network implementation for the BETA Database BCI.

Developer: Marco Antonio Cardoso Moreno (mcardosom2021@cic.ipn.mx
                                         marcoacardosom@gmail.com)

[1] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, “Beta: A large
        benchmark database toward ssvep-bci application,” Frontiers in
        Neuroscience, vol. 14, p. 627, 2020.
"""
from math import floor, ceil

import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy.io import loadmat
import scipy.signal as signal

np.set_printoptions(2)

WHOLE_SAMPLE = False
TOTAL_SUBJECTS = 15  # only testing right now for first 15 subjects
NUM_BLOCKS = 4
NUM_CHARS = 40  # number of symbols in screen keyboard
FS = 250  # sampling rate [Hz]
SIGNAL_LEN = 1  # lapse to be evaluated in the neural network [s]
VISUAL_LATENCY = 0.13  # according to BETA paper
VISUAL_CUE = 0.5  # time where the target is highlighted before stimulus
SAMPLE_LEN = FS * SIGNAL_LEN  # number of sample points in the final signal
MAX_EPOCHS = 100
FILTER_DEGREE = 3
CHANNELS_MAP = {'FP1': 0, 'FPZ': 1, 'FP2': 2, 'AF3': 3, 'AF4': 4, 'F7': 5,
                'F5': 6, 'F3': 7, 'F1': 8, 'FZ': 9, 'F2': 10, 'F4': 11,
                'F6': 12, 'F8': 13, 'FT7': 14, 'FC5': 15, 'FC3': 16, 'FC1': 17,
                'FCZ': 18, 'FC2': 19, 'FC4': 20, 'FC6': 21, 'FT8': 22,
                'T7': 23, 'C5': 24, 'C3': 25, 'C1': 26, 'CZ': 27, 'C2': 28,
                'C4': 29, 'C6': 30, 'T8': 31, 'M1': 32, 'TP7': 33, 'CP5': 34,
                'CP3': 35, 'CP1': 36, 'CPZ': 37, 'CP2': 38, 'CP4': 39,
                'CP6': 40, 'TP8': 41, 'M2': 42, 'P7': 43, 'P5': 44, 'P3': 45,
                'P1': 46, 'PZ': 47, 'P2': 48, 'P4': 49, 'P6': 50, 'P8': 51,
                'PO7': 52, 'PO5': 53, 'PO3': 54, 'POZ': 55, 'PO4': 56,
                'PO6': 57, 'PO8': 58, 'CB1': 59, 'O1': 60, 'OZ': 61, 'O2': 62,
                'CB2': 63}

"""
Channel selection
"""
# only occipital and parietal region
# CHANNELS = [47, 54, 53, 56, 57, 55, 60, 61, 62]
# with Broca and Wernicke areas; plus [F7, F5, F3, FT7, FC5, FC3, T7, C5, C3]
CHANNELS = [47, 54, 53, 56, 57, 55, 60, 61, 62, 5, 6, 7, 14, 15, 16, 23, 24,
            25]
# all channels
# CHANNELS = [i for i in range(64)]
NUM_CHNNLS = len(CHANNELS)

"""
Mapping characters to indexes
"""
CHARS_MAP = '.,<abcdefghijklmnopqrstuvwxyz0123456789 '
CHARS_MAP = {char: i for i, char in enumerate(CHARS_MAP)}

# probando para usuario 1
S = loadmat("../data/S1_own.mat")

"""
Creating filterbank. Set of filters to be applied to the signals
"""
low_freqs_bands = [7, 23, 39, 55, 71]
num_filters = len(low_freqs_bands)
high_freq_band = 90
filterbank = []
for low_freq in low_freqs_bands:
    filterbank.append(signal.butter(FILTER_DEGREE, (low_freq, high_freq_band),
                                    btype='bandpass', fs=FS, output='sos'))

# apply filters to each signal, per channel, per block, per character
undesired_samples = floor((VISUAL_CUE + VISUAL_LATENCY) * FS)
last_point = undesired_samples + ceil(SAMPLE_LEN)
sample_len = last_point - undesired_samples
print(f"sample len: {sample_len}")

cnn_input = np.zeros((NUM_CHNNLS, sample_len, num_filters, NUM_CHARS,
                      NUM_BLOCKS))

eeg = S['eeg'][:, undesired_samples:last_point, :, :]
for char in range(NUM_CHARS):
    for blck in range(NUM_BLOCKS):
        signal_to_filt = eeg[:, :, blck, char]
        for filt in filterbank:
            signal_filtered = np.empty((NUM_CHNNLS, sample_len))
            for channel in range(NUM_CHNNLS):
                signal_filtered[channel, :] = \
                    signal.sosfilt(filt, signal_to_filt[channel, :])

            print(signal_to_filt)
            exit()

    exit()
#     for chr = 1 : 1 : totalcharacter
#         for blk = 1 : totalblock
#             % isolating data before filters; per character and per block
#             if strcmp(dataset, 'Bench')
#                 tmp_raw = sub_data(:, :, chr, blk);
#             elseif strcmp(dataset, 'BETA')
#                 tmp_raw = sub_data(:, :, blk, chr);
#                 %else
#             end
#             % apply each filter of the filterbank
#             for i = 1 : subban_no
#                 processed_signal = zeros(total_channels, sample_length); % Initialization
#                 % apply to each channel
#                 for j = 1 : total_channels
#                     processed_signal(j, :) = filtfilt(bpFilters{i}, tmp_raw(j,:)); % Filtering raw signal with ith bandpass filter
#                 end
#                 % AllData(chn, sample points, #filter, chr, blk, subject)
#                 AllData(:, :, i, chr, blk, subject) = processed_signal;
#
#                 y_AllData(1, chr, blk, subject) = chr;
#             end
#         end
#     end
# end
# end
# extract eeg signal from mat file and remove undesired signal samples
eeg_data = S['eeg']

# take corresponding section if we dont work with the whole sample
if not WHOLE_SAMPLE:
    undesired_samples = floor((VISUAL_CUE + VISUAL_LATENCY) * FS)
    last_point = undesired_samples + ceil(SAMPLE_LEN)
    eeg_data = eeg_data[:, undesired_samples:last_point, :, :]


fft_eeg = scipy.fft.rfft(example_signal)
freqs = scipy.fft.rfftfreq(example_signal.shape[0], d=1/FS)

plt.figure(1)
plt.plot(example_signal)

plt.figure(2)
plt.plot(freqs, np.abs(fft_eeg))
plt.xlabel("Frequency [Hz]")

band_freqs = (1, 90)
sos = signal.butter(5, band_freqs, btype='bandpass',
                    fs=FS, output='sos')
other_array = signal.sosfilt(sos, example_signal)

plt.figure(1)
plt.plot(other_array)

plt.show()
