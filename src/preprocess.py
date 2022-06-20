#!/usr/bin/env python3
"""
Test for a Neural Network implementation for the BETA Database BCI.

Developer: Marco Antonio Cardoso Moreno (mcardosom2021@cic.ipn.mx
                                         marcoacardosom@gmail.com)

References
[1] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "Beta: A large
        benchmark database toward ssvep-bci application", Frontiers in
        Neuroscience, vol. 14, p. 627, 2020.
"""

from math import floor, ceil
import os


import matplotlib.pyplot as plt
import numpy as np
import scipy.fft
from scipy import fft
from scipy.io import loadmat
import scipy.signal as signal
from scipy import signal

np.set_printoptions(2)


OUT_PATH = os.path.join(os.path.expanduser("~"), "garbage")
WHOLE_SAMPLE = False
TOTAL_SUBJECTS = 15  # only testing right now for first 15 subjects
NUM_BLOCKS = 4
NUM_CHARS = 40  # number of symbols in screen keyboard
FS = 250  # sampling rate [Hz]
SIGNAL_LEN = 1  # lapse to be evaluated in the neural network [s]
VISUAL_LATENCY = 0.13  # according to BETA paper
VISUAL_CUE = 0.5  # time where the target is highlighted before stimulus
SAMPLE_LEN = FS * SIGNAL_LEN  # number of sample points in the final signal

FILTER_DEGREE = 8
HIGH_FREQ_BAND = 90
LOW_FREQS_BANDS = [7, 23, 39, 55, 71, 79]

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


# occs = ['PZ', 'PO3', 'PO5', 'PO4', 'PO6', 'POZ', 'O1', 'OZ', 'O2']

"""
Channel selection
"""
# only occipital and parietal region
CONF = "occ"
CHANNELS = [47, 54, 53, 56, 57, 55, 60, 61, 62]

# with Broca area; plus [F7, F5, F3, FT7, FC5, FC3, T7, C5, C3]
# CONF = "broca_occ"
# CHANNELS = [47, 54, 53, 56, 57, 55, 60, 61, 62, 5, 6, 7, 14, 15, 16, 23, 24,
#             25]

# with Wernicke area; plus [T7, C5, C3, TP7, CP5, CP3, P7, P5, P3]
# CONF = "wernicke_occ"
# CHANNELS = [47, 54, 53, 56, 57, 55, 60, 61, 62, 23, 24, 25, 33, 34, 35, 43, 44,
#             45]

# with Broca and Wernicke areas
# CONF = "broca_and_wernicke_occ"
# CHANNELS = [47, 54, 53, 56, 57, 55, 60, 61, 62, 5, 6, 7, 14, 15, 16, 23, 24,
#             25, 33, 34, 35, 43, 44, 45]

# all channels
# CONF = "all"
# CHANNELS = [i for i in range(64)]
NUM_CHNNLS = len(CHANNELS)

"""
Mapping characters to indexes
"""
CHARS_MAP = '.,<abcdefghijklmnopqrstuvwxyz0123456789 '
CHARS_MAP = {char: i for i, char in enumerate(CHARS_MAP)}

for i in range(1, 71):
    S = loadmat(f"../BETA_database/S{i}_own.mat")

    """
    Creating filterbank. Set of filters to be applied to the signals
    Every other harmonic
    """
    # low_freqs_bands = [7.5, 15.5, 23.5, 31.5, 39.5, 47.5, 55.5]
    num_filters = len(LOW_FREQS_BANDS)
    filterbank = []
    for low_freq in LOW_FREQS_BANDS:
        filterbank.append(signal.butter(FILTER_DEGREE, (low_freq,
                                                        HIGH_FREQ_BAND),
                                        btype='bandpass', fs=FS, output='sos'))

    """
    apply filters to each signal, per channel, per block, per character
    """
    # undesired_samples = floor((VISUAL_CUE + VISUAL_LATENCY) * FS)
    # undesired_samples = 0
    undesired_samples = floor((VISUAL_CUE) * FS)

    last_point = undesired_samples + ceil(SAMPLE_LEN)
    sample_len = last_point - undesired_samples

    # chan_1 = 5
    # block_1 = 0
    # char_1 = CHARS_MAP[' ']

    # chan_2 = 10
    # block_2 = 0
    # char_2 = CHARS_MAP[' ']

    eeg = S['eeg'][CHANNELS, undesired_samples:last_point, :, :]

    # fig, ax = plt.subplots(2)
    # X = scipy.fft.rfft(eeg[chan_1, :, block_1, char_1])
    # freqs = scipy.fft.rfftfreq(eeg[chan_1, :, block_1, char_1].shape[0],
    #                            d=1/FS)
    # ax[0].plot(freqs, np.abs(X))
    # X = scipy.fft.rfft(eeg[chan_2, :, block_2, char_2])
    # freqs = scipy.fft.rfftfreq(eeg[chan_2, :, block_2, char_2].shape[0],
    #                            d=1/FS)
    # ax[1].plot(freqs, np.abs(X))

    # cnn_input = np.zeros((NUM_CHNNLS, sample_len, num_filters, NUM_CHARS,
    #                     NUM_BLOCKS))
    cnn_input = np.zeros((num_filters, NUM_CHNNLS, sample_len, NUM_CHARS,
                          NUM_BLOCKS))

    cnn_output = np.zeros((NUM_CHARS, NUM_BLOCKS))
    for char in range(NUM_CHARS):
        for blck in range(NUM_BLOCKS):
            signal_to_filt = eeg[:, :, blck, char]
            for filt_num, filt in enumerate(filterbank):
                signal_filtered = np.zeros((NUM_CHNNLS, sample_len))
                for channel in range(NUM_CHNNLS):
                    signal_filtered[channel, :] = \
                        signal.sosfilt(filt, signal_to_filt[channel, :])
                # cnn_input[:, :, filt_num, char, blck] = signal_filtered
                cnn_input[filt_num, :, :, char, blck] = signal_filtered[:, :]
                cnn_output[char, blck] = char

    """
    Saving cnn_input and cnn_output to npy binary files
    """
    np.save(os.path.join(OUT_PATH, f"S{i}_input_{CONF}.npy"), cnn_input)
    np.save(os.path.join(OUT_PATH, f"S{i}_output_{CONF}.npy"), cnn_output)

    """
    plotting the different filters applied to the signals
    ideally each filter tries to leave behind some harmonics of the signal
    """
    # fig, ax = plt.subplots(6, 2)
    # ax[0, 0].set_title('Occipital')
    # ax[0, 0].plot(eeg[chan_1, :, block_1, char_1])
    # ax[1, 0].plot(cnn_input[chan_1, :, 0, char_1, block_1])
    # ax[2, 0].plot(cnn_input[chan_1, :, 1, char_1, block_1])
    # ax[3, 0].plot(cnn_input[chan_1, :, 2, char_1, block_1])
    # ax[4, 0].plot(cnn_input[chan_1, :, 3, char_1, block_1])
    # ax[5, 0].plot(cnn_input[chan_1, :, 4, char_1, block_1])

    # ax[0, 1].set_title('Broca Area')
    # ax[0, 1].plot(eeg[chan_2, :, block_2, char_2])
    # ax[1, 1].plot(cnn_input[chan_2, :, 0, char_2, block_2])
    # ax[2, 1].plot(cnn_input[chan_2, :, 1, char_2, block_2])
    # ax[3, 1].plot(cnn_input[chan_2, :, 2, char_2, block_2])
    # ax[4, 1].plot(cnn_input[chan_2, :, 3, char_2, block_2])
    # ax[5, 1].plot(cnn_input[chan_2, :, 4, char_2, block_2])

    """
    Plotting average of occipital vs broca areas
    """
    # fig, ax = plt.subplots(6, 2)
    # _x1_ = np.mean(S['eeg'][(47, 54, 53, 56, 57, 55, 60, 61, 62),
    #                         undesired_samples:last_point, 0, char_1], axis=0)
    # _x2_ = np.mean(S['eeg'][(5, 6, 7, 14, 15, 16, 23, 24, 25),
    #                         undesired_samples:last_point, 0, char_1], axis=0)
    # ax[0, 0].set_title('occipital summing')
    # ax[0, 0].plot(_x1_)
    # ax[0, 1].set_title('broca summing')
    # ax[0, 1].plot(_x2_)
    # for filt_num, filt in enumerate(filterbank):
    #     s1 = signal.sosfilt(filt, _x1_)
    #     ax[filt_num + 1, 0].plot(s1)
    #     s2 = signal.sosfilt(filt, _x2_)
    #     ax[filt_num + 1, 1].plot(s2)
    # plt.show()

