#!/usr/bin/env python3
"""
Main program after preprocessing of signals.

This program executes after preprocessing.
It expects to open '*.npy' binary files for a given subject from the BETA
database. It reads, per subject, an EEG file and a labels file.

The code must check for number of channels, since this is variable depending on
the configuration being tested.


Developer: Marco Antonio Cardoso Moreno (mcardosom2021@cic.ipn.mx
                                         marcoacardosom@gmail.com)


[1] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "Beta: A large
        benchmark database toward ssvep-bci application", Frontiers in
        Neuroscience, vol. 14, p. 627, 2020.
"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

np.set_printoptions(2)

"""
GLOBAL VARIABLES
"""
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
CHARS_MAP = '.,<abcdefghijklmnopqrstuvwxyz0123456789 '
CHARS_MAP = {char: i for i, char in enumerate(CHARS_MAP)}

SUBJECTS_FOR_TRAIN = 4

"""
Main code
"""

# BETA has 70 subjects. Select 4 randomly for training. Test in the rest of
# subjects
subjects = [i for i in range(70)]
random.shuffle(subjects)
train_subjects = subjects[:SUBJECTS_FOR_TRAIN]
test_subjects = subjects[SUBJECTS_FOR_TRAIN:]

print("In main: TODO")
