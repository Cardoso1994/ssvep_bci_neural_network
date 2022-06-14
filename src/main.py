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

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

import bci

np.set_printoptions(2)

"""
GLOBAL VARIABLES
"""
TOTAL_SUBJECTS = 70  # only testing right now for first 15 subjects
NUM_BLOCKS = 4
NUM_CHARS = 40  # number of symbols in screen keyboard
MAX_EPOCHS = 100

SUBJECTS_FOR_TRAIN = 4
SUBJECTS_FOR_VAL = 1

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device where pytorch will execute: {DEVICE}")

"""
Main code
"""

ELECTRODE_CONFIGURATION = "occ"
file_suffix = f"_{ELECTRODE_CONFIGURATION}.npy"
data_location = os.path.join("..", "BETA_database",
                             f"BETA_{ELECTRODE_CONFIGURATION}")

"""
PREPARING TRAINING DATASET
"""
train_inputs = []
train_outputs = []
for i in range(1, SUBJECTS_FOR_TRAIN + 1):
    train_inputs.append(np.load(os.path.join(data_location,
                                             f"S{i}_input{file_suffix}")))
    train_outputs.append(np.load(os.path.join(data_location,
                                              f"S{i}_output{file_suffix}")))

input_shape = train_inputs[0].shape
# num_channels = train_inputs[0].shape[0]
# first three dimensions are the same, then we multiply by 40 chars and 4
# blocks, as well as by the number of subjects
# (num_filters, num_eeg_channels, sample_points, chars, blocks)
# (5, 9, 250, 40, 4)
_ds_ = np.zeros((input_shape[0], input_shape[1], input_shape[2],
                 NUM_CHARS * NUM_BLOCKS * SUBJECTS_FOR_TRAIN))
_labels_ = np.zeros(NUM_CHARS * NUM_BLOCKS * SUBJECTS_FOR_TRAIN)
_ds_[:, :, :, :] = np.inf

for subject in range(SUBJECTS_FOR_TRAIN):  # number of subjects
    for blck in range(NUM_BLOCKS):  # number of blocks
        for char in range(NUM_CHARS):  # number of characters
            _ds_[:, :, :, subject * NUM_BLOCKS * NUM_CHARS
                 + blck * NUM_CHARS + char] = \
                train_inputs[subject][:, :, :, char, blck]
            _labels_[subject * NUM_BLOCKS * NUM_CHARS
                     + blck * NUM_CHARS + char] = \
                train_outputs[subject][char, blck]

train_ds = bci.beta_dataset(_ds_, _labels_)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

"""
PREPARING VALIDATION DATASET
"""
val_inputs = []
val_outputs = []
for i in range(SUBJECTS_FOR_TRAIN + 1,
               SUBJECTS_FOR_TRAIN + SUBJECTS_FOR_VAL + 1):
    val_inputs.append(np.load(os.path.join(data_location,
                                           f"S{i}_input{file_suffix}")))
    val_outputs.append(np.load(os.path.join(data_location,
                                            f"S{i}_output{file_suffix}")))

# first three dimensions are the same, then we multiply by 40 chars and 4
# blocks, as well as by the number of subjects
# (num_filters, num_eeg_channels, sample_points, chars, blocks)
# (5, 9, 250, 40, 4)
# _ds_ = np.zeros((input_shape[2], input_shape[0], input_shape[1],
#                  NUM_CHARS * NUM_BLOCKS * len(val_inputs)))
_ds_ = np.zeros((input_shape[0], input_shape[1], input_shape[2],
                 NUM_CHARS * NUM_BLOCKS * SUBJECTS_FOR_VAL))
_labels_ = np.zeros(NUM_CHARS * NUM_BLOCKS * len(val_inputs))
_ds_[:, :, :, :] = np.inf

for subject in range(SUBJECTS_FOR_VAL):  # number of subjects
    for blck in range(NUM_BLOCKS):  # number of blocks
        for char in range(NUM_CHARS):  # number of characters
            _ds_[:, :, :, subject * NUM_BLOCKS * NUM_CHARS
                 + blck * NUM_CHARS + char] = \
                val_inputs[subject][:, :, :, char, blck]
            _labels_[subject * NUM_BLOCKS * NUM_CHARS
                     + blck * NUM_CHARS + char] = \
                val_outputs[subject][char, blck]

val_ds = bci.beta_dataset(_ds_, _labels_)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=True)

bci_net = bci.bci_cnn().to(device=DEVICE)

n_epochs = 10
optimizer = torch.optim.Adam(bci_net.parameters(), lr=0.002)
loss_fn = torch.nn.CrossEntropyLoss()

bci.training_loop(n_epochs, optimizer, bci_net, loss_fn, train_dl, DEVICE)
