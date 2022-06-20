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

np.set_printoptions(3)
torch.set_printoptions(precision=3)
# torch.manual_seed(7)

"""
GLOBAL VARIABLES
"""
TOTAL_SUBJECTS = 70  # only testing right now for first 15 subjects
NUM_BLOCKS = 4
NUM_CHARS = 40  # number of symbols in screen keyboard
NUM_EPOCHS = 100
NUM_EPOCHS = 150

SUBJECTS_FOR_TRAIN = 4
SUBJECTS_FOR_VAL = 1

BATCH_SIZE = 32
BATCH_SIZE = 16

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device where pytorch will execute: {DEVICE}")

"""
Main code
"""
ELECTRODE_CONFIGURATION = "occ"
file_suffix = f"_{ELECTRODE_CONFIGURATION}.npy"
data_location = os.path.join("..", "BETA_database",
                             f"BETA_{ELECTRODE_CONFIGURATION}")

"""
TRAINING DATASET
"""
# input shape: (5, 9, 250, 40, 4)
# First dimension in dataset, is second in npy file
# Second dimension in dataset, is first in npy file
# npy file: (num_filters, num_eeg_channels, sample_points, chars, blocks)
input_shape = (5, 9, 250, 40, 4)

# dataset:
#        (num_eeg_channels, num_filters, sample_points, chars *  blocks * users)
# (9, 5, 250, 40 *  4 * SUBJECTS_FOR_VAL)
# axis reordering at the end of reading and storing npy file
train_inputs = []
train_outputs = []
for subject in range(1, SUBJECTS_FOR_TRAIN + 1):
    train_inputs.append(os.path.join(data_location,
                                             f"S{subject}_input{file_suffix}"))
    train_outputs.append(os.path.join(data_location,
                                             f"S{subject}_output{file_suffix}"))

train_ds = bci.beta_dataset(input_shape, train_inputs, train_outputs,
                            NUM_BLOCKS, NUM_CHARS, eeg_first=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                                       shuffle=True)


"""
VALIDATION DATASET
"""
val_inputs = []
val_outputs = []
for subject in range(SUBJECTS_FOR_TRAIN + 1,
               SUBJECTS_FOR_TRAIN + SUBJECTS_FOR_VAL + 1):
    val_inputs.append(os.path.join(data_location,
                                   f"S{subject}_input{file_suffix}"))
    val_outputs.append(os.path.join(data_location,
                                            f"S{subject}_output{file_suffix}"))

val_ds = bci.beta_dataset(input_shape, val_inputs, val_outputs,
                            NUM_BLOCKS, NUM_CHARS, eeg_first=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE,
                                       shuffle=True)


"""
NEURAL NETWORK
"""
bci_net = bci.bci_cnn_eeg_first().to(device=DEVICE)

optimizer = torch.optim.Adam(bci_net.parameters(), lr=0.0005)
loss_fn = torch.nn.CrossEntropyLoss()

bci.training_loop(NUM_EPOCHS, optimizer, bci_net, loss_fn, train_dl,
                  val_dl, DEVICE)
