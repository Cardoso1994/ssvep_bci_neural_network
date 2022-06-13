#!/usr/bin/env python3
"""
Definition of convolutional neural network.

Developer: Marco Antonio Cardoso Moreno (mcardosom2021@cic.ipn.mx
                                         marcoacardosom@gmail.com)

References
[1] B. Liu, X. Huang, Y. Wang, X. Chen, and X. Gao, "Beta: A large
        benchmark database toward ssvep-bci application", Frontiers in
        Neuroscience, vol. 14, p. 627, 2020.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class bci_cnn(nn.Module):
    """
    Definition of the convolutional neural network used for SSVEP-BCI systems.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(8 * 8 * 8, 32)
        self.act3 = nn.Tanh()
        self.fc2 = nn.Linear(32, 2)


    def forward(self, x):
        out = self.pool1(self.act1(self.conv1(x)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 8 * 8 * 8)
        out = self.act3(self.fc1(out))
        out = self.fc2(out)
        return out


class beta_dataset(torch.utils.data.Dataset):
    def __init__(self, eeg_signals, labels):
        """
        Create a dataset from the BETA database for SSVEP BCI

        Parameters
        ----------
        signals : np.array
            array containing all signal information. With shape:
                - channels: the number of EEG channels
                - samples: a sample in each timestep
                - num of filters: number of bandpass filters applied
                - number of tests in bci. Ideally, there is a pattern per
                  character, per block, per subject. At least for training
        labels : np.array
            array containing all labels for each pattern in 'signals'. With
            shape:
            - number of tests in bci (same as 4th dimension in signals)
        """
        self.eeg_signals = eeg_signals
        self.labels = labels


    def __len__(self):
        """Return number of patterns in dataset."""
        return self.eeg_signals.shape[3]


    def __getitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()

        return (self.eeg_signals[:, :, :, idx], self.labels[idx])
