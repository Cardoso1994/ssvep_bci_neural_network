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

    Loss function at page 187!!!
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(5, 2, kernel_size=1, padding=1)
        self.act1 = nn.Tanh()
        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, padding=1)
        self.act2 = nn.Tanh()
        # self.conv2 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(2)
        # self.fc1 = nn.Linear(8 * 8 * 8, 32)
        # self.act3 = nn.Tanh()
        # self.fc2 = nn.Linear(32, 2)


    def forward(self, x):
        print("\n\n\nBEFORE FORWARD PASS\n\n\n")
        print(self.conv1(x))
        out = self.act1(self.conv1(x))
        print(out)
        out = self.act2(self.conv2(x))
        # out = self.pool1(self.act1(self.conv1(x)))
        # out = self.pool2(self.act2(self.conv2(out)))
        # out = out.view(-1, 8 * 8 * 8)
        # out = self.act3(self.fc1(out))
        # out = self.fc2(out)
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
        if isinstance(eeg_signals, np.ndarray):
            self.eeg_signals = torch.from_numpy(eeg_signals)
        elif isinstance(eeg_signals, list):
            self.eeg_signals = torch.tensor(eeg_signals)
        else:
            self.eeg_signals = torch.tensor(eeg_signals)

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels)
        elif isinstance(labels, list):
            self.labels = torch.tensor(labels)
        else:
            self.labels = torch.tensor(labels)


    def __len__(self):
        """Return number of patterns in dataset."""
        return self.eeg_signals.shape[3]


    def __getitem__(self, idx):
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()

        return (self.eeg_signals[:, :, :, idx], self.labels[idx])


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device=device)  # <1>
            labels = labels.to(device=device)
            outputs = model(signals)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {}, Training loss {}'.format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)))
