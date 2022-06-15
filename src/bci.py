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
        """Convolutional Neural Network definition."""
        super().__init__()
        self.conv1 = nn.Conv2d(5, 2, kernel_size=1, padding="same").double()
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(2, 1, kernel_size=1, padding="same").double()
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv3 = nn.Conv1d(9, 2, kernel_size=1, padding="same").double()
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv1d(2, 1, kernel_size=1, padding="same").double()
        self.act4 = nn.Tanh()
        self.pool4 = nn.MaxPool1d(kernel_size=(2))

        self.linear5 = nn.Linear(62, 40).double()

        # self.fc1 = nn.Linear(8 * 8 * 8, 32)
        # self.act3 = nn.Tanh()
        # self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        """Convolutional Neural Network forward pass."""
        # filterbank
        out = self.act1(self.conv1(x))

        # second stage of filter bank
        out = self.act2(self.conv2(out))
        out = torch.squeeze(out)
        out = self.pool2(out)

        # channel selection
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))
        out = torch.squeeze(out)
        out = self.pool4(out)

        out = self.linear5(out)
        # print(out.shape)
        # exit()

        return out


class beta_dataset(torch.utils.data.Dataset):
    """Create a dataset from the BETA database for SSVEP BCI."""

    def __init__(self, eeg_signals, labels):
        """
        Parameters.
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
        """Return dataset element at index `index`."""
        if torch.is_tensor(idx) or isinstance(idx, np.ndarray):
            idx = idx.tolist()

        return (self.eeg_signals[:, :, :, idx], self.labels[idx])


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader, device):
    """Training loop definition."""
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for signals, labels in train_loader:
            signals = signals.to(device=device)  # <1>
            labels = labels.to(device=device)
            outputs = model(signals)
            loss = loss_fn(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            # print('{} Epoch {}, Training loss {}'.format(
            #     datetime.datetime.now(), epoch,
            #     loss_train / len(train_loader)))
            print(f"Epoch: {epoch}, "
                  + f"Training loss: {loss_train / len(train_loader)}")
