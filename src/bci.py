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
# import torch.nn.functional as F
# import torch.optim as optim

torch.set_printoptions(precision=3)

class bci_cnn(nn.Module):
    """
    Definition of the convolutional neural network used for SSVEP-BCI systems.

    Loss function at page 187!!!
    """

    def __init__(self):
        """Convolutional Neural Network definition."""
        super().__init__()

        # filter bank
        self.conv1 = nn.Conv2d(5, 6, kernel_size=1, padding="same").double()
        self.norm1 = nn.BatchNorm2d(6).double()
        self.act1 = nn.Tanh()

        self.conv2 = nn.Conv2d(6, 1, kernel_size=(1, 3),
                               padding="same").double()
        self.norm2 = nn.BatchNorm2d(1).double()
        self.act2 = nn.Tanh()

        # channel selection
        self.conv3 = nn.Conv2d(1, 10, kernel_size=(5, 1), padding="valid",
                               stride=(4, 1)).double()
        self.norm3 = nn.BatchNorm2d(10).double()
        self.act3 = nn.Tanh()

        self.conv4 = nn.Conv2d(10, 1, kernel_size=(2, 3), padding=(0, 1),
                               stride=(1, 1)).double()
        self.norm4 = nn.BatchNorm2d(1).double()
        self.act4 = nn.Tanh()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))

        # self.conv5 = nn.Conv2d(1, 10, kernel_size=(1, 3),
        #                        padding="same").double()
        # self.norm5 = nn.BatchNorm2d(10).double()
        # self.act5 = nn.Tanh()
        # self.pool5 = nn.MaxPool2d(kernel_size=(1, 2))

        self.linear6 = nn.Linear(1 * 1 * 125, 40).double()

    def forward(self, x):
        """Convolutional Neural Network forward pass."""
        # filterbank
        # print(f"shape of input: {x.shape}")
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        # print(f"shape of output after layer 1: {out.shape}")

        # second stage of filter bank
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        # out = self.pool2(out)
        # out = self.drop2(out)
        # print(f"shape of output after layer 2: {out.shape}")

        # channel selection
        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act3(out)
        # print(f"shape of out after layer 3: {out.shape}")

        out = self.conv4(out)
        out = self.norm4(out)
        out = self.act4(out)
        out = self.pool4(out)
        # print(f"shape of out after layer 4: {out.shape}")

        # out = self.conv5(out)
        # out = self.norm5(out)
        # out = self.act5(out)
        # out = self.pool5(out)
        # exit()

        out = self.linear6(out)
        out = torch.squeeze(out)
        # print(out.shape)
        # print()
        # print()
        # print()
        # print()
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


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader,
                  val_loader, device):
    """Training loop definition."""
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        loss_val = 0.0

        correct_train, total_train = 0, 0
        correct_val, total_val = 0, 0
        # training
        for signals, labels in train_loader:
            signals = signals.to(device=device)
            labels = labels.to(device=device)
            outputs = model(signals)
            # print(outputs.shape)
            # exit()
            _, predicted = torch.max(outputs, dim=1)
            total_train += labels.shape[0]
            correct_train += int((predicted == labels).sum())
            loss = loss_fn(outputs, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        # validation
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device=device)
                labels = labels.to(device=device)
                outputs = model(signals)

                _, predicted = torch.max(outputs, dim=1)
                total_val += labels.shape[0]
                correct_val += int((predicted == labels).sum())

                loss = loss_fn(outputs, labels.long())

                loss_val += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}: "
                  + f"Training: loss: {loss_train / len(train_loader):.3f}"
                  + f", Acc: {correct_train / total_train:.3f}"
                  + f" || Validation loss: {loss_val / len(val_loader):.3f}"
                  + f", Acc: {correct_val / total_val:.3f}")
