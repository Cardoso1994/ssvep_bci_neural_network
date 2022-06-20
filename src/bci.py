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
        self.conv1 = nn.Conv2d(5, 10, kernel_size=(1, 5), padding="same").double()
        self.act1 = nn.Tanh()
        self.norm1 = nn.BatchNorm2d(10).double()

        self.conv2 = nn.Conv2d(10, 5, kernel_size=(1, 5),
                               padding="same").double()
        self.act2 = nn.Tanh()
        self.norm2 = nn.BatchNorm2d(5).double()

        self.conv2_1 = nn.Conv2d(5, 1, kernel_size=(1, 5),
                               padding="same").double()
        self.act2_1 = nn.ReLU()
        self.norm2_1 = nn.BatchNorm2d(1).double()

        # channel selection
        self.conv3 = nn.Conv2d(1, 1, kernel_size=(5, 1), padding="valid",
                               stride=(4, 1)).double()
        self.act3 = nn.Tanh()
        self.norm3 = nn.BatchNorm2d(1).double()

        self.conv4 = nn.Conv2d(1, 1, kernel_size=(2, 5), padding=(0, 2),
                               stride=(1, 1)).double()
        self.act4 = nn.Tanh()
        self.norm4 = nn.BatchNorm2d(1).double()
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2))

        self.conv5 = nn.Conv2d(1, 10, kernel_size=(1, 5),
                               padding="same").double()
        self.act5 = nn.Tanh()
        self.norm5 = nn.BatchNorm2d(10).double()
        self.pool5 = nn.MaxPool2d(kernel_size=(1, 2))
        self.drop5 = nn.Dropout(p=0.9)

        self.linear6 = nn.Linear(620, 40).double()

    def forward(self, x):
        """Convolutional Neural Network forward pass."""
        batch_size = x.shape[0]

        # filterbank
        out = self.conv1(x)
        # out = self.norm1(out)
        out = self.act1(out)
        # print(f"shape of output after layer 1: {out.shape}")

        # second stage of filter bank
        out = self.conv2(out)
        out = self.act2(out)
        out = self.norm2(out)

        out = self.conv2_1(out)
        out = self.act2_1(out)
        out = self.norm2_1(out)
        # out = self.pool2(out)
        # out = self.drop2(out)
        # print(f"shape of output after layer 2: {out.shape}")

        # channel selection
        out = self.conv3(out)
        out = self.act3(out)
        ut = self.norm3(out)
        # print(f"shape of out after layer 3: {out.shape}")

        out = self.conv4(out)
        out = self.act4(out)
        out = self.norm4(out)
        out = self.pool4(out)
        # print(f"shape of out after layer 4: {out.shape}")

        out = self.conv5(out)
        out = self.act5(out)
        out = self.norm5(out)
        out = self.pool5(out)
        out = self.drop5(out)
        # print(f"shape of out after layer 5: {out.shape}")

        out = out.view(batch_size, -1)
        # print(f"shape of out after view: {out.shape}")

        out = self.linear6(out)
        # print(f"shape of out after layer 6: {out.shape}")
        # exit()

        return out


class bci_cnn_eeg_first(nn.Module):
    """
    Definition of the convolutional neural network used for SSVEP-BCI systems.

    Loss function at page 187!!!
    """

    def __init__(self, input_shape, input_files, output_files):
        """ Convolutional Neural Network definition."""
        super().__init__()


        # filter bank
        self.conv1 = nn.Conv2d(9, 20, kernel_size=1, padding="same").double()
        self.act1 = nn.ReLU()
        # self.norm1 = nn.BatchNorm2d(20).double()
        self.dropout1 = nn.Dropout(p=0.6)


        self.conv2 = nn.Conv2d(20, 20, kernel_size=(5, 3),
                               padding="same").double()
        self.act2 = nn.ReLU()
        # self.norm2 = nn.BatchNorm2d(15).double()
        self.dropout2 = nn.Dropout(p=0.6)
        self.pool2 = nn.MaxPool2d(kernel_size= 2)


        self.conv3 = nn.Conv2d(20, 50, kernel_size=(3, 3),
                               padding="same").double()
        self.act3 = nn.ReLU()
        # self.norm3 = nn.BatchNorm2d(50).double()
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(p=0.6)


        self.conv4 = nn.Conv2d(50, 50, kernel_size=1, padding="same").double()
        self.act4 = nn.ReLU()
        self.norm4 = nn.BatchNorm2d(50).double()
        self.dropout4 = nn.Dropout(p=0.6)


        self.linear5 = nn.Linear(3100, 40).double()

    def forward(self, x):
        """Convolutional Neural Network forward pass."""
        batch_size = x.shape[0]

        # filterbank
        out = self.conv1(x)
        out = self.act1(out)
        out = self.dropout1(out)
        # print(f"shape of output after layer 1: {out.shape}")


        # second stage of filter bank
        out = self.conv2(out)
        out = self.act2(out)
        # out = self.norm2(out)
        out = self.pool2(out)
        out = self.dropout2(out)
        # print(f"shape of output after layer 2: {out.shape}")


        # channel selection
        out = self.conv3(out)
        out = self.act3(out)
        # out = self.norm3(out)
        out = self.pool3(out)
        out = self.dropout3(out)
        # print(f"shape of out after layer 3: {out.shape}")


        out = self.conv4(out)
        out = self.act4(out)
        out = self.norm4(out)
        out = self.dropout4(out)
        # print(f"shape of out after layer 4: {out.shape}")


        out = out.view(batch_size, -1)
        # print(f"shape of out after view: {out.shape}")

        out = self.linear5(out)
        # print(f"shape of out after layer 6: {out.shape}")
        # exit()
        # exit()

        return out


class beta_dataset(torch.utils.data.Dataset):
    """
    Create a dataset from the BETA database for SSVEP BCI.
    """

    def __init__(self, input_shape, input_files, output_files, num_blocks,
                 num_chars, eeg_first=False):
        """
        Parameters
        ----------
        input_shape : tuple
            len of input shape is the number of dimensions. Each element value
            corresponds to the cardinality of that dimension
        input_files : list
            list of input files to be loaded with np.load
        output_files : list
            list of output (labels) files to be loaded with np.load
        num_blocks : int
            number of blocks performed in each test
        num_chars : int
            number of characters in screen keyboard
        eef_first : boo
        """
        num_subjects = len(input_files)
        _ds_ = np.zeros((input_shape[0], input_shape[1], input_shape[2],
                         num_chars * num_blocks * num_subjects))
        _labels_ = np.zeros(num_chars * num_blocks * num_subjects)
        zip_files = zip(input_files, output_files)

        for subject, (_input_, _output_) in enumerate(zip_files):
            input = np.load(_input_)
            output = np.load(_output_)
            for blck in range(num_blocks):
                for char in range(num_chars):
                    _ds_[:, :, :, subject * num_blocks * num_chars
                        + blck * num_chars + char] = \
                        input[:, :, :, char, blck]
                    _labels_[subject * num_blocks * num_chars
                            + blck * num_chars + char] = \
                        output[char, blck]
            del(input)
            del(output)

        if eeg_first:
            _ds_ = np.transpose(_ds_, axes=[1, 0, 2, 3])

        self.eeg_signals = torch.from_numpy(_ds_)
        self.labels = torch.from_numpy(_labels_)


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
