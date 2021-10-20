# obsolete file

import tensorflow as tf

import numpy as np

from helpers import addColorChannels


class GeneratorInstance(tf.keras.utils.Sequence):
    """
    initializes batches of features and targets by iterating through filenames

    parameters
    ----------
    filenames : list or ndarray of shape (number of files in path,)
        names of files in data folder
    batch_size : float
        batch size per training/validation iteration
    num_classes : int
        number of classes in each target
    path : string
        path to data folder (inclusive)
    labelToInteger : functions
        normalization to apply on data
    permutations : functions
        permutation functions from dataPermutation.py
    is_val : bool
        whether the passed data is from training dataset (False) or validation dataset (True)

    returns
    -------
    xBatch : ndarray of shape(batch_size, INPUT_SHAPE[0], INPUT_SHAPE[1], ...)
        batch of features per iteration
        INPUT_SHAPE taken from globalVariables.py
    yBatch : ndarray of shape (batch_size, num_classes)
        batch of targets per iteration
    """

    def __init__(self, filenames, batch_size, num_classes, path, normalization, permutations, permutation_probability, is_val):
        self.filenames = filenames
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.path = path
        self.normalization = normalization
        self.permutations = permutations
        self.permutation_probability = permutation_probability
        self.is_val = is_val

    def __len__(self):
        """
        mandatory method passed to keras.utils.Sequence to ensure the network trains only once on each sample per epoch
        """
        return (np.ceil(len(self.filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        """
        mandatory method passed to keras.utils.Sequence to generate batches of features and targets

        parameters
        ----------
        idx : int
            starting index in variable 'filenames' to generate not overlapping batch of features
        """
        xBatchFilenames = self.filenames[idx *
                                         self.batch_size: (idx + 1) * self.batch_size]
        yBatch = np.zeros((len(xBatchFilenames), self.num_classes))

        xBatch = []
        for idx, filename in enumerate(xBatchFilenames):
            xBatch.append(np.load(self.path + filename))

            if self.num_classes == 2:
                label = filename.split('_')[-1].replace('.npy', '')
                label = label.split('-')

            elif self.num_classes == 4:
                label = filename.replace('.npy', '').replace(
                    '.dcm', '').split('_')[2:]

            label = [int(i) for i in label]

            yBatch[idx] = label

        xBatch = np.array(xBatch)

        if not self.is_val:
            for permutation in self.permutations:
                for idx in range(xBatch.shape[0]):
                    xBatch[idx] = permutation(
                        xBatch[idx], self.permutation_probability)

        for idx in range(xBatch.shape[0]):
            xBatch[idx] = self.normalization(xBatch[idx])

        if self.num_classes == 4:
            xBatch = addColorChannels(xBatch)

        return xBatch, yBatch
