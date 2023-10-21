import os
import re

import numpy as np

def load_data_withlabels(subject_files):
    """Load data from subject files."""

    signals = []
    labels = []
    sampling_rate = None
    for sf in subject_files:
        with np.load(sf) as f:
            x = f['x']
            y = f['y']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            # x = np.squeeze(x)
            x = np.squeeze(x, axis=(2,))  # Remove the axis=2 dimension
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)
            y = y.astype(np.int32)

            signals.append(x)
            labels.append(y)

    return signals, labels, sampling_rate

def load_data_nolabels(subject_files):
    """Load data from subject files."""
    print("load_data_nolabels......... ")
    signals = []
    sampling_rate = None
    for sf in subject_files:
        print("sf ", sf)
        with np.load(sf) as f:
            x = f['x']
            fs = f['fs']

            if sampling_rate is None:
                sampling_rate = fs
            elif sampling_rate != fs:
                raise Exception("Mismatch sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            np.set_printoptions(formatter={'float': lambda x: "{:.8f}".format(x)})
            # x = np.squeeze(x)
            x = np.squeeze(x, axis=(2,))  # Remove the axis=2 dimension
            x = x[:, :, np.newaxis, np.newaxis]

            # Casting
            x = x.astype(np.float32)

            signals.append(x)

    return signals, sampling_rate
