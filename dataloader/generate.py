import os
import torch
import numpy as np


def generate_withlabels(base_path, data_path):

    print("********** data_path", data_path)
    test_files = np.array([os.path.join(base_path, data_path)])
    print("******* test_files ", test_files)
    X_train = np.load(test_files[0])["x"]
    y_train = np.load(test_files[0])["y"]

    print("X_train shape:", X_train.shape)

    for np_file in test_files[1:]:
        X_train = np.vstack((X_train, np.load(np_file)["x"]))
        y_train = np.append(y_train, np.load(np_file)["y"])

    data_save = dict()
    data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    data_save["labels"] = torch.from_numpy(y_train)
    torch.save(data_save, os.path.join(base_path, "data/test_data.pt"))

def generate_nolabels(base_path, data_path):

    test_files = np.array([os.path.join(base_path, data_path)])
    print("******* test_files ", test_files)
    X_train = np.load(test_files[0])["x"]

    print("X_train shape:", X_train.shape)

    for np_file in test_files[1:]:
        X_train = np.vstack((X_train, np.load(np_file)["x"]))

    data_save = dict()
    data_save["samples"] = torch.from_numpy(X_train.transpose(0, 2, 1))
    torch.save(data_save, os.path.join(base_path, "data/test_data.pt"))

# ######## Test One Stage ##########
# # test_files = files[(len_train + len_valid):]
# test_files = test_files
# # load files
# X_train = np.load(test_files[0])["x0"]

# print("X_train shape:", X_train.shape)

# for np_file in test_files[1:]:
#     X_train = np.vstack((X_train, np.load(np_file)["x"]))


# X_tensor = torch.from_numpy(X_train)

# data_save = dict()
# data_save["samples"] = torch.from_numpy(X_train)
# torch.save(data_save, os.path.join(output_dir, "stage_0.pt"))
