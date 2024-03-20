#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 14:29:09 2021

@author: xwu05

Main script to fit the neural network
"""
import torch
import numpy as np
from loading_data import CustomData
from utils import get_default_device
from loading_data import DeviceDataLoader
from utils import fit
from utils import to_device
from torch.utils.data.dataloader import DataLoader
from neural_network import NeuralNetwork
import os
import sklearn.utils as skutils
from utils import rev_comp
import sys


file_suffix = sys.argv[1]

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '1' # determine GPU number =1
    X_train = np.load('../dataset/X_train.npy')
    X_train_rev = rev_comp(X_train)
    X_train = np.concatenate([X_train, X_train_rev])  #reverse

    y_train = np.load('../dataset/y_train.npy')
    y_train = np.concatenate([y_train, y_train])

    X_train, y_train = skutils.shuffle(X_train, y_train, random_state=2)
	# random shuffled, reduce overgitting

    X_train = torch.tensor(X_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)# transfer to tensor varibale

    X_val = np.load('../dataset/X_val.npy')
    X_val = torch.tensor(X_val, dtype=torch.float)

    y_val = np.load('../dataset/y_val.npy')
    y_val = torch.tensor(y_val, dtype=torch.float)

    train_data = CustomData(X_train, y_train)
    val_data = CustomData(X_val, y_val)
    # determine if device is CPU or GPU
    device = get_default_device()

    batch_size = 32  # this is the batch size
    train_loader = DataLoader(train_data, batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size,
                            num_workers=4, pin_memory=True)

    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    params = np.load('../saved_models/params.npy', allow_pickle='TRUE').item()
    model = NeuralNetwork(4,
                          params["h"],
                          params["f"],
                          2,
                          params["fcs"],
                          params["p"],
                          params["mha_p"])
    to_device(model, device)

    fit(1000, 5e-4, params["wd"], params["mo"], model, train_loader, val_loader, file_suffix)
