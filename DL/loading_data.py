#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 13:30:54 2021

@author: xwu05
classes and methods to laod data in pytorch
"""
get_ipython().system('pip install torch')


from torch.utils.data import Dataset
from utils import to_device
from torch.utils.data.dataloader import DataLoader

class CustomData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.y) #返回dataset大小
    
    def __getitem__(self, idx): # 获取指定索引的数据和标签
        X_slice = self.X[idx] 
        y_slice = self.y[idx]
        return X_slice, y_slice
    

class DeviceDataLoader(): #放到GPU上
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self): # 迭代数据集并将数据加载到GPU
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def get_testloader(X_test, y_test, device):  
    
    batch_size= 128
    #train_data = CustomData(X_train, y_train)
    test_data = CustomData(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size, num_workers=4, pin_memory=True)
    test_loader = DeviceDataLoader(test_loader, device)
    return test_loader
