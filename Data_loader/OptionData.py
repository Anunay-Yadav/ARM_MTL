import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt 
import torch
from datetime import datetime
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle
import random

class OptionsDataset():
    def __init__(self):
        self.meta_train_tasks=[]
        self.meta_test_tasks=[]
    def get_meta_train_task(self,r):
        return self.meta_train_tasks[r]
    def get_meta_test_task(self,r):
        return self.meta_train_tasks[r]
import os
class DataLoaderOptions():
    def __init__(self) -> None:
        with open(str(os.getcwd()) + '/Data_loader/Data/optionsMetaTasksPE.pickle','rb') as f: self.options_dataset_PE =pickle.load(f)
         
        with open(str(os.getcwd()) + '/Data_loader/Data/optionsMetaTasksCE.pickle','rb') as f: self.options_dataset_CE=pickle.load(f)

        self.len_meta_train_PE = len(self.options_dataset_PE.meta_train_tasks)
        self.len_meta_test_PE = len(self.options_dataset_PE.meta_test_tasks)
        self.len_meta_train_CE = len(self.options_dataset_CE.meta_train_tasks)
        self.len_meta_test_CE = len(self.options_dataset_CE.meta_test_tasks)
        
    def get_task(self, kind = "meta_train", data_type = "CE"):
        ind = 0
        if(kind == "meta_train"):
            if(data_type == "CE"):
                ind = random.randint(0, self.len_meta_train_CE - 1)
            else:
                ind = random.randint(0, self.len_meta_train_PE - 1)
        else:
            if(data_type == "CE"):
                ind = random.randint(0, self.len_meta_test_CE - 1)
            else:
                ind = random.randint(0, self.len_meta_test_PE - 1)
        if(kind == "meta_train"):
            if(data_type == "CE"):
                return self.options_dataset_CE.get_meta_train_task(ind)
            
            return self.options_dataset_PE.get_meta_train_task(ind)
        else:
            if(data_type == "CE"):
                return self.options_dataset_CE.get_meta_test_task(ind)
            
            return self.options_dataset_PE.get_meta_test_task(ind)

if __name__ == "__main__":
    loader = DataLoaderOptions()

    train, test = loader.get_task("meta_train", "CE")
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)

    train, test = loader.get_task("meta_train", "PE")
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)
    train, test = loader.get_task("meta_test", "CE")
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)

    train, test = loader.get_task("meta_test", "PE")
    print(train[0].shape, train[1].shape)
    print(test[0].shape, test[1].shape)