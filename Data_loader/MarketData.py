import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt 
import torch
from datetime import datetime
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import pickle

test_flat = (4, 25)
test_time_series = (4, 25)
train_time_series = (4, 25)
train_flat = (4, 25)

class TsDS(Dataset):
    def __init__(self, XL,yL,flatten=False,lno=None,long=True):
        self.samples=[]
        self.labels=[]
        self.flatten=flatten
        self.lno=lno
        self.long=long
        self.scaler = StandardScaler()
        for X,Y in zip(XL,yL):
            self.samples += [torch.tensor(X).float()]
            self.labels += [torch.tensor(Y)]
    def __len__(self):
        return sum([s.shape[0] for s in self.samples])
    def __getitem__(self, idx):
        if self.flatten: sample=self.samples[idx].flatten(start_dim=1)
        else: sample=self.samples[idx]
        if self.lno==None: label=self.labels[idx]
        elif self.long: label=self.labels[idx][:,self.lno].long()
        else: label=self.labels[idx][:,self.lno].float()
        return (sample,label)
    def fit(self,kind='seq'):
        if kind=='seq':
            self.lastelems=[torch.cat([s[:,-1,:] for s in self.samples],dim=0)]
            self.scaler.fit(torch.cat([le for le in self.lastelems],dim=0))            
        elif kind=='flat': self.scaler.fit(torch.cat([s for s in self.samples],dim=0))
    def scale(self,kind='flat',scaler=None):
        def cs(s):
            return (s.shape[0]*s.shape[1],s.shape[2])
        if scaler==None: scaler=self.scaler
        if kind=='seq':
            self.samples=[torch.tensor(scaler.transform(s.reshape(cs(s))).reshape(s.shape)).float() for s in self.samples]
            pass
        elif kind=='flat':
            self.samples=[torch.tensor(scaler.transform(s)).float() for s in self.samples]
    def unscale(self,kind='flat',scaler=None):
        def cs(s):
            return (s.shape[0]*s.shape[1],s.shape[2])
        if scaler==None: scaler=self.scaler
        if kind=='seq':
            self.samples=[torch.tensor(scaler.inverse_transform(s.reshape(cs(s))).reshape(s.shape)).float() for s in self.samples]
            pass
        elif kind=='flat':
            self.samples=[torch.tensor(scaler.inverse_transform(s)).float() for s in self.samples]
import random

class DataLoaderMarket():
    def __init__(self) -> None:
        self.meta_test_cs = []
        self.meta_train_cs = []
        self.meta_test_ds = []
        self.meta_train_ds = []

        for i in range(test_flat[0]):
            for j in range(test_flat[1]):
                train_data = []
                test_data = []
                with open('./Data/market_data/train_cs_' + str(i) + '_' + str(j) + '_2.pickle','rb') as f: 
                    a=pickle.load(f)
                    for k in range(len(a.samples)):
                        train_data.append([a.samples[k], a.labels[k]])

                with open('./Data/market_data/test_cs_' + str(i) + '_' + str(j) + '_2.pickle','rb') as f: 
                    a=pickle.load(f)
                    for k in range(len(a.samples)):
                        test_data.append([a.samples[k], a.labels[k]])
                
                if(j < 20):
                    self.meta_train_cs.append([train_data, test_data])   
                else:
                    self.meta_test_cs.append([train_data, test_data])

        for i in range(test_time_series[0]):
            for j in range(test_time_series[1]):
                train_data = []
                test_data = []
                with open('./Data/market_data/train_ds_' + str(i) + '_' + str(j) + '_2.pickle','rb') as f: 
                    a=pickle.load(f)
                    for k in range(len(a.samples)):
                        train_data.append([a.samples[k], a.labels[k]])

                with open('./Data/market_data/test_ds_' + str(i) + '_' + str(j) + '_2.pickle','rb') as f: 
                    a=pickle.load(f)
                    for k in range(len(a.samples)):
                        test_data.append([a.samples[k], a.labels[k]])
                
                if(j < 20):
                    self.meta_train_ds.append([train_data, test_data])   
                else:
                    self.meta_test_ds.append([train_data, test_data])
            
        self.len_train_ds = len(self.meta_train_ds)
        self.len_test_ds = len(self.meta_test_ds)
        self.len_train_cs = len(self.meta_train_cs)
        self.len_test_cs = len(self.meta_test_cs)
    def get_task(self, kind = "meta_train", data_type = "flat"):
        ind = 0
        if(kind == "meta_train"):
            if(data_type == "flat"):
                ind = random.randint(0, self.len_train_cs - 1)
            else:
                ind = random.randint(0, self.len_train_ds - 1)
        else:
            if(data_type == "flat"):
                ind = random.randint(0, self.len_test_cs - 1)
            else:
                ind = random.randint(0, self.len_test_ds - 1)
        if(kind == "meta_train"):
            if(data_type == "flat"):
                return (self.meta_train_cs[ind][0], self.meta_train_cs[ind][1])
            
            return (self.meta_train_ds[ind][0], self.meta_train_ds[ind][1])
        else:
            if(data_type == "flat"):
                return (self.meta_test_cs[ind][0], self.meta_test_cs[ind][1])
            
            return (self.meta_test_ds[ind][0], self.meta_test_ds[ind][1])

if __name__ == "__main__":
    loader = DataLoaderMarket()
    (train, test) = loader.get_task(kind = "meta_test", data_type= "time_series");
    for i in train:
        print(i[0].shape, i[1].shape)
    for i in test:
        print(i[0].shape, i[1].shape)