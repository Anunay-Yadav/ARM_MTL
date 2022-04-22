import numpy as np
import pandas as pd
import json, os
import matplotlib.pyplot as plt 
from matplotlib import colors
from PIL import Image
import io
import random
import copy
import pickle
import random
import torch

class ARC():
    def __init__(self,trn_dir='./training_orig/',tes_dir='./test_eval/'):
        pass
    def plot_task(self,task,kind='orig',show=True,ways=4):
        # Call with ways=4 for padded case and ways=6 for unpadded case
        n = len(task["train"]) + len(task["test"])
        if kind=='orig':fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
        elif kind=='fewshot': fig, axs = plt.subplots(ways+1, n, figsize=(6*n,12), dpi=100)
        plt.subplots_adjust(wspace=0, hspace=0)
        fig_num = 0
        cmap=self.cmap
        norm=self.norm
        for i, t in enumerate(task["train"]):
            if kind=='fewshot':t_in, t_out = np.array(t["input"]), t["output"]
            elif kind=='orig':t_in, t_out = np.array(t["input"]), np.array(t["output"])
            axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
            axs[0][fig_num].set_title(f'Train-{i} in')
            # axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
            # axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
            if kind=='orig':
                axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
                axs[1][fig_num].set_title(f'Train-{i} out')
                # axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
                # axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
            elif kind=='fewshot':
                for j in range(ways):
                    if j==t['label']: iscorrect='CORRECT'
                    else: iscorrect=''
                    axs[j+1][fig_num].imshow(np.array(t_out[j]), cmap=cmap, norm=norm)
                    axs[j+1][fig_num].set_title(f'Out-{i},{j} '+iscorrect)
                    # axs[j+1][fig_num].set_yticks(list(range(np.array(t_out[j]).shape[0])))
                    # axs[j+1][fig_num].set_xticks(list(range(np.array(t_out[j]).shape[1])))
            fig_num += 1
        for i, t in enumerate(task["test"]):
            if kind=='fewshot':t_in, t_out = np.array(t["input"]), t["output"]
            elif kind=='orig':t_in, t_out = np.array(t["input"]), np.array(t["output"])
            axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
            axs[0][fig_num].set_title(f'Test-{i} in')
            # axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
            # axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
            if kind=='orig' and show:
                axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
                axs[1][fig_num].set_title(f'Test-{i} out')
                # axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
                # axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
            elif kind=='fewshot' and show:
                for j in range(ways):
                    if j==t['label']: iscorrect='CORRECT'
                    else: iscorrect=''
                    axs[j+1][fig_num].imshow(np.array(t_out[j]), cmap=cmap, norm=norm)
                    axs[j+1][fig_num].set_title(f'Test-{i},{j} '+iscorrect)
                    # axs[j+1][fig_num].set_yticks(list(range(np.array(t_out[j]).shape[0])))
                    # axs[j+1][fig_num].set_xticks(list(range(np.array(t_out[j]).shape[1])))
            fig_num += 1
        plt.tight_layout()
        plt.show()
    def example2img(self,example):
        shp=np.array(example).shape
        fig=plt.Figure(figsize=(.5*shp[0],.5*shp[1]))
        ax = fig.add_subplot()
        cmap,norm=self.cmap,self.norm
        ax.imshow(np.array(example), cmap=cmap, norm=norm)
        """Convert a Matplotlib figure to a PIL Image and return it"""
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
    def example2numpy(self,example):
        return np.array(example)


class FewShotARC(ARC):
    def __init__(self,trn_dir='./training_orig/',tes_dir='./test_eval/',ways=6):
        super().__init__(trn_dir='./training_orig/',tes_dir='./test_eval/')
        self.nrand=ways-1
        self.ntrain=len(self.trn_tasks)
        self.ntest=len(self.tes_tasks)
        self.meta_train_tasks=[]
        self.meta_test_tasks=[]
    def get_fs_task(self,taskid,kind='meta_train'):
        if kind=='meta_train': return self.meta_train_tasks[taskid]
        elif kind=='meta_test': return self.meta_test_tasks[taskid]
    def get_examples(self,taskid,trte,inout,kind='meta_train'):
        if kind=='meta_train':taskL=[self.get_task(taskid,kind) for taskid in self.ntrain]
        elif kind=='meta_test':taskL=[self.get_task(taskid,kind) for taskid in self.ntrain]
        return [taskL[taskid][trte][k][inout] for k in range(len(taskL[taskid][trte]))]

class FewShotPaddedARC(ARC):
    def __init__(self,trn_dir='./training_orig/',tes_dir='./test_eval/',ways=6):
        super().__init__(trn_dir='./training_orig/',tes_dir='./test_eval/')
        self.nrand=ways-1
        self.ntrain=len(self.trn_tasks)
        self.ntest=len(self.tes_tasks)
        self.meta_train_tasks=[]
        self.meta_test_tasks=[]
    def get_fs_task(self,taskid,kind='meta_train'):
        if kind=='meta_train': return self.meta_train_tasks[taskid]
        elif kind=='meta_test': return self.meta_test_tasks[taskid]
    def get_examples(self,taskid,trte,inout,kind='meta_train'):
        if kind=='meta_train':taskL=[self.get_task(taskid,kind) for taskid in self.ntrain]
        elif kind=='meta_test':taskL=[self.get_task(taskid,kind) for taskid in self.ntrain]
        return [taskL[taskid][trte][k][inout] for k in range(len(taskL[taskid][trte]))]


class DataLoader():
    def __init__(self, data_path):
        if(data_path == "FewShotPaddedARC"):
            with open('./Data/FewShotPaddedARC.pickle','rb') as f: 
                b=pickle.load(f)
                b.cmap=colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
                b.norm=colors.Normalize(vmin=0, vmax=9)
                self.data = b
        else:
            with open('./Data/FewShotARC.pickle','rb') as f: 
                a=pickle.load(f)
                a.cmap=colors.ListedColormap(['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00','#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
                a.norm=colors.Normalize(vmin=0, vmax=9)
                self.data = a
        self.train_tasks = self.data.trn_tasks
        self.test_tasks = self.data.ntest
    def get_task(self, kind = "meta_train", technique = 2):
        ind = 0
        if(kind == "meta_train"):
            ind = random.randint(0, self.train_task - 1)
        else:
            ind = random.randint(0, self.test_task - 1)
        # print(ind)
        b = self.data.get_fs_task(ind, kind)
        train = b["train"]
        test = b["test"]
        if(technique == 2):
          return self.process(train), self.process(test)
        return self.process2(train), self.process2(test)
    def process(self, batch):
        pair_list, pair_label = [], []
        for ind, sample in enumerate(batch):
            if("label" not in sample):
                continue
            train = sample["input"]
            label = sample["label"]
            test = sample["output"]
            train_torch = torch.FloatTensor(train)
            train_torch = torch.flatten(train_torch, start_dim = 0)
            # train_torch = torch.unsqueeze(train_torch, 0)
            
            for ind_out, j in enumerate(test):
                if(len(j) == len(train)):
                    test_torch = torch.FloatTensor(j)
                    test_torch = torch.flatten(test_torch, start_dim = 0)
                    pair_list.append(torch.cat((train_torch, test_torch), 0))
                    pair_label.append(ind_out == label)
        # print(torch.stack(pair_list).shape)
        return (torch.stack(pair_list), torch.LongTensor(pair_label)) 
    def process2(self, batch):
        pair_list, pair_label = [], []
        for ind, sample in enumerate(batch):
            if("label" not in sample):
                continue
            train = sample["input"]
            label = sample["label"]
            test = sample["output"]
            train_torch = torch.FloatTensor(train)
            train_torch = torch.flatten(train_torch, start_dim = 0)
            # train_torch = torch.unsqueeze(train_torch, 0)
            
            for ind_out, j in enumerate(test):
                if(len(j) == len(train)):
                    test_torch = torch.FloatTensor(j)
                    test_torch = torch.flatten(test_torch, start_dim = 0)
                    train_torch = torch.cat((train_torch, test_torch), 0)
            pair_label.append(label)
            pair_list.append(train_torch)
        # print(torch.stack(pair_list).shape)
        return (torch.stack(pair_list), torch.LongTensor(pair_label)) 

if __name__ == "__main__":
    loader = DataLoader("FewShotPaddedARC")
    padded_task = loader.data.get_fs_task(0, kind = "meta_test")
    loader.data.plot_task(padded_task,kind='fewshot')