import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set()

from tqdm import tqdm

import torch
from torch import nn,optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os

# Load Digits Data

#データはdigitsを使用すると、
from sklearn.datasets import load_digits
digits=load_digits()
X=digits.data
y=digits.target
print(X.shape)
print(y.shape)
#(1797, 64)
#(1797,)

from sklearn.model_selection import train_test_split
#pytorch用のTensorに変換する。
#stratify=yを指定することでデータ分類を安定させておく.
X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1,stratify=y)
print(X_train.shape,y_train.shape)
#testデータとして確保
print(X_val.shape,y_val.shape)


#100個をテストデータとして確保
X_test=X_train[:100]
y_test=y_train[:100]
X_train=X_train[100:]
y_train=y_train[100:]

print(X_train.shape)　　
print(X_val.shape)
print(X_test.shape)
>>>
#(1517, 64)
#(180, 64)
#(100, 64)


#整数値を扱う場合はLongTensorに変換する必要あり。
from torch.utils.data import TensorDataset,DataLoader
ds_train=TensorDataset(torch.Tensor(X_train),torch.LongTensor(y_train))
ds_val=TensorDataset(torch.Tensor(X_val),torch.LongTensor(y_val))
#Dataloaderを定義
train_dataloader=DataLoader(ds_train,batch_size=1000,shuffle=True)
val_dataloader=DataLoader(ds_val,batch_size=1000,shuffle=False)

#dict
dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

#outputs_sample
batch_iterator=iter(dataloaders_dict["train"])
inputs,labels=next(batch_iterator)
print(inputs.shape) 
print(labels.shape)
#print(labels)
#出力可能

from torch import nn
class mlp_model(nn.Module):
    def __init__(self):
        super(mlp_model,self).__init__()
        self.fc1=nn.Linear(64,128)
        self.fc2=nn.Linear(128,10)
        self.relu = nn.ReLU()

    def forward(self,x):
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x

model=mlp_model()
output=model(inputs)
print(output.shape)

#出力
loss = F.nll_loss(output,labels)
print(loss)
