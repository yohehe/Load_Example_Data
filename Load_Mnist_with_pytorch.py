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

# Set DataLoader
transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

#Mnist
mnist_train = MNIST('~/tmp/mnist',  train=True, download=True,transform=transform)
mnist_test=MNIST('~/tmp/mnist',  train=False, download=True,transform=transform)

# train (55,000 images), val split (5,000 images)
mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

train_dataloader = DataLoader(mnist_train, batch_size=64)
val_dataloader = DataLoader(mnist_val, batch_size=64)
test_dataloader = DataLoader(mnist_test,batch_size=64)

#dict
dataloaders_dict={"train":train_dataloader,"val":val_dataloader}

#出力確認
batch_iterator=iter(dataloaders_dict["train"])
inputs,labels=next(batch_iterator)
print(inputs.shape) 
print(labels.shape)
print(labels)
>>>
torch.Size([64, 1, 28, 28])
torch.Size([64])
tensor([0, 4, 7, 1, 0, 1, 7, 9, 3, 3, 9, 2, 0, 8, 0, 1, 7, 1, 2, 4, 6, 5, 6, 3,
        8, 6, 6, 2, 8, 8, 0, 3, 1, 7, 4, 2, 6, 3, 6, 9, 1, 3, 1, 2, 2, 2, 8, 1,
        6, 9, 1, 4, 8, 7, 2, 5, 3, 0, 7, 4, 2, 0, 4, 9])
