import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Dataset,ConcatDataset
from PIL import Image
import cv2
import os


# パラメータ、pathの設定
base_dir = '/Users/ikrfun/Desktop/doing_projects/SIGNATE/expression/'
lr = 0.001
#使用するモデル:ResNet18（転移学習）
def get_model():
    trained_model = models.resnet18(pretrained=True)
    for param in trained_model.parameters():
        param.requires_grad = False
    trained_model.fc = nn.Linear(512,4)
    return trained_model
        



transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

class FaceDataset(Dataset):
    def __init__(self,label,file_list,transform = None):
        self.label = label
        self.file_list = file_list
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        file_path = os.path.join(base_dir,self.file_list[idx])
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img
        
