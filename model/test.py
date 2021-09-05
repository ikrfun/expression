import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

import torch.optim as optim
import torchvision
import torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Dataset,ConcatDataset, dataloader
from PIL import Image
import cv2
import os

#CPU環境で学習済みモデルを回すためのコード

base_dir = '/content/drive/MyDrive/コンペ/expression'
data_dir = os.path.join(base_dir,'data/train/')
file_list_file = os.path.join(base_dir,'data/train_master.csv')
device = torch.device('cuda')
label_dic = {'sad':0,'nue':1,'hap':2,'ang':3}
model_path = './model_path.pth'


#学習済みパラメータを読み込んでモデルを復元する
def get_model():
    pretrained_model = models.resnet18(pretrained=True)
    pretrained_model.fc = nn.Linear(512,4)
    return pretrained_model

def get_trained_model():
    model = get_model()
    model_path = 'model.pth'
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

def get_dataset():
    file_list = os.listdir(path = '/Users/ikrfun/Desktop/doing_projects/SIGNATE/expression/data/test')
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    for i in range(len(file_list)):
        

def test():
    model = get_trained_model()
    