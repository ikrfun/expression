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

base_dir = '/Users/ikrfun/Desktop/doing_projects/SIGNATE/expression'
data_dir = os.path.join(base_dir,'data/test/')
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
    return model
    
def test(img):
    model = get_trained_model()
    img = img.unsqueeze(dim = 0)
    output = model(img)
    pred = torch.argmax(output,dim=1)
    return pred,output

def get_result():
    pred_list = []
    file_list = os.listdir(path = '/Users/ikrfun/Desktop/doing_projects/SIGNATE/expression/data/test')
    transform = transforms.Compose([
        transforms.Resize(300),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    for i in range(len(file_list)):
        file_path = os.path.join(data_dir,file_list[i])
        img = np.array(Image.open(file_path))
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        img = transform(img)
        pred,raw = test(img)
        print(raw)
        pred_list.append(pred)
    return pred_list

print(get_result())