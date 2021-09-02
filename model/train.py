import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torchvision
import torch.nn as nn
from torchvision import datasets,transforms,models
from torch.utils.data import DataLoader,Dataset,ConcatDataset, dataloader
from PIL import Image
import cv2
import os


# パラメータ、pathの設定
base_dir = '/Users/ikrfun/Desktop/doing_projects/SIGNATE/expression/'
file_list_file = 'data/train_master.csv'


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

#データセット定義
class FaceDataset(Dataset):
    def __init__(self,label,transform = None):
        self.label = label
        self.file_list = self.get_file_list()
        self.transform = transform
        
    def get_file_list(self):
        df = pd.read_csv(file_list_file)
        file_list = df[self.label]
        return file_list
    
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

#各種データセットのインスタンス化
sad_dataset = FaceDataset('sad',transform=transform)
nue_dataset = FaceDataset('nue',transform=transform)
hap_dataset = FaceDataset('hap',transform=transform)
ang_dataset = FaceDataset('ang',transform=transform)

dataset = ConcatDataset([sad_dataset,nue_dataset,hap_dataset,ang_dataset])
dataloader = DataLoader(dataset,batch_size = 32, shuffle = True)

#訓練開始（追って実装）

lr = 0.001
model = get_model()
optimizer = optim.adam(model.fc.parameters(),lr = lr)

def train():
    pass
