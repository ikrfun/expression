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


# パラメータ、pathの設定
base_dir = '/content/drive/MyDrive/コンペ/expression'
data_dir = os.path.join(base_dir,'data/train/')
file_list_file = os.path.join(base_dir,'data/train_master.csv')
device = torch.device('cuda')
label_dic = {'sad':0,'nue':1,'hap':2,'ang':3}

#使用するモデル:ResNet18（転移学習）
def get_model():
    trained_model = models.resnet18(pretrained=True)
    for param in trained_model.parameters():
        param.requires_grad = False
    trained_model.fc = nn.Linear(512,4)
    return trained_model




transform = transforms.Compose([
    transforms.Resize(300),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

#データセット定義
class FaceDataset(Dataset):
    def __init__(self,label,transform = None):
        self.label_name=label
        self.label = label_dic[label]
        self.file_list = self.get_file_list()
        self.transform = transform
        
    def get_file_list(self):
        df = pd.read_csv(file_list_file)
        file_list = df[self.label_name].dropna()
        return file_list
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self,idx):
        file_path = os.path.join(data_dir,self.file_list[idx])
        img = np.array(Image.open(file_path))
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label

#各種データセットのインスタンス化
sad_dataset = FaceDataset('sad',transform=transform)
nue_dataset = FaceDataset('nue',transform=transform)
hap_dataset = FaceDataset('hap',transform=transform)
ang_dataset = FaceDataset('ang',transform=transform)

dataset = ConcatDataset([sad_dataset,nue_dataset,hap_dataset,ang_dataset])
dataloader = DataLoader(dataset,batch_size = 32, shuffle = True)

#訓練開始（追って実装）

lr = 0.01
model = get_model()
optimizer = optim.Adam(model.fc.parameters(),lr = lr)
criterion = nn.CrossEntropyLoss()

def train(n_epochs):
    losses = []
    accs = []
    
    for epoch in range(n_epochs):
        running_loss = 0
        running_acc = 0
        for imgs, labels in dataloader:
            model.to(device)
            imgs=imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            output = model(imgs)
            loss = criterion(output,labels)
            running_loss += loss.item()
            pred = torch.argmax(output,dim=1)
            running_acc += torch.mean(pred.eq(labels).float())
            loss.backward()
            optimizer.step()
        
        running_loss /= len(dataloader)
        running_acc /= len(dataloader)
        losses.append(running_loss)
        accs.append(running_acc)
        if epoch%10 ==0:
            print('epoch:{},loss:{},acc{}'.format(epoch,running_loss,running_acc))
    
    plt.plot(losses)
    plt.plot(accs)
    plt.show()
        
train(100)
