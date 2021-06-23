#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob, sys, shutil, time, datetime, logging, random, torch, torchvision, h5py
import pandas as pd
import numpy as np
import torch.nn as nn

from sklearn.metrics import roc_curve, auc
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF


# ## PATH SETTINGS

# In[3]:

BASEDIR = f'/data_8t/bao/cell_detection/'


# In[4]:


img_li = glob.glob(f"{BASEDIR}/*/*png")
cancer_img_li = glob.glob(f"{BASEDIR}/cancer/*png")
stromal_img_li = glob.glob(f"{BASEDIR}/stromal/*png")
other_img_li = glob.glob(f"{BASEDIR}/other/*png")


# In[5]:


def get_data_info(img_li):
    cancer_num = sum([1 if 'cancer' in elem else 0 for elem in img_li])
    stromal_num = sum([1  if 'stromal' in elem else 0 for elem in img_li])
    other_num = sum([1  if 'other' in elem else 0 for elem in img_li])
    return [cancer_num, stromal_num, other_num]

def classify_img(img_li, pred_arr, basedir=f'/data_8t/bao/cell_classification'):

    for i, img in enumerate(img_li):
        filename = os.path.basename(img)
        if pred_arr[i] == 0:
            os.makedirs(f'{basedir}/cancer', exist_ok=True)
            dstfile = f'{basedir}/cancer/{filename}'
        elif pred_arr[i] == 1:
            os.makedirs(f'{basedir}/stromal', exist_ok=True)
            dstfile = f'{basedir}/stromal/{filename}'
        else:
            os.makedirs(f'{basedir}/other', exist_ok=True)
            dstfile = f'{basedir}/other/{filename}'
        shutil.copyfile(img, dstfile)

def gen_features(net, sub_li, DATAPATH, DSTPATH, name='all'):
    basefilename = 'resnet18_embedding.hdf5'
    net.eval()
    sub_len = len(sub_li)

    for i, data in enumerate(sub_li):
        sub_id = sub_li[i]
        sub_img_li = glob.glob(f'{DATAPATH}/{sub_id}/*png')
        sub_img_li.sort()
        sub_dataset = net.Dataset(sub_img_li, state='val')
        sub_loader = torch.utils.data.DataLoader(sub_dataset, 
                                                 batch_size=net.batch_size, 
                                                 shuffle=False, 
                                                 num_workers=1)
        sub_slide_out = []
        with torch.no_grad():
            for sub_patch_index, sub_data in enumerate(sub_loader):
                img_data, label = sub_data
                sub_patch_pred = net(to_cuda(img_data)).argmax(dim=1)
                if name.lower() in ['cancer', 'cancers']:
                    sub_patch_out = net.embedding(to_cuda(img_data))
                    sub_patch_out = sub_patch_out[sub_patch_pred==0]
                elif name.lower() in ['stromal', 'lym', 'lymphocyte', 'lymphocytes']:
                    sub_patch_out = net.embedding(to_cuda(img_data))
                    sub_patch_out = sub_patch_out[sub_patch_pred==1]
                elif name.lower() in ['cell', 'cells']:
                    sub_patch_out = net.embedding(to_cuda(img_data))
                    sub_patch_out = sub_patch_out[sub_patch_pred!=2]
                else:
                    sub_patch_out = net.embedding(to_cuda(img_data))
                sub_slide_out.append(sub_patch_out.data.cpu())
        if len(sub_slide_out) == 0:
            print(sub_id)
        else:
            sub_slide_out = torch.cat(sub_slide_out, dim=0)
            os.makedirs(f"{DSTPATH}/{sub_id}", exist_ok=True)
            with h5py.File(f"{DSTPATH}/{sub_id}/{sub_id}_{name}_{basefilename}", "w") as f:
                dataset = f.create_dataset("data", data=sub_slide_out)


# ## DATASET

# In[6]:

class Rotation90:
    """Rotate by one of the given angles."""

    def __call__(self, x):
        if torch.rand(1) > 0.5:
            return TF.rotate(x, 90)
        else:
            return x

class PathologicalDataset(Dataset):
    """docstring for Dataset"""
    def __init__(self, img_li, state='train'):
        self.img_li = list(img_li)
        self.state = state
        if state == 'train':
            self.transform = transforms.Compose([
                                                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=[0.5, 3], hue=0.05),
                                                    transforms.Resize(224),
                                                    Rotation90(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                    transforms.RandomVerticalFlip(),
                                                    transforms.RandomHorizontalFlip(),
                                                ])
        else:
            self.transform = transforms.Compose([
                                                    transforms.Resize(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                ])

    def __len__(self):
        return len(self.img_li)

    def __getitem__(self, index):
        if 'cancer' in self.img_li[index]:
            label = 0
        elif 'stromal' in self.img_li[index]:
            label = 1
        else:
            label = 2

        data = Image.open(self.img_li[index])
        data = data.resize((128, 128))
        data = self.transform(data)

        return (data, label)


class Net(nn.Module):
    def __init__(self, channels=6, classes=2, freeze=True):
        super(Net, self).__init__()
        self.best_acc, self.best_loss = 0, np.inf
        self.lr_dict = {
                        'decay_rate': 0.1,
                        'patience': 15,
                        'min_lr': 1e-6
                       }
        self.resnet = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        if freeze is True:
            self.freeze_model(self.resnet)

        self.classifier = nn.Sequential(nn.Linear(512, 3))

    def __str__(self):
        return str(self.classifier)

    def freeze_model(self, model):
        for p in model.parameters():
            p.requires_grad = False
            
    def forward(self, x):
        out = self.resnet(x)
        out = out.flatten(start_dim=1)
        out = self.classifier(out)
        return out
    
    def embedding(self, x):
        out = self.resnet(x)
        out = out.flatten(start_dim=1)
        return out
    
    def load_cell_model(self, model_name):
        params = torch.load(f"/home/bao/model/{model_name}", map_location='cpu')
        self.classifier[0].weight = nn.Parameter(params['classifier.0.weight'], requires_grad=False)
        self.classifier[0].bias = nn.Parameter(params['classifier.0.bias'], requires_grad=False)


# In[ ]:





# ## Net Model

# In[ ]:





# In[ ]:





# ## Params

# In[7]:


GPU_device = 1
num_epochs = 60
batch_size = 64
num_workers = 4
learning_rate = 1e-2
l2_rate = 0

model_name = f'Cell_detection_model_CNN_one_layer_pretrained.ptl'

print(model_name)

model_config = {
    'device': GPU_device,
    'epochs': num_epochs,
    'shuffle': True,
    'batch_size': batch_size,
    'num_worker': num_workers,
    'lr': learning_rate,
    'l2': l2_rate,
}


# In[8]:


net = Net(channels=3, freeze=True)


# In[9]:


print('--------------------------------------------------------------------------------------------')
print('Network structure')
print(net)
print('--------------------------------------------------------------------------------------------')
total_params = sum(p.numel() for p in net.parameters())
train_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print('Total params:{:,}'.format(total_params))
print('Train params:{:,}'.format(train_params))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, momentum=0.9, weight_decay=l2_rate)


# In[10]:

net.load_state_dict(torch.load(f'/home/bao/model/{model_name}', map_location='cpu'), strict=False)
# net.load_cell_model(model_name)
net.set_criteria(criterion)
net.set_optim(optimizer)
net.set_GPU(GPU_device)

srcname = 'pathology_norm'
datasetname = 'resnet18_embedding_cell'
center = 'guangzhou'
DATAPATH = f'/data_8t/bao/{center}/{srcname}/'
DSTPATH = f'/data_8t/bao/{center}/{datasetname}/'
sub_li = os.listdir(DATAPATH)
gen_features(net, sub_li, DATAPATH, DSTPATH, name='cell')

center = 'foshan'
DATAPATH = f'/data_8t/bao/{center}/{srcname}/'
DSTPATH = f'/data_8t/bao/{center}/{datasetname}/'
sub_li = os.listdir(DATAPATH)
gen_features(net, sub_li, DATAPATH, DSTPATH, name='cell')

center = 'shantou'
DATAPATH = f'/data_8t/bao/{center}/{srcname}/'
DSTPATH = f'/data_8t/bao/{center}/{datasetname}/'
sub_li = os.listdir(DATAPATH)
gen_features(net, sub_li, DATAPATH, DSTPATH, name='cell')

center = 'huaxi'
DATAPATH = f'/data_8t/bao/{center}/{srcname}/'
DSTPATH = f'/data_8t/bao/{center}/{datasetname}/'
sub_li = os.listdir(DATAPATH)
gen_features(net, sub_li, DATAPATH, DSTPATH, name='cell')