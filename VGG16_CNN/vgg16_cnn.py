#!/usr/bin/env python
# coding: utf-8



import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import torch.nn.functional as F
import glob
import os
from skimage.transform import resize
#import cv2
import itertools
import warnings
warnings.filterwarnings('ignore')
import skimage as sk
import logging
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted

logging.basicConfig(filename='vgg16_cnn.log',level=logging.INFO)
device = torch.device("cuda")




normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])




def data_loader(path,batch_size,shuffle):
    all_img_list = []
    all_class_list = []
    
    train_list = natsorted(glob.glob(path+"*"))
    class_no = 0
    for classes in train_list:
        class_list = glob.glob(classes+"/*")
        for folders in class_list:
            img_list = glob.glob(folders+"/*.jpg")
            for images in img_list:
                all_img_list.append(images)
                all_class_list.append(class_no)
        
        class_no += 1
        
    random_img = np.array(all_img_list)
    random_class = np.array(all_class_list)
    if shuffle == True:
        index = np.random.permutation(len(all_img_list))
        random_img = random_img[index]
        random_class = random_class[index]
    
    for i in range(0,len(all_img_list),batch_size):
        yield(random_img[i:i+batch_size],random_class[i:i+batch_size])



def one_hot(array, num_classes):
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])

def prepare_data(im_list,cls_list):
    X = []
    Y = []
    for im,cl in zip(im_list,cls_list):
        image = Image.open(im)
        im_array = preprocess(image).numpy()
        X.append(im_array)
        Y.append(cl)
    return np.array(X),one_hot(np.array(Y),99)




def evaluate(out,real,mdl):
    e = torch.exp(out)
    log_loss = torch.sum(-real*out)
    mean_loss = torch.sum((e-real)**2)

    l1_reg = None
    l2_reg = None
    for W in mdl.parameters():
        if l1_reg is None:
            l1_reg = W.norm(1)
        else:
            l1_reg = l1_reg + W.norm(1)
            
        if l2_reg is None:
            l2_reg = W.norm(2)
        else:
            l2_reg = l2_reg + W.norm(2)
    
    l1_lambda = 0.0
    l2_lambda = 1.0
    
    loss = log_loss + l2_lambda*l2_reg

    real_arg = torch.argmax(real,dim=1)
    out_arg = torch.argmax(out,dim=1)
    correct = sum(out_arg==real_arg).item()
    return loss,correct


# In[ ]:


def model():

    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True


    return model.features

print(model())

class VGG16(nn.Module):
    def __init__(self, original_model):
        super(VGG16, self).__init__()
        self.feature_extractor = original_model
        self.fc1 = nn.Linear(in_features=512*7*7,out_features=2048,bias=True)
        self.fc2 = nn.Linear(in_features=2048,out_features=99,bias=True)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self,x):
        (m,nc,nw,nh) = x.size()
        cnn = self.feature_extractor(x).view(m,-1)
        fc1 = self.fc1(cnn)
        relu1 = F.relu(fc1)
        drop1 = self.dropout(relu1)
        fc2 = self.fc2(drop1)
        out = F.log_softmax(fc2,dim=1)
        return out


# In[ ]:


def train(model,optimizer,batch_size,train_path,device):
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    train_samples = 0
    train_batch = data_loader(train_path,batch_size,True)
    for im_list,cls_list in train_batch:
        X,Y = prepare_data(im_list,cls_list)
        X,Y = torch.from_numpy(X).to(device).float(),torch.from_numpy(Y).to(device).float()
        
        out = model.forward(X)
        batch_loss,batch_correct = evaluate(out,Y,model)
        
        batch_loss_av = batch_loss/X.size(0)
        
        optimizer.zero_grad()
        batch_loss_av.backward()
        optimizer.step()
        epoch_train_loss += batch_loss.item()
        epoch_train_correct += batch_correct
        train_samples += X.size(0)
        
    return epoch_train_loss,epoch_train_correct,train_samples

def test(model,batch_size,test_path,device):
    epoch_test_loss = 0.0
    epoch_test_correct = 0
    test_samples = 0
    test_batch = data_loader(test_path,batch_size,False)
    for im_list,cls_list in test_batch:
        X,Y = prepare_data(im_list,cls_list)
        X,Y = torch.from_numpy(X).to(device).float(),torch.from_numpy(Y).to(device).float()
        
        out = model.forward(X)
        batch_loss,batch_correct = evaluate(out,Y,model)
        
        epoch_test_loss += batch_loss.item()
        epoch_test_correct += batch_correct
        test_samples += X.size(0)
    
    return epoch_test_loss,epoch_test_correct,test_samples


def save_model(epoch,model,optimizer):
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }
    torch.save(state, "vgg16_cnn"+str(epoch)+".pth")


# In[ ]:


def train_model():
    train_path = "C:\\Users/Akin Yilmaz/Desktop/MS_ee/Video_Recognition/Frames/Train/"
    test_path = "C:\\Users/Akin Yilmaz/Desktop/MS_ee/Video_Recognition/Frames/Test/"
        
    orig_model = model()
    vgg16_model = VGG16(orig_model).to(device).float()
    
    learning_rate = 1.e-5
    
    
    optims = filter(lambda p: p.requires_grad,vgg16_model.parameters())
    optimizer = torch.optim.Adam(optims,lr=learning_rate)
    
    nb_epoch = 25
    train_batch_size = 16
    test_batch_size = 16
    
    for epoch in range(1,nb_epoch+1):
        
        vgg16_model = vgg16_model.train()
        epoch_train_loss,epoch_train_correct,train_samples = train(vgg16_model,optimizer,train_batch_size,train_path,device)
        logging.info("epoch: "+str(epoch))
        logging.info("train_loss: "+str(epoch_train_loss/train_samples))
        logging.info("train_corrects: "+str(epoch_train_correct)+"/"+str(train_samples))
        logging.info("-----")
        
        vgg16_model = vgg16_model.eval()
        epoch_test_loss,epoch_test_correct,test_samples = test(vgg16_model,test_batch_size,test_path,device)
        logging.info("test_loss: "+str(epoch_test_loss/test_samples))
        logging.info("test_corrects: "+str(epoch_test_correct)+"/"+str(test_samples))
        logging.info("********************")
        
        save_model(epoch,vgg16_model,optimizer)


# In[ ]:


train_model()


# In[ ]:





# In[ ]:




