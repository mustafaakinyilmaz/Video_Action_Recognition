#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
import itertools
import warnings
warnings.filterwarnings('ignore')
import skimage as sk
import logging
import matplotlib.pyplot as plt
from PIL import Image
from natsort import natsorted
logging.basicConfig(filename='vgg16_lstm.log',level=logging.INFO)

device = torch.device("cuda:0")

truncation_size = 8
np.random.seed(5)


# In[ ]:


normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   transforms.Resize((224,224)),
   transforms.ToTensor(),
   normalize
])


# In[ ]:


def one_hot(array, num_classes):
    return np.squeeze(np.eye(num_classes)[array.reshape(-1)])


# In[ ]:


def data_loader(path,shuffle,batch_size=1):
    all_video_list = []
    all_class_list = []
    
    train_list = natsorted(glob.glob(path+"*"))
    class_no = 0
    for classes in train_list:
        class_list = glob.glob(classes+"/*")
        for folders in class_list:
            all_video_list.append(folders)
            all_class_list.append(class_no)    
        class_no += 1
        
    random_video = np.array(all_video_list)
    random_class = np.array(all_class_list)
    if shuffle == True:
        index = np.random.permutation(len(all_video_list))
        random_video = random_video[index]
        random_class = random_class[index]
    
    for i in range(0,len(all_video_list),batch_size):
        yield(random_video[i:i+batch_size],random_class[i:i+batch_size])


# In[ ]:


def prepare_data(video,clss):
    X = []
    Y = []
    img_list = glob.glob(video[0]+"/*.jpg")
    img_list = natsorted(img_list)
    for jpg in img_list:
        image = Image.open(jpg)
        im_array = preprocess(image).numpy()
        X.append(im_array)
        Y.append(clss)
    return np.array(X),one_hot(np.array(Y),99)


# In[ ]:


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
    l2_lambda = 0.3
    
    loss = log_loss + l2_lambda*l2_reg

    real_arg = torch.argmax(real,dim=1)
    out_arg = torch.argmax(out,dim=1)
    correct = sum(out_arg==real_arg).item()
    return loss,correct


# In[ ]:


def model():

    model = models.vgg16(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    p = 0
    for child in model.features.children():
        if p >= 19:
            for param in child.parameters():
                param.requires_grad = True
        p += 1

    return model.features


class LSTMCell(nn.Module):
    def __init__(self, flatten_dim, hidden_size, bias):
        super(LSTMCell, self).__init__()
    
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size

        self.bias        = bias
        
        self.i2h = nn.Linear(flatten_dim, 4*hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 4*hidden_size, bias=bias)
        
 
        
    def forward(self, x, cur_state):
        
        if cur_state is None:
            cur_state = (Variable(torch.zeros(1, self.hidden_size)).to(device).float(),
                Variable(torch.zeros(1, self.hidden_size)).to(device).float())

        x = x.view(1,-1)
        c_cur, h_cur = cur_state
        
        preact = self.i2h(x) + self.h2h(h_cur)
        #print(preact.size())
        ingate, forgetgate, cellgate, outgate = preact.chunk(4, 1)
        
        
        ingate = F.sigmoid(ingate)
        forgetgate = F.sigmoid(forgetgate)
        cellgate = F.tanh(cellgate)
        outgate = F.sigmoid(outgate)
        
        c_next = (forgetgate * c_cur) + (ingate * cellgate)
        h_next = outgate * F.tanh(c_next)
        
        next_state = (c_next,h_next)
        
        #softmax_out = F.softmax(self.linear(h_next.view(1,-1)),dim=1)
        
        return next_state
    

class VGG16LSTM(nn.Module):
    def __init__(self, original_model):
        super(VGG16LSTM, self).__init__()
        
        self.cnn = original_model
        self.lstm1 = LSTMCell(512*7*7,256,True).to(device).float()
        self.lstm2 = LSTMCell(256,256,True).to(device).float()
        self.fc = nn.Linear(in_features=256,out_features=99,bias=True)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self,x,state1,state2):
        conv = self.cnn(x)
        c1,h1 = self.lstm1(conv,state1)
        c2,h2 = self.lstm2(h1,state2)
        
        fully_c = self.fc(h2)
        
        
        next_state1 = (c1,h1)
        next_state2 = (c2,h2)

        softmax = self.log_softmax(fully_c)    
        return next_state1,next_state2,softmax


# In[ ]:


def train(model,optimizer,train_path,device,truncation_size):
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    train_samples = 0
    train_batch = data_loader(path=train_path,shuffle=True)
    for video,clss in train_batch:
        X,Y = prepare_data(video,clss)
        X,Y = torch.from_numpy(X).to(device).float(),torch.from_numpy(Y).to(device).float()
        
        video_loss = 0.0
                            
        state1 = None
        state2 = None
        
        for trunck in range(0,X.size(0),truncation_size):
            X_trunck = X[trunck:trunck+truncation_size]
            Y_trunck = Y[trunck:trunck+truncation_size]
            
            trunck_loss = 0.0

            if state1 is not None and state2 is not None:

                state1 = (Variable(state1[0].data),Variable(state1[1].data))
                state2 = (Variable(state2[0].data),Variable(state2[1].data))
                
            for i in range(X_trunck.size(0)):
                
                inp = X_trunck[i:i+1]
                real = Y_trunck[i:i+1]                   
                state1,state2,softmax_out = model.forward(inp,state1,state2)
                loss,correct = evaluate(softmax_out,real,model)
                trunck_loss += loss
                video_loss += loss
                
            average_trunck_loss = trunck_loss / X_trunck.size(0)
            optimizer.zero_grad()
            average_trunck_loss.backward()
            optimizer.step()
        
        average_video_loss = video_loss / X.size(0)
        print(softmax_out.argmax(1),real.argmax(1))
        epoch_train_loss += average_video_loss.item()
        epoch_train_correct += correct
        train_samples += 1
        if train_samples % 10 == 0:
            print(epoch_train_loss/train_samples)
            print(epoch_train_correct,train_samples)

        
    return epoch_train_loss,epoch_train_correct,train_samples


# In[ ]:


def test(model,test_path,device):
    epoch_test_loss = 0.0
    epoch_test_correct = 0
    test_samples = 0
    test_batch = data_loader(path=test_path,shuffle=True)
    for video,clss in test_batch:
        X,Y = prepare_data(video,clss)
        X,Y = torch.from_numpy(X).to(device).float(),torch.from_numpy(Y).to(device).float()
                
        state1 = None
        state2 = None
        
        video_loss = 0
        
        for i in range(X.size(0)):
            inp = X[i:i+1]
            real = Y[i:i+1]                   
            state1,state2,softmax_out = model.forward(inp,state1,state2)
            loss,correct = evaluate(softmax_out,real,model)
            video_loss += loss
        
        average_video_loss = video_loss / X.size(0)
        
        epoch_test_loss += average_video_loss.item()
        epoch_test_correct += correct
        test_samples += 1

    return epoch_test_loss,epoch_test_correct,test_samples


def save_model(epoch,model,optimizer):
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    }
    torch.save(state, "vgg16_lstm_"+str(epoch)+".pth")


# In[ ]:


def train_model():
    train_path = "C:\\Users/Akin Yilmaz/Desktop/MS_ee/Video_Recognition/Frames/Train/"
    test_path = "C:\\Users/Akin Yilmaz/Desktop/MS_ee/Video_Recognition/Frames/Test/"
    
    
    orig_model = model()
    vgg16_model = VGG16LSTM(orig_model).to(device).float()
    
    learning_rate = 1.e-5
    
    
    optims = filter(lambda p: p.requires_grad,vgg16_model.parameters())
    optimizer = torch.optim.RMSprop(optims,lr=learning_rate)
    
    nb_epoch = 25

    
    for epoch in range(1,nb_epoch+1):
        
        vgg16_model = vgg16_model.train()
        epoch_train_loss,epoch_train_correct,train_samples = train(vgg16_model,optimizer,train_path,device,truncation_size)
        logging.info("epoch: "+str(epoch))
        logging.info("train_loss: "+str(epoch_train_loss/train_samples))
        logging.info("train_corrects: "+str(epoch_train_correct)+"/"+str(train_samples))
        logging.info("-------")
        
        vgg16_model = vgg16_model.eval()
        epoch_test_loss,epoch_test_correct,test_samples = test(vgg16_model,test_path,device)
        logging.info("test_loss: "+str(epoch_test_loss/test_samples))
        logging.info("test_corrects: "+str(epoch_test_correct)+"/"+str(test_samples))
        logging.info("***********************")
        
        save_model(epoch,vgg16_model,optimizer)


# In[ ]:


train_model()


# In[ ]:


"""orig_model = model()
vgg16_model = VGG16LSTM(orig_model).to(device).float()
model_parameters = filter(lambda p: p.requires_grad, vgg16_model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
params"""


# In[ ]:


for a in range(0,32,8):
    print(a)


# In[ ]:




