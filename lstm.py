
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import Variable
trunck_size = 1
device = torch.device("cpu")
np.random.seed(321)
torch.manual_seed(123) 


# In[2]:


"""delay_time = 40


dataset = pd.read_csv("btc_usd.csv",sep=',')
prices = dataset.iloc[:,-1:].values

X_ = []
for i in range(delay_time):
    X_.append(list(prices[i:i-delay_time]))

X = np.array(X_).reshape(delay_time,-1).T

Y = prices[delay_time:]

index = np.random.permutation(Y.shape[0])

X_train,Y_train = X[index[:2500]],Y[index[:2500]]
X_test,Y_test = X[index[2500:]],Y[index[2500:]]
max_ = np.max(X_train,axis=0).reshape(1,-1)
min_ = np.min(X_train,axis=0).reshape(1,-1)
X_train = (X_train-min_)/(max_-min_)
X_test = (X_test-min_)/(max_-min_)"""


# In[3]:


X_train = np.array(list(range(-50,11,1))).reshape(1,-1) / 61
print(X_train.shape)
Y_train = np.zeros((1,61))
for k in range(61):
    Y_train[0,k] = ((X_train[0,0:k+1].sum())**2)/10
    


# In[23]:


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
        
        
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        
        c_next = (forgetgate * c_cur) + (ingate * cellgate)
        h_next = outgate * torch.tanh(c_next)
        
        next_state = (c_next,h_next)
        
        #softmax_out = F.softmax(self.linear(h_next.view(1,-1)),dim=1)
        
        return next_state

class RNNCell(nn.Module):
    def __init__(self, flatten_dim, hidden_size, bias):
        super(RNNCell, self).__init__()
    
        self.flatten_dim = flatten_dim
        self.hidden_size = hidden_size

        self.bias        = bias
        
        self.i2h = nn.Linear(flatten_dim, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self,x,cur_h):
        
        if cur_h is None:
            cur_h = Variable(torch.zeros(1, self.hidden_size)).to(device).float()
        
        act = self.i2h(x) + self.h2h(cur_h)
        
        next_h = torch.tanh(act)
        
        return next_h
        
        


# In[24]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__() 
        self.lstm1 = LSTMCell(1,1,True)
        self.lin = nn.Linear(in_features=1,out_features=1,bias=True)
        
    def forward(self,x,state1):
        c1,h1 = self.lstm1(x,state1)
        out = self.lin(h1)
        next_state1 = (c1,h1)
        return next_state1,out

class ModelR(nn.Module):
    def __init__(self):
        super(ModelR, self).__init__() 
        self.lstm1 = RNNCell(1,1,True)
        self.lin = nn.Linear(in_features=1,out_features=1,bias=True)
        
    def forward(self,x,state1):
        h1 = self.lstm1(x,state1)
        out = self.lin(h1)
        next_state1 = h1
        return next_state1,out


# In[25]:


model = Model().to(device).float()
optimizer = torch.optim.RMSprop(model.parameters(),lr=5.e-3)


# In[26]:


modelr = ModelR().to(device).float()
optimizerr = torch.optim.RMSprop(modelr.parameters(),lr=5.e-3)


# In[27]:


def calculate_loss(out,real):
    loss = torch.sum((out-real)**2)
    return loss


# In[28]:


def train(m,opt):
    m = m.train()
    epoch_loss = 0.0
    out_list = []
    for i in range(X_train.shape[0]):
        X_batch = X_train[i:i+1]
        Y_batch = Y_train[i:i+1]
        
        #state1 = None
        
        for trunck in range(0,X_batch.shape[1],trunck_size):
            X_trunck = X_batch[:,trunck:trunck+trunck_size]
            Y_trunck = Y_batch[:,trunck:trunck+trunck_size]
            
            """if state1 is not None:
                state1 = Variable(state1[0].data)"""
            state1 = None

            
            for k in range(X_trunck.shape[1]):
                inp = X_trunck[:,k:k+1]
                real = Y_trunck[:,k:k+1]
                inp = torch.from_numpy(inp).to(device).float()
                real = torch.from_numpy(real).to(device).float()
                state1,out = m.forward(inp,state1)
                loss = calculate_loss(out,real)
                out_list.append(out.item())
            opt.zero_grad()
            loss.backward()
            opt.step()
            
       
        epoch_loss += loss.item()
    
    return epoch_loss,out_list
        
    
    
def test(m):
    m = m.eval()
    epoch_loss = 0.0
    out_list = []
    for i in range(X_train.shape[0]):
        X_batch = X_train[i:i+1]
        Y_batch = Y_train[i:i+1]
        
        state1 = None
        
        for k in range(X_batch.shape[1]):
            inp = X_batch[:,k:k+1]
            real = Y_batch[:,k:k+1]
            inp = torch.from_numpy(inp).to(device).float()
            real = torch.from_numpy(real).to(device).float()
            state1,out = m.forward(inp,state1)
            loss = calculate_loss(out,real)
            out_list.append(out.item())
        epoch_loss += loss.item()
    
    return epoch_loss,out_list


# In[29]:


for epoch in range(1,201,1):
    epoch_loss,out = train(modelr,optimizerr)
    if epoch % 20 == 0:
        print("epoch: "+str(epoch))
        print(epoch_loss)
        print("***")


test_loss,test_out = test(modelr)


# In[30]:


print(np.array(test_out))
print(Y_train)


# In[31]:


test_loss

