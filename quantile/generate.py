#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 16:15:52 2021

@author: shenguohao
"""
#%%
import numpy as np
import torch
import torch.utils.data as Data
import scipy.stats as st


#%%

def block(input):
    t=torch.FloatTensor([0.1,0.15,0.23,0.28,0.40,0.44,0.65,0.76,0.78,0.81])
    h=torch.FloatTensor([4,-5,0,1.5,0,2.5,-2.3,0,-1.2,-1.2])
    fx=torch.sum(torch.mul((torch.sgn(torch.sub(input,t))+1)/2,h),1)
    output=torch.FloatTensor(fx.reshape(-1,1))
    return output
#x=torch.linspace(0,1,1000).unsqueeze(1)
#y=block(x)
#plt.plot(x,y)

def bump(input):
    t=torch.FloatTensor([0.1,0.15,0.23,0.28,0.40,0.44,0.65,0.76,0.78,0.81])
    h=torch.FloatTensor([4,5,2.5,4,3,2.1,4.3,1.1,2.1,4.2])
    w=2*torch.FloatTensor([0.005,0.005,0.006,0.01,0.01,0.03,0.01,0.01,0.005,0.008])
    fx=torch.sum(torch.mul(torch.pow(torch.abs(torch.sub(input,t))/w+1,-4),h),1)
    output=torch.FloatTensor(fx.reshape(-1,1))
    return output


def gen_univ(size=2**10,model='wave',error='expsinex',df=2,sigma=1,xi='uniform',alpha=0.5,beta=0.5):
    x = torch.rand([size,1]).float()
    errors={'t':torch.from_numpy(np.random.standard_t(df,[size,1])),
            'normal':torch.randn([size,1]),
            'cauchy':torch.from_numpy(np.random.standard_cauchy([size,1])),
            'sinex':torch.randn(x.shape)*torch.sin(np.pi*x),
            'expx':torch.randn(x.shape)*(torch.exp((x-0.5)*2)),
            'cross': torch.randn(x.shape)*torch.sin(0.9*np.pi*x),
            }
    eps=errors[error].float()
    ys={'wave':2*x*torch.sin(4*np.pi*x)+sigma*eps,
         'linear':2*x+sigma*eps,
         'exp': torch.exp(2*x)+sigma*eps,
         'blocks': block(x)+sigma*eps,
         'triangle': (4-4*torch.abs(x-0.5))+sigma*eps,
         'iso':2*np.pi*x+torch.sin(2*np.pi*x)+sigma*eps,
         'constant': torch.ones_like(x)+sigma*eps,
        }
    if xi=='uniform':
        u=torch.rand([size,1]).float();
    if xi=='beta':
        u=torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta])).sample([size]).float();
    y=ys[model].float()
    return Data.TensorDataset(x, y, u, eps)


def gen_multi(A=None,B=None,size=2**10,d=8,model='sim',error='t',df=3,sigma=0.5,xi='uniform',alpha=0.5,beta=0.5):
    if A==None:
        torch.manual_seed(2022);A=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
    if B==None:
        torch.manual_seed(2021);B=-1*torch.randint(0,3,[d,1])*torch.randn([d,1]);
    x = torch.rand([size,d]).float()
    errors={'t':torch.from_numpy(np.random.standard_t(df,[size,1])),
            'normal':torch.randn([size,1]),
            'sinex':torch.randn([size,1])*(torch.sin(np.pi*torch.from_numpy(np.dot(x.data.numpy(),B.data.numpy())))),
            'expx':torch.randn([size,1])*(torch.exp((torch.from_numpy(np.dot(x.data.numpy(),B.data.numpy()))-0.5)*0.1))
            }
    eps=errors[error].float()
    ys={'sim':torch.exp(0.1*torch.from_numpy(np.dot(x.data.numpy(),A.data.numpy())))+sigma*eps,
        'add':3*x[:,0].unsqueeze(1)+4*torch.pow(x[:,1].unsqueeze(1)-0.5,2)+2*torch.sin(np.pi*x[:,2].unsqueeze(1))-5*torch.abs(x[:,3].unsqueeze(1)-0.5)+sigma*eps,
        'linear':2*torch.from_numpy(np.dot(x.data.numpy(),A.data.numpy()))+sigma*eps,
        }
    if xi=='uniform':
        u=torch.rand([size,1]).float();
    if xi=='beta':
        u=torch.distributions.Beta(torch.tensor([alpha]), torch.tensor([beta])).sample([size]).float();
    y=ys[model].float()
    return Data.TensorDataset(x, y, u, eps)


def quant_multi(x,taus,A,B,model='sim',error='t',df=3,sigma=0.5):
    if x.shape[0]!=taus.shape[0]:
        taus=(taus.T).repeat([x.shape[0],1]);
    errors={'t':torch.from_numpy(sigma*st.t.ppf(taus,df=df)),
            'sinex':torch.abs(torch.sin(np.pi*torch.from_numpy(np.dot(x.data.numpy(),B.data.numpy()))))*sigma*st.norm.ppf(taus),
            'expx':(torch.exp((torch.from_numpy(np.dot(x.data.numpy(),B.data.numpy()))-0.5)*0.1))*sigma*st.norm.ppf(taus),
            'normal':torch.from_numpy(sigma*st.norm.ppf(taus)),
            }
    eps=errors[error].float()
    quantiles={'sim':torch.exp(0.1*torch.from_numpy(np.dot(x.data.numpy(),A.data.numpy())))+eps,
               'add':3*x[:,0].unsqueeze(1)+4*torch.pow(x[:,1].unsqueeze(1)-0.5,2)+2*torch.sin(np.pi*x[:,2].unsqueeze(1))-5*torch.abs(x[:,3].unsqueeze(1)-0.5)+eps,
               'linear':2*torch.from_numpy(np.dot(x.data.numpy(),A.data.numpy()))+eps,
        }
    quantile=quantiles[model].float()
    return quantile


def quant_univ(x,taus,model='sine',error='expx',df=2,sigma=1):
    if x.shape[0]!=taus.shape[0]:
        taus=(taus.T).repeat([x.shape[0],1]);
    errors={'t':torch.from_numpy(sigma*st.t.ppf(taus,df=df)),
            'normal':torch.from_numpy(sigma*st.norm.ppf(taus)),
            'sinex':torch.sin(np.pi*x)*sigma*st.norm.ppf(taus),
            'expx':torch.exp((x-0.5)*2)*sigma*st.norm.ppf(taus),
            'cross':torch.sin(0.9*np.pi*x)*sigma*st.norm.ppf(taus),
            }
    eps=errors[error].float()
    quantiles={'wave':2*x*torch.sin(4*np.pi*x)+eps,
         'linear':2*x+eps,
         'exp': torch.exp(2*x)+eps,
         'blocks': block(x)+eps,
         'triangle': (4-4*torch.abs(x-0.5))+eps,
         'iso':2*np.pi*x+torch.sin(2*np.pi*x)+eps,
         'constant': torch.ones_like(x)+eps,
        }
    quantile=quantiles[model].float()
    return quantile
    
    


#%%
class qloss(torch.nn.Module):
    def __init__(self,mode='process',reduction='mean'):
        super(qloss,self).__init__()
        self.reduction = reduction
        self.mode = mode
    def derive(self,x,y,u):
        diff = torch.sub(x,y)
        index = (diff>0).float()
        totloss = (u-1)*(1-index)+u*index
        return totloss
    def forward(self,x,y,u):
        size = y.size()[0]
        diff = torch.sub(y,x)
        index = (diff<0).float()
        if self.mode =='process':
            totloss = diff*(u-index)
        if self.mode =='multiple':
            totloss=0;
            for j in range(u.shape[0]):
                totloss = totloss+ diff*(u[j].repeat([size,1])-index);
        if self.reduction =='mean':
            totloss=torch.sum(totloss/size)
        return totloss    
    
    
    