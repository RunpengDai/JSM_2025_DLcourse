from generate import gen_univ
from generate import quant_univ
import torch
import matplotlib.pyplot as plt
from generate import gen_univ,quant_univ,qloss
from generate import gen_multi,quant_multi
from model import DQRP,DQR,DQR_NC, NC_QR
from functions import train_multi,train_process
import numpy as np



SIZE=2**9
epochs=1000;batch_size=int(SIZE/2)
width=128
check=qloss(mode='multiple'); check1=qloss(mode='process');

def train_test(model, error, df, sigma, taus, d=1):
    x_test=torch.linspace(0,1,1000).unsqueeze(1)

    preds_DQR = torch.zeros([1000,len(taus)]);
    preds_DQR_NC = torch.zeros([1000,len(taus)]);
    preds_NC = torch.zeros([1000,len(taus)]);
    preds_DQRP=torch.zeros([1000,len(taus)]);
    
    # Generating the train/val datasets
    data_train= gen_univ(model=model,size=SIZE,error=error,df=df,sigma=sigma)
    data_val= gen_univ(model=model,size=int(SIZE/4),error=error,df=df,sigma=sigma)

    # Initialte the models
    net_DQR = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=False);optimizer_DQR = torch.optim.Adam(net_DQR.parameters(), lr=0.001);
    net_DQR_NC = DQR(width_vec=[d,width,width,width,len(taus)], Noncrossing=True);optimizer_DQR_NC = torch.optim.Adam(net_DQR_NC.parameters(), lr=0.001);
    #net_DQRP = DQRP(width_vec=[d+1,width,width,width,1],activation='ReQU');optimizer_DQRP=torch.optim.Adam(net_DQRP.parameters(), lr=0.001);
    net_NC = DQR_NC(value_layer=[d,int(width/2),int(width/2),int(width/2),1],delta_layer=[d,int(width/2),int(width/2),int(width/2),len(taus)]); 
    optimizer_NC = torch.optim.Adam(net_NC.parameters(), lr=0.001);
    netNC_QR = NC_QR(delta_layer=[d,int(width/2),int(width/2),int(width/2),1],value_layer=[d,int(width/2),int(width/2),int(width/2),len(taus)])
    optimizer_QR = torch.optim.Adam(netNC_QR.parameters(), lr=0.001);
    # Train the regression models

    print("Train")
    netNC_QR = train_multi(netNC_QR,optimizer_QR, epochs, batch_size,100,check,data_train, data_val,taus);
    net_DQR = train_multi(net_DQR,optimizer_DQR, epochs, batch_size,100,check,data_train, data_val,taus);
    net_DQR_NC = train_multi(net_DQR_NC,optimizer_DQR_NC, epochs, batch_size,100,check,data_train, data_val,taus);
    #net_DQRP = train_process(net_DQRP,optimizer_DQRP,epochs,batch_size,100,np.log(SIZE),check1,data_train,data_val,algo=True)
    net_NC = train_multi(net_NC,optimizer_NC, epochs, batch_size,100,check,data_train, data_val,taus);
        
    # Predict on test set
    for j in range(len(taus)):
        preds_DQR[:,j] = net_DQR(x_test)[:,j].squeeze().detach();
        preds_DQR_NC[:,j] = net_DQR_NC(x_test)[:,j].squeeze().detach();
        preds_NC[:,j] = net_NC(x_test)[:,j].squeeze().detach();
        preds_DQRP[:,j]=netNC_QR(x_test)[:,j].squeeze().detach();
            
    preds=[preds_NC,preds_DQR,preds_DQR_NC,preds_DQRP]
    return preds