#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 13:50:51 2024

@author: shen
"""
import torch
from torch import nn
import torch.nn.functional as F


def relu2(x):
    return nn.functional.relu(x).pow(2)

class ReLU2(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return relu2(x)


class DQRP(torch.nn.Module):
    def __init__(self, width_vec: list = None,activation='ReLU'):
        super(DQRP, self).__init__()
        self.width_vec= width_vec
        self.activation=activation
        modules = []
        if width_vec is None:
            width_vec = [256, 256, 256]
        if self.activation=='ReQU':
            # Network
            for i in range(len(width_vec) - 2):
                modules.append(
                    nn.Sequential(
                        nn.Linear(width_vec[i],width_vec[i+1]),
                        ReLU2()))
        if self.activation=='ReLU':
            # Network
            for i in range(len(width_vec) - 2):
                modules.append(
                    nn.Sequential(
                        nn.Linear(width_vec[i],width_vec[i+1]),
                        nn.ReLU()))

        self.net = nn.Sequential(*modules,
                                 nn.Linear(width_vec[-2],width_vec[-1]))

    def forward(self,x,u):
        x=torch.cat((x,u),dim=1)
        output = self.net(x)
        return  output


class DQR(torch.nn.Module):
    def __init__(self, width_vec: list = None, Noncrossing = False):
        super(DQR, self).__init__()
        self.width_vec= width_vec
        self.Noncrossing = Noncrossing
        modules = []
        if width_vec is None:
            width_vec = [256, 256, 256]
    
        for i in range(len(width_vec) - 2):
                modules.append(
                    nn.Sequential(
                        nn.Linear(width_vec[i],width_vec[i+1]),
                        nn.ReLU()))

        self.net = nn.Sequential(*modules,
                                 nn.Linear(width_vec[-2],width_vec[-1]))

    def forward(self,x):
            if self.Noncrossing == False:
                output = self.net(x)
            if self.Noncrossing == True: 
                # Output the smallerst quantile 
                h_0 = self.net(x)[:,0].unsqueeze(1);
                # Output the values for compute gaps among quantiles 
                gaps = self.net(x)[:,1:];
                # Apply positive activation function to the gaps
                gaps = torch.log(1 + torch.exp(gaps))
                # Perform cumsum operation on gaps
                cumsum_gaps = torch.cumsum(gaps, dim=1)
                # Add smallerst quantile and cumsum_output to form the final output vector
                output = torch.cat((h_0,h_0 + cumsum_gaps),dim=1)
                
            return  output

   

class DQR_NC(torch.nn.Module):
    def __init__(self, value_layer: list = None, delta_layer:list = None, activation='ELU'):
        super(DQR_NC, self).__init__()
        self.value_layer = value_layer
        self.delta_layer = delta_layer
        self.activation = activation
        if value_layer is None:
            value_layer = [256, 256, 256];
        if delta_layer is None:
            delta_layer = [256, 256, 256];
        # Value Network
        value_layers = []
        for i in range(len(value_layer) - 2):
                value_layers.append(
                    nn.Sequential(
                        nn.Linear(value_layer[i],value_layer[i+1]),
                        nn.ReLU()))

        self.value = nn.Sequential(*value_layers,nn.Linear(value_layer[-2],value_layer[-1]))
                                 
        # Delta Network
        delta_layers = []
        for i in range(len(delta_layer) - 2):
                delta_layers.append(
                    nn.Sequential(
                        nn.Linear(delta_layer[i],delta_layer[i+1]),
                        nn.ReLU()))

        self.delta = nn.Sequential(*delta_layers, nn.Linear(delta_layer[-2],delta_layer[-1]))
                                 
    def forward(self, x):
        value = self.value(x)
        if self.activation=="ELU":
            delta = torch.nn.functional.elu(self.delta(x))+1;
        if self.activation=="ReLU":
            delta = torch.nn.functional.relu(self.delta(x));
        if self.activation=="log":
            delta = torch.log(1 + torch.exp(self.delta(x)));
        # Perform cumsum operation on delta
        cumsum_delta = torch.cumsum(delta, dim=1)
        # Subtract the mean from elu_output to make the mean 0
        cumsum_delta0 = cumsum_delta - cumsum_delta.mean(dim=1, keepdim=True)
        # Add value and cumsum_output to form the final output vector
        output = value + cumsum_delta0
        return output


class NC_QR(torch.nn.Module):
    def __init__(self, value_layer: list = None, delta_layer:list = None):
        super(NC_QR, self).__init__()
        self.value_layer = value_layer
        self.delta_layer = delta_layer        
        if value_layer is None:
            value_layer = [256, 256, 256];
        if delta_layer is None:
            delta_layer = [256, 256, 256];

        # Q Network
        q_layers = []
        for i in range(len(value_layer) - 2):
            q_layers.append(
                nn.Sequential(
                    nn.Linear(value_layer[i],value_layer[i+1]),
                    nn.ReLU()))

        # Weight Network
        weight_layers = []
        for i in range(len(delta_layer) - 2):
            weight_layers.append(
                nn.Sequential(
                    nn.Linear(delta_layer[i],delta_layer[i+1]),
                    nn.ReLU()))

        self.weight = nn.Sequential(*weight_layers, nn.Linear(delta_layer[-2], 2))      
        self.q_net = nn.Sequential(*q_layers,nn.Linear(value_layer[-2], value_layer[-1]), nn.Softmax())
                                 
    def forward(self, x):
        quantiles = self.q_net(x)
        weight = self.weight(x)
        w = weight[:,0].unsqueeze(dim=1).expand(quantiles.shape)
        b = weight[:,1].unsqueeze(dim=1).expand(quantiles.shape)
        probs = torch.cumsum(quantiles, dim =1)
        #print(f"w_shape{w.shape}", f"prob_shape{probs.shape}")
        #print(w[0], b[0], probs[0])
        # print(w.shape, probs.shape)
        nc_quantities = w*probs +b

        return nc_quantities