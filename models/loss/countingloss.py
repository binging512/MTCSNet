import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss
import numpy as np
import math

class Bulk_Loss(_Loss):
    def __init__(self):
        super(Bulk_Loss, self).__init__()
    def forward(self, input, target, mask=None):
        C = 1e-32
        assert C > 0.0, 'Only C over zero'
        mul = torch.sqrt(input * target+C).sum()
        dis = 1.0 - (2.0*mul)/(input+target+2*math.sqrt(C)).sum()
        return dis

class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss,self).__init__()
        self.eps=1e-5
    def forward(self,x,y):
        total_x = torch.sum(x)
        total_y = torch.sum(y)
        err = torch.abs(total_x-total_y)
        loss = torch.log10(err/100 + 1)
        # loss = err/(total_y + self.eps)
        return loss

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss,self).__init__()
        self.eps=1e-5
    def forward(self,x,y):
        total_x = torch.sum(x)
        total_y = torch.sum(y)
        err = torch.pow((total_x-total_y),2)
        loss = err/(torch.pow(total_y,2) + self.eps)
        return loss

class BlockMAELoss(nn.Module):
    def __init__(self,block_size=16):
        super(BlockMAELoss,self).__init__()
        self.loss_layer = nn.Conv2d(1,1,(block_size,block_size),stride=block_size,bias=False)
        weight = torch.ones((1,1,block_size,block_size))
        self.loss_layer.weight = torch.nn.Parameter(weight)
        self.loss_layer.eval()
        self.eps = 1e-10
        #print(self.loss_layer.weight)
    def forward(self,x,y):
        self.loss_layer.eval()
        total_x = torch.sum(x)
        total_y = torch.sum(y)
        total = total_x + total_y
        x = self.loss_layer(x)
        y = self.loss_layer(y)
        loss_map = torch.abs(x-y)
        loss = torch.sum(loss_map)
        loss = loss/(total+self.eps)
        return loss

class BlockMSELoss(nn.Module):
    def __init__(self,block_size=16):
        super(BlockMSELoss,self).__init__()
        self.loss_layer = nn.Conv2d(1,1,(block_size,block_size),stride=block_size,bias=False)
        weight = torch.ones((1,1,block_size,block_size))
        self.loss_layer.weight = torch.nn.Parameter(weight)
        self.loss_layer.eval()
        self.eps = 1e-10
        #print(self.loss_layer.weight)
    def forward(self,x,y):
        total_x = torch.sum(x)
        total_y = torch.sum(y)
        x = self.loss_layer(x)
        y = self.loss_layer(y)
        loss = torch.pow((x-y),2)
        loss = torch.sum(loss)
        loss = loss/(total_x+total_y+self.eps)
        return loss

#==== pyramid loss ====#
class PyMAELoss(nn.Module):
    def __init__(self,block_size=[8,16,32]):
        super(PyMAELoss,self).__init__()
        self.num_layer = len(block_size)
        self.loss_layer = nn.ModuleList()
        for ii, bs in enumerate(block_size):
            self.loss_layer.append(nn.Conv2d(1,1,(bs,bs),stride=bs,bias=False))
            weight = torch.ones((1,1,bs,bs))
            self.loss_layer[-1].weight = torch.nn.Parameter(weight)
            self.loss_layer[-1].eval()
        self.eps=1e-10
    def forward(self,x,y):
        total_x = torch.sum(x)
        total_y = torch.sum(y)
        loss_list=[]
        total_loss = 0
        # total_loss = torch.tensor(0,dtype=torch.float32,device='cuda')
        for ii in range(self.num_layer):
            block_x = self.loss_layer[ii](x)
            block_y = self.loss_layer[ii](y)
            loss = torch.abs(block_x-block_y)
            loss = torch.sum(loss)
            loss = loss/(total_x+total_y+self.eps)
            loss_list.append(loss)
            total_loss += loss
        return total_loss,loss_list
