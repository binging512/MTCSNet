import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskDistLoss(nn.Module):
    def __init__(self, ignore_index = 255):
        super(MaskDistLoss, self).__init__()
        self.ignore_index = ignore_index
        self.mseloss = nn.MSELoss(reduction='none')
        self.alpha = 1

    def forward(self, pred_dist, gt_dist, mask):
        # pred_dist = torch.tanh(self.alpha * pred_dist)
        # gt_dist = torch.tanh(self.alpha * gt_dist)
        loss = self.mseloss(pred_dist, gt_dist)
        loss = torch.sum(loss*mask)/torch.sum(mask)
        return loss
    
class MaskDistLoss_v2(nn.Module):
    def __init__(self, args) -> None:
        super(MaskDistLoss_v2, self).__init__()
        self.N = args.net_N
        self.mseloss = nn.MSELoss(reduction='none')
        self.L1_loss = nn.L1Loss(reduction='none')
        self.alpha = 0.05
        self.loss_version = 'L2'
    
    def forward(self, pred, gt_dist, gt_deg, mask):
        pred_an = pred[:,          0:   self.N+1, :, :]
        pred_bn = pred[:,   self.N+1: 2*self.N+1, :, :]
        pred_cn = pred[:, 2*self.N+1: 3*self.N+2, :, :]
        pred_dn = pred[:, 3*self.N+2: 4*self.N+2, :, :]
        pred_ts = pred[:, 4*self.N+2: 4*self.N+3, :, :]
        n = torch.ones_like(pred_ts)
        pred_ts = torch.cat([pred_ts]*self.N, dim=1)
        n = torch.cat([n*(i+1) for i in range(self.N)], dim=1)
        pred_dist = pred_an[:,:1,:,:] + torch.sum(pred_an[:, 1:,:,:]*torch.sin(2*torch.pi*n*pred_ts) + pred_bn*torch.cos(2*torch.pi*n*pred_ts), dim=1, keepdim=True)
        pred_theta = pred_cn[:,:1,:,:] + torch.sum(pred_cn[:, 1:,:,:]*torch.sin(2*torch.pi*n*pred_ts) + pred_dn*torch.cos(2*torch.pi*n*pred_ts), dim=1, keepdim=True)
        
        pred_dist = torch.tanh(self.alpha * pred_dist)
        gt_dist = torch.tanh(self.alpha * gt_dist)
        
        # int_theta = torch.floor(pred_theta)
        # pred_theta = pred_theta - int_theta     # keep the theta in 0~1
        pred_theta = torch.clip(pred_theta, 0, 1)
        
        gt_dist = gt_dist.unsqueeze(1)
        gt_deg = gt_deg.unsqueeze(1)
        mask = mask.unsqueeze(1)
        if self.loss_version=='L2':
            loss_dist = torch.sum(self.mseloss(pred_dist, gt_dist)*mask)/torch.sum(mask)
            loss_theta_1 = self.mseloss(pred_theta, gt_deg)
            loss_theta_2 = self.mseloss(pred_theta, gt_deg-1)
            loss_theta_min, _ = torch.min(torch.cat((loss_theta_1,loss_theta_2), dim=1), dim=1, keepdim=True)
            loss_theta = torch.sum(loss_theta_min*mask)/torch.sum(mask)
        elif self.loss_version=='L1':
            loss_dist = torch.sum(self.L1_loss(pred_dist, gt_dist)*mask)/torch.sum(mask)
            loss_theta_1 = self.L1_loss(pred_theta, gt_deg)
            loss_theta_2 = self.L1_loss(pred_theta, gt_deg-1)
            loss_theta_min, _ = torch.min(torch.cat((loss_theta_1,loss_theta_2), dim=1), dim=1, keepdim=True)
            loss_theta = torch.sum(loss_theta_min*mask)/torch.sum(mask)
        
        return loss_dist + loss_theta
    
class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, pred_dist, gt_dist, mask):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
        '''
        mask = mask.permute(0,2,3,1).reshape(-1, 8)[:, 0]
        pos_idx = mask.nonzero().reshape(-1)
        pred_dist = pred_dist.permute(0,2,3,1).reshape(-1, 8)[pos_idx].exp()
        gt_dist = gt_dist.permute(0,2,3,1).reshape(-1, 8)[pos_idx]
        
        total = torch.stack([pred_dist,gt_dist], -1)
        l_max = torch.max(total, dim=2)[0]
        l_min = torch.min(total, dim=2)[0]
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss.sum()/mask.sum()
        return loss