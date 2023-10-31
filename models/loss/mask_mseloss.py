import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskMSELoss(nn.Module):
    def __init__(self, ignore_index = 255):
        super(MaskMSELoss, self).__init__()
        self.ignore_index = ignore_index
        self.mseloss = nn.MSELoss(reduction='none')

    def forward(self, pred_heat, gt_heat, mask):
        gt_heat = gt_heat.unsqueeze(1)
        mask = mask.unsqueeze(1)
        loss = self.mseloss(pred_heat, gt_heat)
        loss = torch.sum(loss*mask)/torch.sum(mask)
        return loss