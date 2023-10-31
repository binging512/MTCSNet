import torch
import torch.nn as nn
from .ssimloss import SSIM

class ConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super(ConsistencyLoss, self).__init__()
        self.mseloss = nn.MSELoss()
        self.ssimloss = SSIM()
        
    def forward(self, x ,y):
        mse = self.mseloss(x, y)
        ssim = self.ssimloss(x,y)
        # loss = mse
        loss = mse+ssim
        return loss