import torch
import torch.nn as nn

def tversky_loss(inputs, targets, alpha=.5, beta=.5, sigmoid=True, smooth=1.):
    if sigmoid:
        inputs = inputs.sigmoid()
    a = (inputs * targets).sum()
    loss = (a + smooth) / (smooth + a + alpha * (inputs * (1 - targets)).sum() + beta * ((1 - inputs) * targets).sum())
    return loss


def focal_tversky_loss(inputs, targets, alpha=.5, beta=.5, gamma=1.5, sigmoid=True, smooth=1.):
    loss = tversky_loss(inputs, targets, alpha=alpha, beta=beta, gamma=gamma, sigmoid=sigmoid, smooth=smooth)
    return (1 - loss) ** gamma

class TverskyLoss(nn.Module):
    def __init__(self, sigmoid = True):
        self.alpha = .5
        self.beta = .5
        self.sigmoid = sigmoid
        self.smooth = 1.0
    
    def forward(self, inputs, targets):
        loss = tversky_loss(inputs, targets, self.alpha, self.beta, self.sigmoid, self.smooth)
        return loss
        
class FocalTverskyLoss(nn.Module):
    def __init__(self):
        
        pass
    
        