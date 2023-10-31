import torch.nn as nn
import torch.nn.functional as F
import torch 

class PixelContrastLoss(nn.Module):
    def __init__(self):
        super(PixelContrastLoss, self).__init__()

    def forward(self, inputs, targets):
        B,C,H,W = inputs.shape
        inputs = inputs.permute(0,2,3,1).contiguous()
        targets = targets.permute(0,2,3,1).contiguous()
        inputs = inputs.reshape(B*H*W,C)
        targets = targets.reshape(B*H*W,C)
        sim = F.cosine_similarity(inputs,targets,dim=1)
        sim = torch.mean(sim)
        return 1-sim
        
if __name__=="__main__":
    a = torch.arange(0,512).unsqueeze(1).unsqueeze(0).unsqueeze(0)
    a = a.expand((2,3,512,512))
    b = torch.arange(0,512).unsqueeze(0).unsqueeze(0).unsqueeze(0)
    b = b.expand((2,3,512,512))
    loss_func = PixelContrastLoss()
    loss = loss_func(a.float(),a.float())