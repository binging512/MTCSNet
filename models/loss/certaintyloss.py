import torch
import torch.nn as nn
import torch.nn.functional as F

class CertaintyLoss(nn.Module):
    def __init__(self, args, ignore_index = 255):
        super(CertaintyLoss, self).__init__()
        self.ignore_index = ignore_index
        self.celoss = nn.CrossEntropyLoss(ignore_index=255)
        self.num_classes = args.net_num_classes

    def forward(self, pred, cert, anno):
        scaled_pred = pred/(torch.exp(cert).repeat(1, self.num_classes, 1, 1))
        loss = self.celoss(scaled_pred, anno)
        return loss