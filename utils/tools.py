import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def resize(img, dst_size, mode, if_deg=False):
    ori_h, ori_w = img.shape
    dst_h, dst_w = dst_size
    dst_h = dst_h.item()
    dst_w = dst_w.item()
    if mode =='nearest':
        img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    elif mode == 'bilinear':
        img = cv2.resize(img, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
    
    if if_deg:
        mag_ratio = dst_h/ori_h
        img = img*mag_ratio
    
    return img
    