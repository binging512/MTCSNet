import numpy as np
import torch

def rand_bbox(size, lam):  # size bchw  H和W写反没关系，反正是正方形
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def CutMix(args, img, anno, heatmap, deg_map, dist_map, dist_mask, vor):
    '''
    :param args:
    :param img: BCHW
    :param anno: BHW
    :param heatmap: BHW
    :param boundary: BHW
    :return:
    '''
    r = np.random.rand(1)
    if args.beta > 0 and r < args.cutmix_prob:  # 就进行cutmix
        # generate mixed sample
        lam = np.random.beta(args.beta, args.beta)  # 随机的lam
        rand_index = torch.randperm(img.size()[0]).cuda()  # 打乱的序号
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)  # 随机产生一个box的四个坐标
        img[:, :, bbx1:bbx2, bby1:bby2] = img[rand_index, :, bbx1:bbx2, bby1:bby2]  # 现在的img是混合后的img
        anno[:, bbx1:bbx2, bby1:bby2] = anno[rand_index, bbx1:bbx2, bby1:bby2]
        heatmap[:, bbx1:bbx2, bby1:bby2] = heatmap[rand_index, bbx1:bbx2, bby1:bby2]
        deg_map[:, bbx1:bbx2, bby1:bby2] = deg_map[rand_index, bbx1:bbx2, bby1:bby2]
        dist_map[:, bbx1:bbx2, bby1:bby2] = dist_map[rand_index, bbx1:bbx2, bby1:bby2]
        dist_mask[:, bbx1:bbx2, bby1:bby2] = dist_mask[rand_index, bbx1:bbx2, bby1:bby2]
        vor[:, bbx1:bbx2, bby1:bby2] = vor[rand_index, bbx1:bbx2, bby1:bby2]
            
        return img, anno, heatmap, deg_map, dist_map, dist_mask, vor #, boundary
    else:
        return img, anno, heatmap, deg_map, dist_map, dist_mask, vor#, boundary