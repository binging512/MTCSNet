import os
import cv2
import numpy as np
import torch

def gen_grad(img_dir, output_dir):
    img_list = sorted(os.listdir(img_dir))
    for ii,img_name in enumerate(img_list):
        print("Processing the {}/{} images...".format(ii, len(img_list)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path, flags=0)
        img= cv2.GaussianBlur(img, [3, 3], 3)
        sobely = cv2.Sobel(img,cv2.CV_64F,0,1)
        sobelx = cv2.Sobel(img,cv2.CV_64F,1,0)
        absy = cv2.convertScaleAbs(sobely)
        absx = cv2.convertScaleAbs(sobelx)
        dst = cv2.addWeighted(absx, 0.5, absy, 0.5, 0)
        # _, dst = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
        save_path = os.path.join(output_dir,img_name)
        cv2.imwrite(save_path,dst)

def gen_prob_mask(grad_dir, output_dir):
    for ii, grad_name in enumerate(sorted(os.listdir(grad_dir))):
        print("Processing the {}/{} images....".format(ii, len(os.listdir(grad_dir))),end='\r')
        grad_path = os.path.join(grad_dir, grad_name)
        grad = cv2.imread(grad_path, flags=0)
        H,W = grad.shape
        superpoints = np.zeros((H,W))
        # Normalize
        grad = torch.tensor(grad)
        grad = grad-torch.min(grad)
        grad = grad/torch.max(grad) + 0.1
        flat_grad = grad.view(H*W)
        flat_grad = torch.softmax(flat_grad, dim=0).numpy()
        flat_grad = flat_grad/np.sum(flat_grad)
        
        n_seg = round(H*W/900)
        index = np.random.choice(H*W, n_seg, replace=False, p = flat_grad)
        coor_h = [int(idx/W) for idx in index]
        coor_w = [idx%W for idx in index]
        for idx in range(len(index)):
            superpoints[coor_h[idx]][coor_w[idx]] = 255
        save_path = os.path.join(output_dir, grad_name)
        cv2.imwrite(save_path, superpoints)

if __name__ =="__main__":
    img_dir = './data/NeurIPS2022_CellSeg/images'
    output_dir = './data/NeurIPS2022_CellSeg/grads'
    img_dir = "/home/zby/WSCellseg/data/NeurIPS2022_CellSeg/temp/images"
    output_dir = '/home/zby/WSCellseg/data/NeurIPS2022_CellSeg/temp/results'
    os.makedirs(output_dir, exist_ok=True)
    gen_grad(img_dir, output_dir)
    
    # grad_dir = './data/NeurIPS2022_CellSeg/grads'
    # output_dir = './data/NeurIPS2022_CellSeg/superpoints'
    # os.makedirs(output_dir,exist_ok=True)
    # gen_prob_mask(grad_dir, output_dir)