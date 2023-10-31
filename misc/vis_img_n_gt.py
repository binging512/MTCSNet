import os
import cv2
import tifffile as tif
from skimage.color import label2rgb

import numpy as np

def vis_img_n_gt(img_dir, gt_dir, output_dir):
    img_names = os.listdir(img_dir)
    os.makedirs(output_dir,exist_ok=True)
    for ii,img_name in enumerate(img_names):
        img_path = os.path.join(img_dir,img_name)
        gt_path = os.path.join(gt_dir,img_name.replace('.tif','.tiff').replace('.png','.tiff'))
        
        img = cv2.imread(img_path)
        gt = tif.imread(gt_path).astype(np.int32)
        print(np.max(gt))
        gt = label2rgb(gt, bg_label=0)*255
        
        save_img_path=os.path.join(output_dir, img_name.replace('.tif','.png'))
        save_gt_path=os.path.join(output_dir, img_name.replace('.tif','_gt.png'))
        cv2.imwrite(save_img_path, img)
        cv2.imwrite(save_gt_path, gt)
    pass

if __name__=="__main__":
    img_dir='./data/MoNuSeg/images'
    gt_dir = './data/MoNuSeg/gts'
    output_dir = './data/MoNuSeg/vis_inst'
    vis_img_n_gt(img_dir, gt_dir,output_dir)