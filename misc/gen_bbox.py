import tifffile as tif
import json
import cv2
import os
import numpy as np

def gen_bbox(img_dir, gt_dir, output_dir):
    img_names = sorted(os.listdir(img_dir))
    for ii, img_name in enumerate(img_names):
        print("Processing the {}/{} images...".format(ii,len(img_names)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name.replace('.tif','.tiff'))
        gt = tif.imread(gt_path).astype(np.int32)
        H,W = gt.shape
        bbox_dict = {}
        cell_num = np.max(gt)
        for cell_idx in range(1, cell_num+1):
            
            cell_mask = np.zeros((H,W)).astype(np.uint8)
            cell_mask[gt == cell_idx] =1
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cell_mask)
            try:
                x,y,w,h,s = stats[1]
            except:
                print(img_name)
                print(cell_idx)
                print(stats)
            bbox_dict[int(cell_idx)] = {'pt1':[int(x),int(y)], 'pt2':[int(x+w),int(y+h)]}
        save_path = os.path.join(output_dir,img_name.replace('.tif','.json'))
        json.dump(bbox_dict, open(save_path,'w'), indent=2)
    
    
if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/images'
    gt_dir = './data/MoNuSeg/train/gts'
    output_dir = './data/MoNuSeg/train/bboxes_v5'
    os.makedirs(output_dir,exist_ok=True)
    gen_bbox(img_dir, gt_dir, output_dir)
