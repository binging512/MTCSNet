import cv2
import tifffile as tif
import numpy as np
import json
import os

def vis_annotated_points(img_dir, point_dir, semantic_dir, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    img_names = os.listdir(img_dir)
    for ii, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        point_path = os.path.join(point_dir, img_name.replace('.tif','.json'))
        semantic_path = os.path.join(semantic_dir, img_name.replace('.tif','.png'))
        img = tif.imread(img_path)
        point_dict = json.load(open(point_path,'r'))
        semantic = cv2.imread(semantic_path)
        bg_point_dict = point_dict['background']
        fg_point_dict = point_dict['foreground']
        for point_idx, point_prop in bg_point_dict.items():
            point_x = point_prop['x']
            point_y = point_prop['y']
            img = cv2.circle(img, (point_x, point_y), radius=2, color=(255,0,0), thickness=1)
            semantic = cv2.circle(semantic, (point_x, point_y), radius=2, color=(255,0,0), thickness=1)
            
        for point_idx, point_prop in fg_point_dict.items():
            select_point = point_prop['select_point']
            bnd_point = point_prop['boundary_point']
            img = cv2.circle(img, select_point, radius=2, color=(0, 255, 0), thickness=1)
            semantic = cv2.circle(semantic, select_point, radius=2, color=(0, 255, 0), thickness=1)
            img = cv2.circle(img, bnd_point, radius=2, color=(0, 0, 255), thickness=1)
            semantic = cv2.circle(semantic, bnd_point, radius=2, color=(0, 0, 255), thickness=1)
        
        vis = np.hstack((img,semantic))
        save_path = os.path.join(output_dir, img_name.replace('.tif', '.png'))
        cv2.imwrite(save_path, vis)
            
    return

if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/images'
    point_dir = './data/MoNuSeg/train/points_v5_long_fuse'
    semantic_dir = './data/MoNuSeg/train/semantics_v5'
    output_dir = './data/MoNuSeg/train/vis_point_long_fuse'
    vis_annotated_points(img_dir, point_dir, semantic_dir, output_dir)