import os
import cv2
import json
import math
import numpy as np
import tifffile as tif
from skimage import measure, segmentation

def get_degree_n_distance(pt1,pt2):
    """Caculate the degree and distance between two points

    Args:
        pt1 (_type_): the first point (x1, y1)
        pt2 (_type_): the second point (x2, y2)

    Returns:
        _type_: _description_
    """
    dx = pt2[0] - pt1[0]
    dy = pt2[1] - pt1[1]
    dist = math.sqrt(dx**2 + dy**2)
    if dist == 0:
        deg = 0
    else:
        sin = dy/dist
        cos = dx/dist
        if sin>= 0 and cos>=0:
            rad = math.asin(sin)
        elif sin>=0 and cos<0:
            rad = math.pi - math.asin(sin)
        elif sin<0 and cos<0:
            rad = math.pi - math.asin(sin)
        elif sin<0 and cos>=0:
            rad = 2*math.pi+ math.asin(sin)
        deg = rad*360/(2*math.pi)
    return deg, dist

def gen_dist_label(img_dir, vor_dir, point_dir, output_dir):
    os.makedirs(output_dir,exist_ok=True)
    img_names = os.listdir(img_dir)
    for ii, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        vor_path = os.path.join(vor_dir, img_name.replace('.tif','.png'))
        point_path = os.path.join(point_dir, img_name.replace('.tif', '.json'))
        img = tif.imread(img_path)
        vor = cv2.imread(vor_path,flags=0)
        point_dict = json.load(open(point_path, 'r'))
        fg_points_dict = point_dict['foreground']
        label = np.zeros(img.shape[:2])
        
        for inst_idx, inst_prop in fg_points_dict.items():
            mask = np.zeros(img.shape[:2])
            select_point = inst_prop['select_point']
            bnd_point = inst_prop['boundary_point']
            deg, dist = get_degree_n_distance(select_point, bnd_point)
            mask = cv2.circle(mask, select_point, round(dist), color=1, thickness=-1)
            mask[vor==255] = 0
            mask = measure.label(mask)
            valid_mask = mask[select_point[1], select_point[0]]
            label[mask == valid_mask]= 1
        label[label==1]=255
        save_path = os.path.join(output_dir, img_name.replace('.tif','.png'))
        cv2.imwrite(save_path, label)

    pass

def gen_full_dist_label(full_dir, output_dir):
    full_label_names = os.listdir(full_dir)
    os.makedirs(output_dir, exist_ok=True)
    for ii,full_label_name in enumerate(full_label_names):
        full_label_path = os.path.join(full_dir, full_label_name)
        full_label = cv2.imread(full_label_path, flags=0)
        full_label = full_label*255
        bnd = segmentation.find_boundaries(full_label, mode='outer')
        full_label[bnd==1]=0
        save_path = os.path.join(output_dir, full_label_name)
        cv2.imwrite(save_path, full_label)

if __name__=="__main__":
    # img_dir = './data/MoNuSeg/train/images'
    # vor_dir = './data/MoNuSeg/train/voronois_v5'
    # point_dir = './data/MoNuSeg/train/points_v5_long'
    # output_dir = './vis'
    # gen_dist_label(img_dir, vor_dir, point_dir, output_dir)
    
    full_dir = './data/MoNuSeg/train/labels_v5_fixed/full'
    output_dir = './data/MoNuSeg/train/labels_v5_fixed/full_distmap'
    gen_full_dist_label(full_dir, output_dir)