import os
import cv2
import numpy as np
import math
import json
import tifffile as tif
from skimage import segmentation

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

def change_bnd_points(point_dir, gt_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    point_names = os.listdir(point_dir)
    for ii,point_name in enumerate(point_names):
        point_path = os.path.join(point_dir, point_name)
        gt_path = os.path.join(gt_dir, point_name.replace('.json','.tif'))
        
        point_dict = json.load(open(point_path))
        gt = tif.imread(gt_path)
        fg_point_dict = point_dict['foreground']
        bg_point_dict = point_dict['background']
        new_point_dict = {'foreground':{}, 'background':bg_point_dict}
        for point_idx, point_prop in fg_point_dict.items():
            select_point = point_prop['select_point']
            cell_value = gt[select_point[1], select_point[0]]
            mask = np.zeros(gt.shape)
            mask[gt==cell_value] = 1
            bnd = segmentation.find_boundaries(mask, mode='inner')
            bnd_points = np.array(np.where(bnd==1)).T
            dist_list = []
            for bnd_point in bnd_points:
                deg, dist = get_degree_n_distance(select_point, [bnd_point[1],bnd_point[0]])
                dist_list.append(dist)
            dist_max_idx = np.argmax(np.array(dist_list))
            select_bnd_point = [int(bnd_points[dist_max_idx][1]),int(bnd_points[dist_max_idx][0])]
            point_prop['boundary_point'] = select_bnd_point
            new_point_dict['foreground'][point_idx] = point_prop
        save_path = os.path.join(output_dir, point_name)
        json.dump(new_point_dict, open(save_path,'w'), indent=2)
    pass

if __name__=="__main__":
    point_dir = './data/MoNuSeg/train/points_v5'
    gt_dir = './data/MoNuSeg/train/gts'
    output_dir = './data/MoNuSeg/train/points_v5_long/'
    change_bnd_points(point_dir, gt_dir, output_dir)
