import os
import json
import cv2
from skimage import segmentation, morphology
import random
import numpy as np

def add_boundary_point(point_dict, semantic):
    semantic[semantic==128] = 1
    semantic[semantic==255] = 0
    fg_point_dict= point_dict['foreground']
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(semantic.astype(np.uint8),connectivity=8)
    H,W = semantic.shape
    # Find the superpixel the foreground points in
    for p_idx, p_prop in fg_point_dict.items():
        select_point = p_prop['select_point']
        sp_idx = labels[select_point[1], select_point[0]]
        assert sp_idx>0
        
        # get a boundary point
        x,y,w,h,s = stats[sp_idx]
        cell = np.zeros(labels.shape[:2])
        cell[labels==sp_idx] = 1
        boundary = segmentation.find_boundaries(cell.astype(np.int8), mode='inner')
        boundary = morphology.dilation(boundary, morphology.disk(0))
        flag = 0
        x_min = max(0, x - 1)
        x_max = min(W-1, x + w + 1)
        y_min = max(0, y - 1)
        y_max = min(H-1, y + h + 1)
        while flag==0:
            bnd_x = random.randint(x_min,x_max)
            bnd_y = random.randint(y_min,y_max)
            if boundary[bnd_y, bnd_x] == 1:
                bnd_point = [bnd_x, bnd_y]
                flag = 1
        point_dict['foreground'][str(p_idx)]['boundary_point'] = bnd_point
    return point_dict

def fix_background_point(point_dict, semantic):
    semantic[semantic==128] = 1
    semantic[semantic==255] = 0
    H,W = semantic.shape
    semantic = morphology.dilation(semantic, morphology.disk(3))
    bg_point_dict = point_dict['background']
    bg_num = len(list(bg_point_dict.keys()))
    background_points= []
    while len(background_points)<bg_num:
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        if semantic[y][x] == 0:
            background_points.append([x,y])
    for ii, point in enumerate(background_points):
        point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                'y':int(point[1])}
    return point_dict

def fix_point(point_dir, semantic_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    new_point_dict = {'background':{}, 'foreground':{}}
    for ii, point_dict_name in enumerate(os.listdir(point_dir)):
        print("Fixing the {}/{} point dicts...".format(ii, len(os.listdir(point_dir))), end='\r')
        point_dict_path = os.path.join(point_dir, point_dict_name)
        semantic_path = os.path.join(semantic_dir, point_dict_name.replace('.json','.png'))
        point_dict = json.load(open(point_dict_path, 'r'))
        semantic = cv2.imread(semantic_path, flags=0)
        # Add the boundary point
        # new_point_dict = add_boundary_point(point_dict, semantic)
        new_point_dict = fix_background_point(point_dict, semantic.copy())
        save_path = os.path.join(output_dir, point_dict_name)
        json.dump(new_point_dict, open(save_path, 'w'), indent=2)
    pass

if __name__ == "__main__":
    point_dir = './data/MoNuSeg/train/points_v5'
    semantic_dir = './data/MoNuSeg/train/semantics'
    output_dir = './data/MoNuSeg/train/points_v5_fixed'
    fix_point(point_dir, semantic_dir, output_dir)