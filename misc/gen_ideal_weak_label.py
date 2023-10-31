import os
import json
import cv2
import numpy as np
import tifffile as tif

def gen_ideal_weak_label(full_label_dir, full_heat_dir, point_dict_dir, superpixel_dir, output_dir):
    ideal_label_dir = os.path.join(output_dir, 'ideal')
    ideal_heat_dir = os.path.join(output_dir,'ideal_heatmap')
    os.makedirs(ideal_label_dir, exist_ok=True)
    os.makedirs(ideal_heat_dir, exist_ok=True)
    label_names = sorted(os.listdir(full_label_dir))
    for ii, label_name in enumerate(label_names):
        print("Generating the {}/{} ideal labels...".format(ii, len(label_names)),end='\r')
        full_label_path = os.path.join(full_label_dir, label_name)
        full_label = cv2.imread(full_label_path, flags=0)
        full_heat_path = os.path.join(full_heat_dir, label_name)
        full_heat = cv2.imread(full_heat_path, flags=0)
        superpixel_path = os.path.join(superpixel_dir, label_name.replace('.png','.tif'))
        superpixel = tif.imread(superpixel_path)
        point_dict_path = os.path.join(point_dict_dir, label_name.replace('.png', '.json'))
        point_dict = json.load(open(point_dict_path, 'r'))

        mask = np.zeros(superpixel.shape)
        ignore = np.ones(superpixel.shape)*255
        fg_point_dict = point_dict['foreground']
        bg_point_dict = point_dict['background']
        for inst_idx, inst_prop in fg_point_dict.items():
            select_point = inst_prop['select_point']
            superpixel_value = superpixel[select_point[1], select_point[0]]
            mask[superpixel==superpixel_value] = 1
        for p_idx, p_prop in bg_point_dict.items():
            x = p_prop['x']
            y = p_prop['y']
            superpixel_value = superpixel[y,x]
            mask[superpixel==superpixel_value] = 1
        
        ideal_label = full_label*mask+ ignore*(1-mask)
        ideal_heat = full_heat*mask
        label_save_path =os.path.join(ideal_label_dir, label_name)
        heat_save_path = os.path.join(ideal_heat_dir, label_name)
        cv2.imwrite(label_save_path, ideal_label)
        cv2.imwrite(heat_save_path, ideal_heat)
    
if __name__=="__main__":
    full_label_dir = './data/MoNuSeg/train/labels_v5/full'
    full_heat_dir = './data/MoNuSeg/train/labels_v5/full_heatmap'
    point_dict_dir = './data/MoNuSeg/train/points_v5'
    superpixel_dir = './data/MoNuSeg/train/superpixels_v5'
    output_dir = './data/MoNuSeg/train/labels_v5'
    gen_ideal_weak_label(full_label_dir, full_heat_dir, point_dict_dir, superpixel_dir, output_dir)