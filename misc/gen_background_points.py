import os
import cv2
import json
import numpy as np
import random
import tifffile as tif
import torch
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from skimage import morphology, segmentation, measure

def gen_background_points_w_otsu(img_dir, point_dir, output_dir):
    gray_otsu_dir = os.path.join(output_dir, 'gray_otsu')
    # kmeans_dir = os.path.join(output_dir,'kmeans')
    output_point_dir = os.path.join(output_dir, 'points_v5_long_otsu')
    os.makedirs(gray_otsu_dir, exist_ok=True)
    os.makedirs(output_point_dir, exist_ok=True)
    # os.makedirs(kmeans_dir, exist_ok=True)
    img_names = os.listdir(img_dir)
    
    for ii, img_name in enumerate(img_names):
        img_path = os.path.join(img_dir, img_name)
        point_path = os.path.join(point_dir, img_name.replace('.tif','.json'))
        img =tif.imread(img_path)
        H,W,C = img.shape
        point_dict = json.load(open(point_path, 'r'))
        new_point_dict = {'foreground': point_dict['foreground'], 'background':{}}
        bg_point_num = len(list(point_dict['background'].keys()))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        otsu = morphology.opening(otsu, morphology.disk(3)) # delete the small regions
        otsu = morphology.closing(otsu, morphology.disk(3)) # fill the small gaps
        label = measure.label(otsu)
        fg_point_dict = point_dict['foreground']
        
        mask = np.zeros(img.shape[:2])
        for point_idx, point_prop in fg_point_dict.items():
            select_point = point_prop['select_point']
            label_value = label[select_point[1], select_point[0]]
            if label_value==0:
                pass
            else:
                mask[label==label_value] = 1
                
        background_points= []
        while len(background_points)<bg_point_num:
            x = random.randint(0, W-1)
            y = random.randint(0, H-1)
            if mask[y][x] == 0:
                background_points.append([x,y])
                
        for ii, point in enumerate(background_points):
            new_point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                       'y':int(point[1])}
        
        save_point_path = os.path.join(output_point_dir, img_name.replace('.tif','.json'))
        json.dump(new_point_dict, open(save_point_path, 'w'), indent=2)

        # vis
        otsu_vis = np.hstack((img, np.stack((otsu,otsu,otsu),axis=-1)))
        otsu_vis_save_path = os.path.join(gray_otsu_dir, img_name.replace('.tif','.png'))
        cv2.imwrite(otsu_vis_save_path, otsu_vis)
        
    return


def gen_background_points_w_vor(vor_dir, point_dir, output_dir):
    output_point_dir = os.path.join(output_dir, 'points_v5_long_vor')
    os.makedirs(output_point_dir, exist_ok=True)
    
    vor_names = os.listdir(vor_dir)
    for ii, vor_name in enumerate(vor_names):
        vor_path = os.path.join(vor_dir, vor_name)
        vor = cv2.imread(vor_path, flags=0)
        H,W = vor.shape
        point_path = os.path.join(point_dir, vor_name.replace('.png','.json'))
        point_dict = json.load(open(point_path, 'r'))
        new_point_dict = {'foreground':point_dict['foreground'], 'background':{}}
        
        bg_point_num = len(list(point_dict['background'].keys()))
        background_points = []
        while len(background_points)<bg_point_num:
            x = random.randint(0, W-1)
            y = random.randint(0, H-1)
            if vor[y][x] == 0:
                background_points.append([x,y])
                
        for ii, point in enumerate(background_points):
            new_point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                       'y':int(point[1])}
            
        save_point_path = os.path.join(output_point_dir, vor_name.replace('.png','.json'))
        json.dump(new_point_dict, open(save_point_path, 'w'), indent=2)
    return


def gen_background_points_w_kmeans(img_dir, point_dir, semantic_dir, vor_dir, output_dir):
    kmeans_dir = os.path.join(output_dir, 'fuse')
    output_point_dir = os.path.join(output_dir, 'points_v5_long_fuse')
    os.makedirs(kmeans_dir, exist_ok=True)
    os.makedirs(output_point_dir, exist_ok=True)
    img_names = os.listdir(img_dir)
    for ii, img_name in enumerate(sorted(img_names)):
        img_path = os.path.join(img_dir, img_name)
        point_path = os.path.join(point_dir, img_name.replace('.tif','.json'))
        semantic_path = os.path.join(semantic_dir, img_name.replace('.tif','.png'))
        vor_path = os.path.join(vor_dir, img_name.replace('.tif', '.png'))
        img =tif.imread(img_path)
        H,W,C = img.shape
        point_dict = json.load(open(point_path, 'r'))
        semantic = cv2.imread(semantic_path, flags=0)
        vor = cv2.imread(vor_path, flags=0)

        pixels = img.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.1)
        K = 10
        cluster_list = [0,0,0,0,0,0,0,0,0,0]
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        labels = labels.reshape((H,W))
        for point_idx, point_prop in point_dict['foreground'].items():
            select_point = point_prop['select_point']
            cluster_idx = labels[select_point[1], select_point[0]]
            cluster_list[cluster_idx] += 1
        cluster_sorted_idx = np.argsort(np.array(cluster_list))
        
        bg_cluster = [cluster_sorted_idx[i] for i in range(5)]
        mask = np.ones((H,W))
        for bg_idx in bg_cluster:
            mask[labels==bg_idx] = 0
        
        mask = morphology.opening(mask, morphology.disk(3))
        mask = morphology.closing(mask, morphology.disk(3))
        mask = morphology.erosion(mask, morphology.disk(3))
        vor[vor==255]=1
        mask = mask*vor
        ori_bg_dict = {'in_mask':[],'out_mask':[]}
        for point_idx, point_prop in point_dict['background'].items():
            point_x = point_prop['x']
            point_y = point_prop['y']
            ori_bg_value = mask[point_y, point_x]
            if ori_bg_value == 1:
                ori_bg_dict['in_mask'].append(point_prop)
            else:
                ori_bg_dict['out_mask'].append(point_prop)
        print('Origin points: In mask point:{}. Out of mask point:{}'.format(len(ori_bg_dict['in_mask']), len(ori_bg_dict['out_mask'])))
        
        mask = mask*255
        vis = np.hstack((img, np.stack((mask,mask,mask), axis=-1)))
        vis_save_path = os.path.join(kmeans_dir,img_name.replace('.tif','.png'))
        cv2.imwrite(vis_save_path, vis)
        
        new_point_dict = {'foreground':point_dict['foreground'], 'background':{}}
        bg_point_num = len(list(point_dict['background'].keys()))
        background_points = []
        new_bg_dict = {'in_mask':[], 'out_mask':[]}
        semantic[semantic==128] = 1
        while len(background_points)<bg_point_num:
            x = random.randint(0, W-1)
            y = random.randint(0, H-1)
            if mask[y][x] == 0:
                background_points.append([x,y])
                if semantic[y][x] == 1:
                    new_bg_dict['in_mask'].append([x,y])
                else:
                    new_bg_dict['out_mask'].append([x,y])
        print("New points: In mask point:{}. Out of mask point:{}".format(len(new_bg_dict['in_mask']),len(new_bg_dict['out_mask'])))
        
        for ii, point in enumerate(background_points):
            new_point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                       'y':int(point[1])}
            
        save_point_path = os.path.join(output_point_dir, img_name.replace('.tif','.json'))
        json.dump(new_point_dict, open(save_point_path, 'w'), indent=2)
            
    pass
    
if __name__=="__main__":
    # img_dir = './data/MoNuSeg/train/images'
    # point_dir = './data/MoNuSeg/train/points_v5_long'
    # output_dir = './data/MoNuSeg/train/'
    # gen_background_points_w_otsu(img_dir, point_dir, output_dir)
    
    # vor_dir = './data/MoNuSeg/train/labels_v5_fixed/weak_distmap'
    # point_dir = './data/MoNuSeg/train/points_v5_long'
    # output_dir = './data/MoNuSeg/train/'
    # gen_background_points_w_vor(vor_dir, point_dir, output_dir)
    
    img_dir = './data/MoNuSeg/train/images'
    point_dir = './data/MoNuSeg/train/points_v5_long'
    semantic_dir = './data/MoNuSeg/train/semantics_v5'
    vor_dir = './data/MoNuSeg/train/labels_v5_long_kmeans/weak_distmap'
    output_dir = './data/MoNuSeg/train/'
    gen_background_points_w_kmeans(img_dir, point_dir, semantic_dir, vor_dir, output_dir)