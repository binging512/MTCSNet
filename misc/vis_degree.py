import pickle
import os
import cv2
import json
import tifffile as tif
import numpy as np
from skimage import segmentation, morphology
import glob
import math

def vis_degree(image_dir, semantic_dir, degree_dir, score_dir, split_path, output_dir):
    split_info = json.load(open(split_path,'r'))
    train_list = [os.path.basename(train_item['img_path']) for train_item in split_info['train']]
    for ii, train_item in enumerate(train_list):
        print("Visualing the {}/{} degree images...".format(ii, len(train_list)), end='\r')
        image_path = os.path.join(image_dir,train_item)
        semantic_path = os.path.join(semantic_dir,train_item.replace('.tif', '.png'))
        score_path = os.path.join(score_dir, train_item.replace('.tif','.png'))
        degree_paths = list(glob.glob(os.path.join(degree_dir, train_item.replace('.tif','_deg*.pkl')), recursive=True))
        img = tif.imread(image_path)
        semantic = cv2.imread(semantic_path)
        score = cv2.imread(score_path, flags=0)/255
        seed = np.zeros(score.shape).astype(np.uint8)
        seed[score>=0.7] = 1
        num_labels, labels, stats, centroids= cv2.connectedComponentsWithStats(seed)
        for degree_path in degree_paths:
            degree_res = pickle.load(open(degree_path, 'rb'))
            degree = int(degree_path.split('deg')[-1].replace('.pkl',''))
            for centroid in centroids:
                centroid = [round(centroid[0]), round(centroid[1])]
                dist = degree_res[centroid[1], centroid[0]]
                dx = round(math.cos(math.radians(degree))*dist)
                dy = round(math.sin(math.radians(degree))*dist)
                dest_point = [centroid[0]+dx, centroid[1]+dy]
                img = cv2.circle(img, centroid, radius=1, color=[255, 0, 0], thickness=-1)
                img = cv2.circle(img, dest_point, radius=1, color=[0, 255, 0], thickness=-1)
                img = cv2.arrowedLine(img, centroid, dest_point, color=[0, 0, 255], thickness=1, tipLength=0.3)
                semantic = cv2.circle(semantic, centroid, radius=1, color=[255, 0, 0], thickness=-1)
                semantic = cv2.circle(semantic, dest_point, radius=1, color=[0, 255, 0], thickness=-1)
                semantic = cv2.arrowedLine(semantic, centroid, dest_point, color=[0, 0, 255], thickness=1, tipLength=0.3)
            degree_vis = degree_res/np.max(degree_res)*255
            cv2.imwrite(os.path.join(output_dir, os.path.basename(degree_path).replace('.pkl', '.png')), degree_vis)
        vis = np.hstack((img, semantic))
        save_path = os.path.join(output_dir, train_item.replace('.tif','.png'))
        cv2.imwrite(save_path, vis)
    pass

def vis_dist(image_dir, weak_distmap_dir):
    img_name = 'TCGA-18-5592-01Z-00-DX1.png'


if __name__=="__main__":
    image_dir = './data/MoNuSeg/train/images'
    semantic_dir = './data/MoNuSeg/train/semantics_v3'
    degree_dir = './workspace/MoNuSeg_ablation/Mo_Mo_unet50rvdc_cls2_1head_ep500_b4_crp512_iter0_cn/results_test/deg'
    score_dir = './workspace/MoNuSeg_ablation/Mo_Mo_unet50rvdc_cls2_1head_ep500_b4_crp512_iter0_cn/results_test/score'
    split_path = './data/splits/train_Mo_val_Mo_iter0_v5.json'
    output_dir = './workspace/MoNuSeg_ablation/Mo_Mo_unet50rvdc_cls2_1head_ep500_b4_crp512_iter0_cn/results_test/deg_vis'
    os.makedirs(output_dir, exist_ok=True)
    vis_degree(image_dir, semantic_dir, degree_dir, score_dir, split_path, output_dir)
    
    image_dir = './data/MoNuSeg/train/vis_point_long_fuse/'
    