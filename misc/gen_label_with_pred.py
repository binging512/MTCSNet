import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import tifffile as tif
import json
from skimage import morphology, segmentation, measure, metrics
from skimage.segmentation import watershed
from skimage.transform import rescale
from skimage.color import label2rgb
import time


def get_adjacent_superpixel(superpixel_value_list, superpixel):
    temp = np.zeros(superpixel.shape)
    for superpixel_value in superpixel_value_list:
        temp[superpixel == superpixel_value] = 1
    temp = cv2.morphologyEx(temp,cv2.MORPH_DILATE, kernel=np.ones((3,3)))
    superpixel = superpixel*temp
    adj_superpixel = list(map(int,list(np.unique(superpixel))))
    adj_superpixel.remove(0)
    for superpixel_value in superpixel_value_list:
        if superpixel_value in adj_superpixel:
            adj_superpixel.remove(superpixel_value)
    return adj_superpixel

def gen_label_with_pred_fb_instance(pred_score_dir, superpixel_dir, last_iter_dict_dir, last_iter_split, output_dir, high_thres, low_thres):
    output_label_dir = os.path.join(output_dir, 'labels')
    output_dict_dir = os.path.join(output_dir, 'instance_dicts')
    output_vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_dict_dir, exist_ok=True)
    os.makedirs(output_vis_dir, exist_ok=True)
    last_iter_split_dict = json.load(open(last_iter_split, 'r'))
    score_names = sorted(os.listdir(pred_score_dir))
    for ii, train_item in enumerate(last_iter_split_dict['train']):
        start_time = time.time()
        print('Generating the {}/{} labels...'.format(ii,len(score_names)),end='\r')
        score_name = os.path.basename(train_item['img_path']).split('.')[0]+'.png'
        pred_score_path = os.path.join(pred_score_dir, score_name)
        superpixel_path = os.path.join(superpixel_dir, score_name.replace('.png','.tiff'))
        last_iter_dict_path = os.path.join(last_iter_dict_dir, score_name.replace('.png','.json'))
        
        score = cv2.imread(pred_score_path, flags=0)/255
        superpixel = tif.imread(superpixel_path)
        last_iter_dict = json.load(open(last_iter_dict_path,'r'))
        label = np.ones(score.shape[:2])*255
        
        # get all the superpixel scores
        H,W = score.shape
        score_scatter = torch.zeros((np.max(superpixel)+1, H, W), dtype=torch.float32)
        score_scatter = torch.scatter(score_scatter, dim=0, index=torch.tensor(superpixel).unsqueeze(0), src=torch.tensor(score, dtype=torch.float32).unsqueeze(0))
        score_masks = torch.zeros(score_scatter.shape)
        score_masks[score_scatter>0] = 1
        superpixel_scores = torch.sum(score_scatter, dim=[1,2])/ torch.sum(score_masks, dim=(1,2))
        
        # for background superpixels
        bg_superpixels = torch.zeros(superpixel_scores.shape)
        bg_superpixels[superpixel_scores<0.1] = 1
        bg_indices = torch.argwhere(bg_superpixels).squeeze()
        bg_indices = bg_indices.tolist()
        
        bg_superpixel_list = last_iter_dict['background'].copy()
        adj_bg_superpixel_list = get_adjacent_superpixel(bg_superpixel_list, superpixel)
        for adj_bg_superpixel in adj_bg_superpixel_list:
            if superpixel_scores[adj_bg_superpixel] < low_thres:
                bg_indices.append(adj_bg_superpixel)
        bg_indices = list(set(bg_indices))
        new_bg_indices = []
        for bg_index in bg_indices:
            if last_iter_dict['superpixels'][str(bg_index)] in [-1, 0]:
                last_iter_dict['superpixels'][str(bg_index)] = 0
                new_bg_indices.append(bg_index)
            else:
                instance_idx = last_iter_dict['superpixels'][str(bg_index)]
                if instance_idx in last_iter_dict['labeled_sp']:
                    pass
                else:
                    last_iter_dict['superpixels'][str(bg_index)] = -1
                    last_iter_dict['instances'][str(instance_idx)].remove(bg_index)
        last_iter_dict['background'] = new_bg_indices
        
        # for foreground superpixels
        for instance_idx, inst_superpixel_list in last_iter_dict['instances'].items():
            adj_inst_superpixel_list = get_adjacent_superpixel(inst_superpixel_list, superpixel)
            for adj_inst_superpixel in adj_inst_superpixel_list:
                if superpixel_scores[adj_inst_superpixel] >= high_thres:
                    if last_iter_dict['superpixels'][str(adj_inst_superpixel)] == -1:
                        last_iter_dict['superpixels'][str(adj_inst_superpixel)] = int(instance_idx)
                        last_iter_dict['instances'][instance_idx].append(adj_inst_superpixel)
                    elif last_iter_dict['superpixels'][str(adj_inst_superpixel)] == 0:
                        if adj_inst_superpixel in last_iter_dict['labeled_sp']:
                            pass
                        else:
                            last_iter_dict['superpixels'][str(adj_inst_superpixel)] = -1
                            last_iter_dict['background'].remove(adj_inst_superpixel)
                    else:
                        pass

        # Changing the background superpixel
        for bg_superpixel in last_iter_dict['background']:
            label[superpixel == bg_superpixel] = 0
        
        # Changing the foreground superpixel
        for instance_idx, inst_superpixel_list in last_iter_dict['instances'].items():
            temp = np.zeros(score.shape[:2])
            for inst_superpixel in inst_superpixel_list:
                temp[superpixel==inst_superpixel] = 1
            mask = temp.copy()
            temp = morphology.erosion(temp, morphology.disk(1))
            # temp = cv2.morphologyEx(temp, cv2.MORPH_ERODE, np.ones((3,3),np.uint8))
            label = np.array(label*(1-mask) + temp*mask, dtype=np.uint8)
            
        save_path = os.path.join(output_label_dir, score_name)
        save_json_path = os.path.join(output_dict_dir, score_name.replace('.png', '.json'))
        cv2.imwrite(save_path, label)
        json.dump(last_iter_dict, open(save_json_path,'w'), indent=2)
        
        vis = np.zeros(label.shape)
        vis[label == 1] = 255
        vis[label == 0] = 0
        vis[label == 255] =128
        vis_path = os.path.join(output_vis_dir, score_name)
        cv2.imwrite(vis_path, vis)
        
        train_item['weak_label_path'] = save_path
        stop_time = time.time()
        print(stop_time-start_time)
    split_save_path = os.path.join(output_dir, 'next_iter.json')
    json.dump(last_iter_split_dict, open(split_save_path, 'w'), indent=2)
    pass

if __name__=="__main__":
    last_iter_dict_dir = './data/MoNuSeg/train/labels/weak_inst'
    superpixel_dir = './data/MoNuSeg/train/superpixels'
    pred_score_dir = './workspace/Mo_Mo_unet50_ep100_b4_crp512_iter0_newlabel/results_test/score'
    last_iter_split = './data/splits/train_Mo_val_Mo_iter0.json'
    output_dir = './workspace/Mo_Mo_unet50_ep100_b4_crp512_iter0_newlabel/label_next_iter_.8_.2'
    high_thres = 0.8
    low_thres = 0.2
    gen_label_with_pred_fb_instance(pred_score_dir, superpixel_dir, last_iter_dict_dir, last_iter_split, output_dir, high_thres, low_thres)