import json
import tifffile as tif
import os
import cv2
import numpy as np
import random
from skimage import io, segmentation, morphology, exposure

def create_interior_map(inst_map):
    """
    Parameters
    ----------
    inst_map : (H,W), np.int16
        DESCRIPTION.
    Returns
    -------
    interior : (H,W), np.uint8 
        three-class map, values: 0,1,2
        0: background
        1: interior
        2: boundary
    """
    # create interior-edge map
    boundary = segmentation.find_boundaries(inst_map, mode='inner')
    boundary = morphology.binary_dilation(boundary, morphology.disk(1))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 2
    return interior

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

def gen_label_with_point_fb(img_dir, point_dir, superpixel_dir, output_dir):
    output_label_dir = os.path.join(output_dir, 'simple')
    output_heat_dir = os.path.join(output_dir, 'simple_weak')
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_heat_dir, exist_ok=True)
    img_names = sorted(os.listdir(img_dir))
    for ii, img_name in enumerate(img_names):
        print('Generating the {}/{} labels...'.format(ii,len(img_names)),end='\r')
        img_path = os.path.join(img_dir,img_name)
        point_path = os.path.join(point_dir,img_name.replace('.tif','.json'))
        superpixel_path = os.path.join(superpixel_dir,img_name.replace('.tif','.tif'))
        img = tif.imread(img_path)
        point_dict = json.load(open(point_path,'r'))
        superpixel = tif.imread(superpixel_path)
        label = np.ones(img.shape[:2])*255
        heat = np.zeros(img.shape[:2])

        fg_points_dict = point_dict['foreground']
        bg_points_dict = point_dict['background']
        # For background
        for point_idx, bg_point_prop in bg_points_dict.items():
            point_x = bg_point_prop['x']
            point_y = bg_point_prop['y']
            superpixel_value = superpixel[point_y, point_x]
            label[superpixel==superpixel_value] = 0
            
        # For foreground
        for point_idx, fg_point_prop in fg_points_dict.items():
            point_x, point_y = fg_point_prop['select_point']
            superpixel_value = superpixel[point_y,point_x]
            label[superpixel==superpixel_value] = 1
        
        heat[label==1] = 1
        heat = cv2.GaussianBlur(heat, (5,5), -1)
        heat = heat/np.max(heat)*255
        save_label_path = os.path.join(output_label_dir,img_name.replace('.tif','.png'))
        save_heat_path = os.path.join(output_heat_dir, img_name.replace('.tif','.png'))
        cv2.imwrite(save_label_path, label)
        cv2.imwrite(save_heat_path, heat)
    pass

def gen_label_with_pred_fb(img_dir, pred_score_dir, superpixel_dir, last_iter_dir, output_dir):
    img_names = sorted(os.listdir(img_dir))
    for ii, img_name in enumerate(img_names):
        print('Generating the {}/{} labels...'.format(ii,len(img_names)),end='\r')
        img_path = os.path.join(img_dir,img_name)
        pred_score_path = os.path.join(pred_score_dir, img_name)
        superpixel_path = os.path.join(superpixel_dir, img_name.replace('.png','.tiff'))
        last_iter_path = os.path.join(last_iter_dir, img_name)
        img = cv2.imread(img_path)
        score = cv2.imread(pred_score_path, flags=0)/255
        superpixel = tif.imread(superpixel_path)
        last_label = cv2.imread(last_iter_path, flags=0)
        label = np.ones(img.shape[:2])*255
        
        # get the adjacent superpixels
        fg_superpixel = np.zeros(img.shape[:2])
        bg_superpixel = np.zeros(img.shape[:2])
        fg_superpixel[last_label==1] = 1
        bg_superpixel[last_label==0] = 1
        fg_superpixel_list = list(map(int,list(np.unique(superpixel*fg_superpixel))))
        if 0 in fg_superpixel_list:
            fg_superpixel_list.remove(0)
        bg_superpixel_list = list(map(int,list(np.unique(superpixel*bg_superpixel))))
        if 0 in bg_superpixel_list:
            bg_superpixel_list.remove(0)
        adj_fg_superpixel_list = get_adjacent_superpixel(fg_superpixel_list, superpixel)
        adj_bg_superpixel_list = get_adjacent_superpixel(bg_superpixel_list, superpixel)
        
        # get all the superpixel scores
        superpixel_score_dict = {}
        for sp in range(1, np.max(superpixel)+1):
            temp = np.zeros(img.shape[:2])
            temp[superpixel==sp] = 1
            superpixel_score = np.sum(score*temp)/np.sum(temp)
            superpixel_score_dict[str(sp)] = superpixel_score
        
        # for background superpixels
        change_bg_superpixel_list = []
        for adj_bg_superpixel in adj_bg_superpixel_list:
            if superpixel_score_dict[str(adj_bg_superpixel)] <= 0.3:
                change_bg_superpixel_list.append(adj_bg_superpixel)
        
        # for foreground superpixels
        change_fg_superpixel_list = []
        for adj_fg_superpixel in adj_fg_superpixel_list:
            # check if the adjacent pixel is adjacent to two individual instances
            _, instance_mask, stats, centroid = cv2.connectedComponentsWithStats(fg_superpixel.astype(np.uint8))
            adj_2_superpxiels = get_adjacent_superpixel([adj_fg_superpixel],superpixel)
            check_mask = np.zeros(img.shape[:2])
            for adj in adj_2_superpxiels:
                check_mask[superpixel == adj] = 1
            adj_instances = list(map(int,list(np.unique(instance_mask*check_mask))))
            adj_instances.remove(0)
            if len(adj_instances) >= 2:
                change_bg_superpixel_list.append(adj_fg_superpixel)
            else:
                if superpixel_score_dict[str(adj_fg_superpixel)] >= 0.7:
                    change_fg_superpixel_list.append(adj_fg_superpixel)
        
        # Changing the background superpixel
        for change_bg_superpixel in change_bg_superpixel_list:
            bg_superpixel[superpixel==change_bg_superpixel] = 1
        # Changing the foreground superpixel
        for change_fg_superpixel in change_fg_superpixel_list:
            fg_superpixel[superpixel==change_fg_superpixel] = 1
        label[bg_superpixel==1] = 0
        label[fg_superpixel==1] = 1
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, label)
    pass

def gen_label_with_pred_fb_ms(img_dir, pred_score_dir, superpixel_dir, last_iter_dir, output_dir, steps = 1):
    img_names = sorted(os.listdir(img_dir))
    for ii, img_name in enumerate(img_names):
        print('Generating the {}/{} labels...'.format(ii,len(img_names)),end='\r')
        img_path = os.path.join(img_dir,img_name)
        pred_score_path = os.path.join(pred_score_dir, img_name)
        superpixel_path = os.path.join(superpixel_dir, img_name.replace('.png','.tiff'))
        last_iter_path = os.path.join(last_iter_dir, img_name)
        img = cv2.imread(img_path)
        score = cv2.imread(pred_score_path, flags=0)/255
        superpixel = tif.imread(superpixel_path)
        last_label = cv2.imread(last_iter_path, flags=0)
        label = np.ones(img.shape[:2])*255
        
        # get the adjacent superpixels
        fg_superpixel = np.zeros(img.shape[:2])
        bg_superpixel = np.zeros(img.shape[:2])
        fg_superpixel[last_label==1] = 1
        bg_superpixel[last_label==0] = 1
        fg_superpixel_list = list(map(int,list(np.unique(superpixel*fg_superpixel))))
        if 0 in fg_superpixel_list:
            fg_superpixel_list.remove(0)
        bg_superpixel_list = list(map(int,list(np.unique(superpixel*bg_superpixel))))
        if 0 in bg_superpixel_list:
            bg_superpixel_list.remove(0)
        adj_fg_superpixel_list = get_adjacent_superpixel(fg_superpixel_list, superpixel)
        adj_bg_superpixel_list = get_adjacent_superpixel(bg_superpixel_list, superpixel)
        for step in range(steps):
            # for background superpixels
            change_bg_superpixel_list = []
            for adj_bg_superpixel in adj_bg_superpixel_list:
                temp = np.zeros(img.shape[:2])
                temp[superpixel==adj_bg_superpixel] = 1
                superpixel_score = np.sum(score*temp)/np.sum(temp)
                if superpixel_score <= 0.3:
                    change_bg_superpixel_list.append(adj_bg_superpixel)
            
            # for foreground superpixels
            change_fg_superpixel_list = []
            for adj_fg_superpixel in adj_fg_superpixel_list:
                temp = np.zeros(img.shape[:2])
                temp[superpixel==adj_fg_superpixel] = 1
                superpixel_score = np.sum(score*temp)/np.sum(temp)
                # check if the adjacent pixel is adjacent to two individual instances
                _, instance_mask, stats, centroid = cv2.connectedComponentsWithStats(fg_superpixel.astype(np.uint8))
                adj_2_superpxiels = get_adjacent_superpixel([adj_fg_superpixel],superpixel)
                check_mask = np.zeros(img.shape[:2])
                for adj in adj_2_superpxiels:
                    check_mask[superpixel == adj] = 1
                adj_instances = list(map(int,list(np.unique(instance_mask*check_mask))))
                adj_instances.remove(0)
                if len(adj_instances) >= 2:
                    change_bg_superpixel_list.append(adj_fg_superpixel)
                else:
                    if superpixel_score >= 0.7:
                        change_fg_superpixel_list.append(adj_fg_superpixel)
            
            # Changing the background superpixel
            for change_bg_superpixel in change_bg_superpixel_list:
                bg_superpixel[superpixel==change_bg_superpixel] = 1
            # Changing the foreground superpixel
            for change_fg_superpixel in change_fg_superpixel_list:
                fg_superpixel[superpixel==change_fg_superpixel] = 1
            label[bg_superpixel==1] = 0
            label[fg_superpixel==1] = 1
            
            # Reset the superpixel for loop
            fg_superpixel = np.zeros(img.shape[:2])
            bg_superpixel = np.zeros(img.shape[:2])
            fg_superpixel[label==1] = 1
            bg_superpixel[label==0] = 1
            fg_superpixel_list = list(map(int,list(np.unique(superpixel*fg_superpixel))))
            if 0 in fg_superpixel_list:
                fg_superpixel_list.remove(0)
            bg_superpixel_list = list(map(int,list(np.unique(superpixel*bg_superpixel))))
            if 0 in bg_superpixel_list:
                bg_superpixel_list.remove(0)
            adj_fg_superpixel_list = get_adjacent_superpixel(fg_superpixel_list, superpixel)
            adj_bg_superpixel_list = get_adjacent_superpixel(bg_superpixel_list, superpixel)
            
        save_path = os.path.join(output_dir, img_name)
        cv2.imwrite(save_path, label)
    pass

def gen_label_with_pred_fb_instance(img_dir, pred_score_dir, superpixel_dir, last_iter_dict_dir, output_dir, output_dict_dir):
    img_names = sorted(os.listdir(img_dir))
    for ii, img_name in enumerate(img_names):
        print('Generating the {}/{} labels...'.format(ii,len(img_names)),end='\r')
        img_path = os.path.join(img_dir,img_name)
        pred_score_path = os.path.join(pred_score_dir, img_name)
        superpixel_path = os.path.join(superpixel_dir, img_name.replace('.png','.tiff'))
        last_iter_dict_path = os.path.join(last_iter_dict_dir, img_name.replace('.png','.json'))
        img = cv2.imread(img_path)
        score = cv2.imread(pred_score_path, flags=0)/255
        superpixel = tif.imread(superpixel_path)
        last_iter_dict = json.load(open(last_iter_dict_path,'r'))
        label = np.ones(img.shape[:2])*255
        
        # get all the superpixel scores
        bg_superpixel_list = last_iter_dict['background'].copy()
        superpixel_score_dict = {}
        for sp in range(1, np.max(superpixel)+1):
            if last_iter_dict['superpixels'][str(sp)] != -1:
                continue
            temp = np.zeros(img.shape[:2])
            temp[superpixel==sp] = 1
            superpixel_score = np.sum(score*temp)/np.sum(temp)
            superpixel_score_dict[str(sp)] = superpixel_score
            if superpixel_score<=0.1 and sp not in bg_superpixel_list:
                last_iter_dict['superpixels'][str(sp)] = 0
                last_iter_dict['background'].append(sp)
            
        # for background superpixels
        adj_bg_superpixel_list = get_adjacent_superpixel(bg_superpixel_list, superpixel)
        for adj_bg_superpixel in adj_bg_superpixel_list:
            if superpixel_score_dict[str(adj_bg_superpixel)] <= 0.3 and adj_bg_superpixel not in last_iter_dict['background']:
                last_iter_dict['superpixels'][str(adj_bg_superpixel)] = 0
                last_iter_dict['background'].append(adj_bg_superpixel)
        
        # for foreground superpixels
        for instance_idx, inst_superpixel_list in last_iter_dict['instances'].items():
            adj_inst_superpixel_list =  get_adjacent_superpixel(inst_superpixel_list, superpixel)
            for adj_inst_superpixel in adj_inst_superpixel_list:
                if superpixel_score_dict[str(adj_inst_superpixel)] >= 0.7:
                    if last_iter_dict['superpixels'][str(adj_inst_superpixel)] == -1:
                        last_iter_dict['superpixels'][str(adj_inst_superpixel)] = int(instance_idx)
                        last_iter_dict['instances'][instance_idx].append(adj_inst_superpixel)
        
        # Changing the background superpixel
        for bg_superpixel in last_iter_dict['background']:
            label[superpixel == bg_superpixel] = 0
        # Changing the foreground superpixel
        for instance_idx, inst_superpixel_list in last_iter_dict['instances'].items():
            temp = np.zeros(img.shape[:2])
            for inst_superpixel in inst_superpixel_list:
                temp[superpixel==inst_superpixel] = 1
            mask = temp.copy()
            temp = cv2.morphologyEx(temp, cv2.MORPH_ERODE, np.ones((3,3),np.uint8))
            label = np.array(label*(1-mask) + temp*mask, dtype=np.uint8)
        save_path = os.path.join(output_dir, img_name)
        save_json_path = os.path.join(output_dict_dir, img_name.replace('.png', '.json'))
        cv2.imwrite(save_path, label)
        json.dump(last_iter_dict, open(save_json_path,'w'), indent=2)
    pass

def gen_full_supervision_label(img_dir, gt_dir, output_dir):
    for ii, img_name in enumerate(sorted(os.listdir(img_dir))):
        print("Generating the {}/{} labels....".format(ii,len(os.listdir(img_dir))), end='\r')
        gt_path = os.path.join(gt_dir, img_name.replace('.png','_label.tiff'))
        inst_map = tif.imread(gt_path)
        label = create_interior_map(inst_map)
        save_path = os.path.join(output_dir,img_name)
        cv2.imwrite(save_path, label)

if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/images'
    point_dir = './data/MoNuSeg/train/points_v5_long_otsu'
    superpixel_dir = './data/MoNuSeg/train/superpixels_v5'
    output_dir = './data/MoNuSeg/train/labels_v5_long_otsu/'
    gen_label_with_point_fb(img_dir, point_dir, superpixel_dir, output_dir)

    # img_dir = './data/NeurIPS2022_CellSeg/images'
    # last_iter_dir = './data/NeurIPS2022_CellSeg/labels/iter0_adaptive'
    # superpixel_dir = './data/NeurIPS2022_CellSeg/superpixel_adaptive'
    # pred_score_dir = './workspace/cellseg_unet50_ep100_b64_crp512_iter0/results_test/score'
    # output_dir = './workspace/cellseg_unet50_ep100_b64_crp512_iter0/label_next_iter'
    # os.makedirs(output_dir, exist_ok=True)
    # gen_label_with_pred_fb(img_dir, pred_score_dir, superpixel_dir, last_iter_dir, output_dir)
    
    # img_dir = './data/NeurIPS2022_CellSeg/images'
    # last_iter_dict_dir = './data/NeurIPS2022_CellSeg/labels/iter0_instance'
    # superpixel_dir = './data/NeurIPS2022_CellSeg/superpixel_adaptive'
    # pred_score_dir = './workspace/cellseg_unet50_ep100_b64_crp512_iter0/results_test/score'
    # output_dir = './workspace/cellseg_unet50_ep100_b64_crp512_iter0/label_next_iter/labels'
    # output_dict_dir = './workspace/cellseg_unet50_ep100_b64_crp512_iter0/label_next_iter/instance_dicts'
    # os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(output_dict_dir, exist_ok=True)
    # gen_label_with_pred_fb_instance(img_dir, pred_score_dir, superpixel_dir, last_iter_dict_dir, output_dir, output_dict_dir)
    
    
    # img_dir = './data/NeurIPS2022_CellSeg/images'
    # gt_dir = './data/NeurIPS2022_CellSeg/gts'
    # output_dir = './data/NeurIPS2022_CellSeg/labels/full'
    # os.makedirs(output_dir,exist_ok=True)
    # gen_full_supervision_label(img_dir, gt_dir, output_dir)