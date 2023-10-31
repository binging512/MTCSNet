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
    boundary = morphology.binary_dilation(boundary, morphology.disk(0))

    interior_temp = np.logical_and(~boundary, inst_map > 0)
    # interior_temp[boundary] = 0
    interior_temp = morphology.remove_small_objects(interior_temp, min_size=16)
    interior = np.zeros_like(inst_map, dtype=np.uint8)
    interior[interior_temp] = 1
    interior[boundary] = 0
    return interior

def mc_distance_postprocessing(cell_prediction, th_cell, seeds, downsample):
    """ Post-processing for distance label (cell + neighbor) prediction.

    :param cell_prediction: 就是heatmap,0-1
    :type cell_prediction:
    :param th_cell:
    :type th_cell: float
    :param th_seed:
    :type th_seed: float
    :param downsample:
    :type downsample: bool

    :return: Instance segmentation mask.
    """

    min_area = 10  # keep only seeds larger than threshold area

    # Instance segmentation (use summed up channel 0)
    # sigma_cell = 0.5
    # cell_prediction[0] = gaussian_filter(cell_prediction[0], sigma=sigma_cell)  # slight smoothing
    mask = cell_prediction > th_cell  # get binary mask by thresholding distance prediction  比0.5大的mask（应该是看作细胞

    seeds = measure.label(seeds, background=0)
    props = measure.regionprops(seeds)  # Remove very small seeds  去除面积过小的种子

    region_miu = np.mean([prop.area for prop in props if prop.area > min_area])
    region_sigma = np.sqrt(np.var([prop.area for prop in props if prop.area > min_area]))
    region_range = [region_miu - 2 * region_sigma, region_miu + 2 * region_sigma]

    for idx, prop in enumerate(props):
        if prop.area < min_area or prop.area < region_range[0]:
            seeds[seeds == prop.label] = 0  # 但是这样就进入争议区域了

    if len(props) <= 50:  # 小的基，可以用腐蚀或者膨胀
        seeds = cv2.dilate((seeds > 0).astype(np.uint16), np.ones((5, 5), np.uint8), iterations=1)

    seeds = measure.label(seeds, background=0)
    prediction_instance = watershed(image=-cell_prediction, markers=seeds, mask=mask, watershed_line=False)  # 确定0.5-0.7之间属于哪个instance
    
    # # Semantic segmentation / classification
    # prediction_class = np.zeros_like(prediction_instance)
    # for idx in range(1, prediction_instance.max()+1):
    #     # Get sum of distance prediction of selected cell for each class (class 0 is sum of the other classes)
    #     pix_vals = cell_prediction[1:][:, prediction_instance == idx]
    #     cell_layer = np.sum(pix_vals, axis=1).argmax() + 1  # +1 since class 0 needs to be considered for argmax
    #     prediction_class[prediction_instance == idx] = cell_layer

    if downsample:
        # Downsample instance segmentation
        prediction_instance = rescale(prediction_instance,
                                      scale=0.8,
                                      order=0,
                                      preserve_range=True,
                                      anti_aliasing=False).astype(np.int32)

        # Downsample semantic segmentation
        # prediction_class = rescale(prediction_class,
        #                            scale=0.8,
        #                            order=0,
        #                            preserve_range=True,
        #                            anti_aliasing=False).astype(np.uint16)

    # Combine instance segmentation and semantic segmentation results
    # prediction = np.concatenate((prediction_instance[np.newaxis, ...], prediction_class[np.newaxis, ...]), axis=0)
    prediction = prediction_instance
    return mask, seeds, prediction.astype(np.int32)

def gen_label_with_heat_fb_instance(heat_dir, weak_dir, last_iter_split, output_dir):
    heat_names = os.listdir(heat_dir)
    output_label_dir = os.path.join(output_dir, 'labels')
    output_heat_dir = os.path.join(output_dir, 'heatmaps')
    output_dict_dir = os.path.join(output_dir, 'instance_dicts')
    os.makedirs(output_label_dir, exist_ok=True)
    os.makedirs(output_heat_dir, exist_ok=True)
    os.makedirs(output_dict_dir, exist_ok=True)
    last_iter_split_dict = json.load(open(last_iter_split, 'r'))
    for ii, train_item in enumerate(last_iter_split_dict['train']):
        print("Processing the {}/{} images....".format(ii,len(heat_names)),end='\r')
        heat_name = os.path.basename(train_item['img_path']).split('.')[0]+'.png'
        heat_path = os.path.join(heat_dir, heat_name)
        weak_path = os.path.join(weak_dir, heat_name)
        
        heat = cv2.imread(heat_path, flags=0)/255
        weak = cv2.imread(weak_path, flags=0)
        label = np.ones(heat.shape[:2])*255
        
        seeds = np.zeros(heat.shape)
        seeds[weak==1] = 1
        heat[weak==1] = 1
        heat[weak==0] = 0
        
        # Foreground mask and background mask
        th_cell = 0.7
        fg_mask, seeds, new_inst_map = mc_distance_postprocessing(heat, th_cell, seeds, downsample=False)
        bg_mask = np.zeros(heat.shape)
        bg_mask[heat< 0.1] = 1
        # generating the label
        fg_label = create_interior_map(new_inst_map)
        label[bg_mask==1]=0
        label = label*(1-fg_mask)+ fg_label*fg_mask
        # generating the heatmap
        fg_heat = cv2.GaussianBlur(fg_label.astype(np.float32), (5,5), -1)
        fg_heat = fg_heat/np.max(fg_heat)*255
        
        save_path = os.path.join(output_label_dir, heat_name)
        heat_save_path = os.path.join(output_heat_dir, heat_name)
        cv2.imwrite(save_path, label)
        cv2.imwrite(heat_save_path, fg_heat)
        
        train_item['weak_label_path'] = save_path
        train_item['weak_heat_path'] = heat_save_path
    split_save_path = os.path.join(output_dir, 'next_iter.json')
    json.dump(last_iter_split_dict, open(split_save_path, 'w'), indent=2)
    pass


if __name__=="__main__":
    heat_dir = './workspace/Mo_Mo_unet50r_1head_ep150_b4_crp512_iter0/results_test/heat'
    weak_dir = './data/MoNuSeg/train/labels/weak'
    last_iter_split = './data/splits/train_Mo_val_Mo_iter0.json'
    output_dir = './workspace/Mo_Mo_unet50r_1head_ep150_b4_crp512_iter0/label_next_iter_heat_.7_.1'

    gen_label_with_heat_fb_instance(heat_dir, weak_dir, last_iter_split, output_dir)