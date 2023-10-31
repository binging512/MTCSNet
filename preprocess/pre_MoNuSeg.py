import os
import cv2
import json
import numpy as np
import tifffile as tif
from xml.etree import ElementTree as ET
from skimage import segmentation, morphology
from skimage.segmentation import slic,mark_boundaries
import random

def gen_point_supervision(semantic):
    point_dict = {"background":{}, "foreground":{}}
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(semantic.astype(np.uint8),connectivity=8)
    # select the foreground points
    num_cells = np.max(labels)
    for ii in range(1,num_cells+1):
        x,y,w,h,s = stats[ii]
        centroid = centroids[ii]
        centroid = list(map(int,centroid))
            
        flag = 0
        attempt_num = 0
        point_range = 0.05
        while flag == 0:
            if attempt_num == 20:
                point_range += 0.05
                attempt_num = 0
            offset_ratio_x = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
            offset_ratio_y = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
            offset_x = round(w*offset_ratio_x)
            offset_y = round(h*offset_ratio_y)
            point_x = centroid[0] + offset_x
            point_y = centroid[1] + offset_y
            if labels[point_y, point_x] == ii:
                flag = 1
            else:
                attempt_num += 1
            if point_range > 0.2:
                print("False point in image!!")
                break
        
        point_dict['foreground'][str(ii)] = {'x':int(x),
                                            'y':int(y),
                                            'w':int(w),
                                            'h':int(h),
                                            's':int(s),
                                            'centroid':centroid,
                                            "select_point":[point_x, point_y]}
    # select the background points
    background = np.zeros(semantic.shape)
    H,W = background.shape
    background[semantic==1] = 1
    background_points= []
    need_point_num = max(10,round(num_cells))
    while len(background_points)<need_point_num:
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        if background[y][x] == 0:
            background_points.append([x,y])
    for ii, point in enumerate(background_points):
        point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                'y':int(point[1])}
    return point_dict

def get_distance(pt1, pt2):
    dist = np.sqrt(np.power(pt1[0]-pt2[0],2)+np.power(pt1[1]-pt2[1],2))
    return dist

def get_min_distance(fg_point_dict):
    select_points = [v['select_point'] for k,v in fg_point_dict.items()]
    if len(select_points)>=2:
        min_dist_list = []
        for i, pt1 in enumerate(select_points):
            dist_list = []
            for ii, pt2 in enumerate(select_points):
                dist_list.append(get_distance(pt1, pt2))
            dist_list.remove(0)
            min_dist_list.append(min(dist_list))
        min_dist = min(min_dist_list)
    else:
        min_dist = 100
    min_dist = min(100,min_dist)
    return min_dist

def gen_superpixel_adaptive(img, point_dict):
    H,W,C = img.shape
    # find the minimum distance between the foreground
    fg_points_dict = point_dict['foreground']
    min_dist = get_min_distance(fg_points_dict)
    min_dist = max(20, min_dist)
    h_num = H/min_dist
    w_num = W/min_dist
    n_segment = h_num*w_num*2
    seg = slic(img, n_segments=n_segment, compactness=10, sigma=3, max_num_iter=10, slic_zero=True)
    vis = mark_boundaries(img,seg)*255
    return seg, vis

def gen_weak_label_with_point_fb(img, point_dict, superpixel):
    weak_label = np.ones(img.shape[:2])*255
    weak_label_count = np.zeros(img.shape[:2])
    weak_label_valid = np.zeros(img.shape[:2])
    fg_points_dict = point_dict['foreground']
    bg_points_dict = point_dict['background']
    instance_dict = {"superpixels":{}, "instances":{}, 'background':[], 'labeled_sp':[]}
    for superpixel_idx in range(1,np.max(superpixel)+1):
        instance_dict['superpixels'][superpixel_idx]=-1
        
    # For background
    for point_idx, bg_point_prop in bg_points_dict.items():
        point_x = bg_point_prop['x']
        point_y = bg_point_prop['y']
        superpixel_value = superpixel[point_y, point_x]
        # get the valid region
        weak_label_valid[superpixel==superpixel_value] = 1
        # get the labeled superpixel boundary 
        weak_temp = np.zeros(img.shape[:2], dtype=np.uint8)
        weak_temp[superpixel == superpixel_value] = 1
        weak_temp_boundary = segmentation.find_boundaries(weak_temp, mode='outer')
        # set the weak label
        weak_label[superpixel == superpixel_value] = 0
        # get the counting image for adj pixels
        weak_label_count = weak_label_count + weak_temp + weak_temp_boundary
        instance_dict['superpixels'][str(superpixel_value)] = 0
        instance_dict['background'].append(int(superpixel_value))
        instance_dict['labeled_sp'].append(int(superpixel_value))

    # For foreground
    for point_idx, fg_point_prop in fg_points_dict.items():
        point_x, point_y = fg_point_prop['select_point']
        superpixel_value = superpixel[point_y,point_x]
        # get the valid region
        weak_label_valid[superpixel==superpixel_value] = 1
        # get the labeled superpixel boundary 
        weak_temp = np.zeros(img.shape[:2], dtype=np.uint8)
        weak_temp[superpixel == superpixel_value] = 1
        weak_temp_boundary = segmentation.find_boundaries(weak_temp, mode='outer')
        # set the weak label
        weak_label[superpixel == superpixel_value] = 1
        # get the counting image for adj pixels
        weak_label_count = weak_label_count + weak_temp + weak_temp_boundary
        instance_dict['superpixels'][str(superpixel_value)] = int(point_idx)
        instance_dict['instances'][str(point_idx)] = [int(superpixel_value)]
        instance_dict['labeled_sp'].append(int(superpixel_value))
    
    # fine-grained fix: 
    adj_pixels = np.zeros(img.shape[:2])
    adj_pixels[weak_label_count>1] = 1
    adj_pixels = morphology.dilation(adj_pixels, morphology.disk(1))
    adj_pixels = adj_pixels*weak_label_valid
    weak_label[adj_pixels == 1] = 0

    return weak_label, instance_dict, weak_label_valid

def gen_full_label_with_semantic(semantic):
    full_label = np.zeros(semantic.shape)
    full_label[semantic>0]=1
    return full_label

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
    interior[boundary] = 0          # it used to be 2
    return interior

def preprocess_MoNuSeg(image_dir, anno_dir, output_dir):
    print("Preprocessing the MoNuSeg dataset...")
    output_gt_dir = os.path.join(output_dir, 'gts')
    output_semantic_dir = os.path.join(output_dir, 'semantics')
    output_vis_dir = os.path.join(output_dir, 'vis')
    os.makedirs(output_gt_dir,exist_ok=True)
    os.makedirs(output_semantic_dir,exist_ok=True)
    os.makedirs(output_vis_dir,exist_ok=True)
    for i, image_name in enumerate(sorted(os.listdir(image_dir))):
        print("Processing the {}/{} images...".format(i, len(os.listdir(image_dir))), end='\r')
        image_path = os.path.join(image_dir, image_name)
        annotation_path = os.path.join(anno_dir, image_name.replace('.tif', '.xml'))
        img = tif.imread(image_path)
        anno = ET.parse(annotation_path)
        H,W,C = img.shape
        gt = np.zeros((H,W))
        regions = anno.iter('Region')
        for ii, region in enumerate(regions):
            vertices = region.getchildren()[1]
            pts = []
            for v in vertices.getchildren():
                x = round(float(v.get('X')))
                y = round(float(v.get('Y')))
                pts.append([x,y])
            cv2.fillPoly(gt, [np.array(pts)], color=ii+1)
        save_path = os.path.join(output_gt_dir, image_name.replace('.tif', '.tiff'))
        tif.imwrite(save_path, gt)
        
        # get the semantic segmentation maps
        semantic = create_interior_map(gt)
        semantic[semantic == 1] = 128
        semantic[semantic == 2] = 255
        save_semantic_path = os.path.join(output_semantic_dir, image_name.replace('.tif','.png'))
        cv2.imwrite(save_semantic_path, semantic)
        
        vis = np.zeros_like(gt)
        vis[gt>0] = 255
        vis = vis[:,:,np.newaxis]
        vis = np.concatenate((vis,vis,vis), axis=2)
        vis_path = os.path.join(output_vis_dir,image_name.replace('.tif','.png'))
        vis = np.hstack((img, vis))
        cv2.imwrite(vis_path, vis)
    pass

def gen_full_n_weak_heatmap(full_label, weak_label, weak_label_valid):
    full_label = full_label.astype(np.float32)
    full_heat = cv2.GaussianBlur(full_label, (5,5), -1)*255
    
    weak_label[weak_label==255] = 0
    weak_label = morphology.dilation(weak_label, morphology.disk(1))
    weak_heat = cv2.GaussianBlur(weak_label.astype(np.float32), (5,5), -1)
    weak_heat = weak_heat*weak_label_valid
    weak_heat = weak_heat/np.max(weak_heat)*255
    return full_heat, weak_heat

def gen_weak_n_full_labels(image_dir, semantic_dir, output_dir):
    print("Generating the weak and full supervision labels...")
    output_superpixel_dir = os.path.join(output_dir,'superpixels')
    output_point_dir = os.path.join(output_dir,'points')
    output_weak_dir = os.path.join(output_dir, 'labels/weak')
    output_weak_heat_dir = os.path.join(output_dir, 'labels/weak_heatmap')
    output_inst_dir = os.path.join(output_dir, 'labels/weak_inst')
    output_full_dir = os.path.join(output_dir, 'labels/full')
    output_full_heat_dir = os.path.join(output_dir, 'labels/full_heatmap')
    os.makedirs(output_superpixel_dir, exist_ok=True)
    os.makedirs(output_point_dir, exist_ok=True)
    os.makedirs(output_weak_dir, exist_ok=True)
    os.makedirs(output_weak_heat_dir, exist_ok=True)
    os.makedirs(output_inst_dir, exist_ok=True)
    os.makedirs(output_full_dir, exist_ok=True)
    os.makedirs(output_full_heat_dir, exist_ok=True)
    for ii, image_name in enumerate(sorted(os.listdir(image_dir))):
        print("Generating the {}/{} supervision labels...".format(ii, len(os.listdir(image_dir))), end='\r')
        img = tif.imread(os.path.join(image_dir, image_name))
        semantic = cv2.imread(os.path.join(semantic_dir, image_name.replace('.tif','.png')), flags=0)
        semantic[semantic==128] = 1
        point_dict = gen_point_supervision(semantic)
        superpixel, sp_vis = gen_superpixel_adaptive(img, point_dict)
        weak_label, instance_dict, weak_label_valid = gen_weak_label_with_point_fb(img, point_dict, superpixel)
        full_label = gen_full_label_with_semantic(semantic)
        full_heat, weak_heat = gen_full_n_weak_heatmap(full_label.copy(), weak_label.copy(), weak_label_valid.copy())
        # save the mid-results
        json.dump(point_dict, open(os.path.join(output_point_dir, image_name.replace('.tif','.json')), 'w'),indent=2)
        tif.imwrite(os.path.join(output_superpixel_dir, image_name), superpixel)
        cv2.imwrite(os.path.join(output_superpixel_dir, image_name.replace('.tif','_vis.png')), sp_vis)
        cv2.imwrite(os.path.join(output_weak_dir, image_name.replace('.tif','.png')), weak_label)
        cv2.imwrite(os.path.join(output_weak_heat_dir, image_name.replace('.tif','.png')), weak_heat)
        cv2.imwrite(os.path.join(output_full_dir, image_name.replace('.tif','.png')), full_label)
        cv2.imwrite(os.path.join(output_full_heat_dir, image_name.replace('.tif','.png')), full_heat)
        json.dump(instance_dict, open(os.path.join(output_inst_dir, image_name.replace('.tif','.json')), 'w'), indent=2)
    pass

if __name__=="__main__":
    image_dir = './data/MoNuSeg/train/images'
    anno_dir = './data/MoNuSeg/train/annotations'
    semantic_dir = './data/MoNuSeg/train/semantics'
    output_dir = './data/MoNuSeg/train'
    # preprocess_MoNuSeg(image_dir, anno_dir, output_dir)
    gen_weak_n_full_labels(image_dir, semantic_dir, output_dir)
    
    image_dir = './data/MoNuSeg/test/images'
    anno_dir = './data/MoNuSeg/test/annotations'
    output_dir = './data/MoNuSeg/test'
    # preprocess_MoNuSeg(image_dir, anno_dir, output_dir)
    pass
