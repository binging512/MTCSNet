import os
import cv2
import glob
import tifffile as tif
import numpy as np
from skimage import measure, morphology, segmentation
from skimage.segmentation import slic,mark_boundaries
import random
import math
import json
from scipy.spatial import Voronoi, voronoi_plot_2d
import shutil
import torch.nn as nn
import torch

def gen_voronoi(semantic,point_dict):
    vor_img = np.zeros(semantic.shape[:2])
    point_list = []
    for point_idx, point_prop in point_dict['foreground'].items():
        select_point = point_prop['select_point']
        point_list.append(select_point)
    if len(point_list)<=2:
        H,W = semantic.shape[:2]
        add_point_list = [[0,0],[0,W-1],[H-1,0],[H-1,W-1]]
        point_list += add_point_list
    
    point_list = np.array(point_list)
    vor = Voronoi(point_list,)
    
    center = vor.points.mean(axis=0)
    ptp_bound = vor.points.ptp(axis=0)
    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            finite_segments.append(vor.vertices[simplex])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            far_point = vor.vertices[i] + direction * ptp_bound.max()

            infinite_segments.append([vor.vertices[i], far_point])
    
    for seg in finite_segments:
        start_p, stop_p = seg
        start_p = [round(start_p[0]), round(start_p[1])]
        stop_p = [round(stop_p[0]), round(stop_p[1])]
        cv2.line(vor_img, start_p, stop_p, 255, thickness=1)
    for seg in infinite_segments:
        start_p, stop_p = seg
        start_p = [round(start_p[0]), round(start_p[1])]
        stop_p = [round(stop_p[0]), round(stop_p[1])]
        cv2.line(vor_img, start_p, stop_p, 255, thickness=1)
    kernel = np.array([[0,1,0],[1,1,0],[0,0,0]])
    vor_img = morphology.dilation(vor_img, kernel)
    return vor_img

def gen_background_points_w_kmeans(img, point_dict, semantic, vor):
    H,W,C = img.shape
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
        
    new_point_dict = {'foreground':point_dict['foreground'], 'background':{}}
    bg_point_num = len(list(point_dict['background'].keys()))
    background_points = []
    
    semantic[semantic==128] = 1
    while len(background_points)<bg_point_num:
        x = random.randint(0, W-1)
        y = random.randint(0, H-1)
        if mask[y][x] == 0:
            background_points.append([x,y])
        
        for ii, point in enumerate(background_points):
            new_point_dict['background'][str(ii+1)] = {'x':int(point[0]),
                                                       'y':int(point[1])}
    return new_point_dict

def gen_point_supervision(gt):
    point_dict = {"background":{}, "foreground":{}}
    region_props = measure.regionprops(gt)
    H,W = gt.shape
    # select the foreground points
    for region_prop in region_props:
        y1, x1, y2, x2= region_prop.bbox
        centroid = region_prop.centroid
        centroid = list(map(int,centroid))
        valid_id = region_prop.label
        w = x2-x1
        h = y2-y1
            
        flag = 0
        attempt_num = 0
        point_range = 0.0  # 0.05
        while flag == 0:
            if attempt_num == 20:
                point_range += 0.05
                attempt_num = 0
            offset_ratio_x = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
            offset_ratio_y = ((random.random()-0.5)/0.5)*point_range      #[-0.25,0.25]
            offset_x = round(w*offset_ratio_x)
            offset_y = round(h*offset_ratio_y)
            point_x = max(0,min(centroid[1] + offset_x, W-1))
            point_y = max(0,min(centroid[0] + offset_y, H-1))
            if gt[point_y, point_x] == valid_id:
                flag = 1
            else:
                attempt_num += 1
            if point_range > 0.3:
                print("False point in image!!")
                break
        select_point = [point_x, point_y]
        
        # get a boundary point
        mask = np.zeros(gt.shape)
        mask[gt==valid_id] = 1
        bnd = segmentation.find_boundaries(mask, mode='inner')
        bnd_points = np.array(np.where(bnd==1)).T
        dist_list = []
        for bnd_point in bnd_points:
            deg, dist = get_degree_n_distance(select_point, [bnd_point[1],bnd_point[0]])
            dist_list.append(dist)
        dist_max_idx = np.argmax(np.array(dist_list))
        select_bnd_point = [int(bnd_points[dist_max_idx][1]),int(bnd_points[dist_max_idx][0])]
        
        point_dict['foreground'][str(valid_id)] = {'x':int(x1),
                                            'y':int(y1),
                                            'w':int(w),
                                            'h':int(h),
                                            'centroid': centroid,
                                            "select_point": select_point,
                                            "boundary_point": select_bnd_point}
    # select the background points
    background = np.zeros(gt.shape)
    H,W = background.shape
    background[gt>1] = 1
    background_points = []
    need_point_num = max(300,round(np.max(gt)))     # background point number
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
    min_dist = min(max(20, min_dist),40)
    h_num = H/min_dist
    w_num = W/min_dist
    n_segment = h_num*w_num*2
    seg = slic(img, n_segments=n_segment, compactness=10, sigma=3, max_num_iter=10, slic_zero=True)
    vis = mark_boundaries(img,seg)*255
    for inst_idx, inst_prop in fg_points_dict.items():
        fg_point = inst_prop['select_point']
        bnd_point = inst_prop['boundary_point']
        vis = cv2.circle(vis, fg_point, 1, color=[255, 0, 0], thickness=-1)
        vis = cv2.circle(vis, bnd_point, 1, color=[0, 255, 0], thickness=-1)
    for inst_idx, inst_prop in point_dict['background'].items():
        bg_point = [inst_prop['x'], inst_prop['y']]
        vis = cv2.circle(vis, bg_point, 1, color=[0, 0, 255], thickness=-1)
    return seg, vis

def gen_weak_label_with_point_fb(img, point_dict, superpixel, vor_img):
    weak_label = np.ones(img.shape[:2])*255
    weak_heat = np.zeros(img.shape[:2])
    weak_label_count = np.zeros(img.shape[:2])
    weak_label_valid = np.zeros(img.shape[:2])
    fg_points_dict = point_dict['foreground']
    bg_points_dict = point_dict['background']
    repeat_foreground_superpixels = []
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
        
        if instance_dict['superpixels'][superpixel_value] == 0:
            pass
        elif instance_dict['superpixels'][superpixel_value] > 0:
            pass
        else:
            # set the weak label
            weak_label[superpixel == superpixel_value] = 0
            # get the counting image for adj pixels
            weak_label_count = weak_label_count + weak_temp + weak_temp_boundary
            instance_dict['superpixels'][superpixel_value] = 0
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
        
        if instance_dict['superpixels'][superpixel_value] > 0:
            repeat_foreground_superpixels.append(superpixel_value)
            print("[Error] Warning: at least 2 foreground points are in the same superpixel!!!!!")
            instance_dict['instances'][int(point_idx)] = [int(superpixel_value)]
            pass
        elif instance_dict['superpixels'][superpixel_value] == 0:
            # set the weak label
            weak_label[superpixel == superpixel_value] = 1
            weak_heat[superpixel == superpixel_value] = 1
            instance_dict['superpixels'][superpixel_value] = int(point_idx)
            instance_dict['background'].remove(int(superpixel_value))
            instance_dict['instances'][int(point_idx)] = [int(superpixel_value)]
        else:
            # set the weak label
            weak_label[superpixel == superpixel_value] = 1
            weak_heat[superpixel == superpixel_value] = 1
            # get the counting image for adj pixels
            weak_label_count = weak_label_count + weak_temp + weak_temp_boundary
            instance_dict['superpixels'][superpixel_value] = int(point_idx)
            instance_dict['instances'][int(point_idx)] = [int(superpixel_value)]
            instance_dict['labeled_sp'].append(int(superpixel_value))
    
    # fine-grained fix: 
    vor_img[vor_img==255] = 1
    boundary_mask = np.zeros(img.shape[:2])
    for sp in repeat_foreground_superpixels:
        boundary_mask[superpixel==sp]=1
    boundary_mask = boundary_mask*vor_img
    
    adj_pixels = np.zeros(img.shape[:2])
    adj_pixels[weak_label_count>1] = 1
    adj_pixels[boundary_mask==1] = 1
    adj_pixels = morphology.dilation(adj_pixels, morphology.disk(1))
    adj_pixels = adj_pixels*weak_label_valid
    weak_label[adj_pixels == 1] = 0
    # weak heat
    weak_heat[adj_pixels == 1] = 0
    weak_heat = morphology.dilation(weak_heat, morphology.disk(1))
    weak_heat = cv2.GaussianBlur(weak_heat.astype(np.float32), (3,3), -1)
    weak_heat = weak_heat*weak_label_valid
    weak_heat = weak_heat/np.max(weak_heat)*255

    return weak_label, instance_dict, weak_label_valid, weak_heat

def gen_full_label_with_semantic(semantic):
    semantic[semantic==128] = 1
    semantic[semantic==255] = 2
    full_label = np.zeros(semantic.shape)
    full_label[semantic==1] = 1
    full_label[semantic==2] = 2
    
    full_cls_label = full_label.astype(np.float32)
    full_heat = cv2.GaussianBlur(full_cls_label, (5,5), -1)*255
    return full_cls_label, full_heat

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

def gen_dist_label(img, vor, point_dict):
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
    return label

def create_interior_map(inst_map, if_boundary):
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
    if if_boundary:
        interior[boundary] = 2
    else:
        interior[boundary]= 0
    return interior

def gen_full_n_weak_heatmap(full_label, weak_label, weak_label_valid):
    full_label = full_label.astype(np.float32)
    full_heat = cv2.GaussianBlur(full_label, (5,5), -1)*255
    
    weak_label[weak_label==255] = 0
    weak_label = morphology.dilation(weak_label, morphology.disk(1))
    weak_heat = cv2.GaussianBlur(weak_label.astype(np.float32), (5,5), -1)
    weak_heat = weak_heat*weak_label_valid
    weak_heat = weak_heat/np.max(weak_heat)*255
    return full_heat, weak_heat

def recollect_files(data_root, img_dir, gt_dir):
    img_paths = sorted(glob.glob(os.path.join(data_root,'Slide_??/*.png'),recursive=True))
    gt_paths = [img_path.replace("Slide","GT") for img_path in img_paths]
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    for ii, img_path in enumerate(img_paths):
        save_img_path = os.path.join(img_dir, os.path.basename(img_path))
        save_gt_path = os.path.join(gt_dir, os.path.basename(gt_paths[ii]).replace('.png','.tiff'))
        shutil.copy(img_path, save_img_path)
        gt = cv2.imread(gt_paths[ii], flags=0)
        gt = measure.label(gt)
        tif.imwrite(save_gt_path,gt)
    
def preprocess_TNBC(img_dir, gt_dir, output_dir, train=True):
    print("Preprocessing the TNBC dataset...")
    output_img_dir = os.path.join(output_dir, 'images')
    output_superpixel_dir = os.path.join(output_dir, 'superpixels')
    output_gt_dir = os.path.join(output_dir, 'gts')
    output_semantic_dir = os.path.join(output_dir,'semantics')
    output_vis_dir = os.path.join(output_dir, 'vis')
    output_point_dir = os.path.join(output_dir,'points')
    output_voronoi_dir = os.path.join(output_dir,'voronois')
    output_full_dist_dir = os.path.join(output_dir,'labels/full_distmap')
    output_weak_dist_dir = os.path.join(output_dir,'labels/weak_distmap')
    output_full_label_dir = os.path.join(output_dir,'labels/full')
    output_full_heatmap_dir = os.path.join(output_dir,'labels/full_heatmap')
    output_weak_label_dir = os.path.join(output_dir,'labels/weak')
    output_weak_heatmap_dir = os.path.join(output_dir,'labels/weak_heatmap')
    
    img_names = os.listdir(img_dir)
    for ii, img_name in enumerate(sorted(img_names)):
        print("Preprocessing the {}/{} images...".format(ii, len(img_names)), end='\r')
        # check if done?
        # if os.path.exists(os.path.join(output_weak_heatmap_dir, img_name)):
        #     continue
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(gt_dir, img_name.replace('.png','.tiff'))
        
        img = cv2.imread(img_path)
        gt = tif.imread(gt_path)
        gt = morphology.dilation(gt, morphology.disk(1))
        
        # get the semantic segmentation maps
        semantic = create_interior_map(gt, if_boundary=False)
        semantic[semantic == 1] = 128
        semantic[semantic == 2] = 255
        save_semantic_path = os.path.join(output_semantic_dir,img_name)
        os.makedirs(os.path.dirname(save_semantic_path), exist_ok=True)
        cv2.imwrite(save_semantic_path, semantic)
        
        # get the visualization
        vis = np.zeros_like(gt)
        vis[gt>0] = 255
        vis = vis[:,:,np.newaxis]
        vis = np.concatenate((vis,vis,vis), axis=2)
        vis = np.hstack((img, vis)).astype(np.uint8)
        save_vis_path = os.path.join(output_vis_dir, img_name)
        os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
        cv2.imwrite(save_vis_path, vis)
        
        if train:
            # generate the point supervision
            point_dict = gen_point_supervision(gt.copy())
            save_point_path = os.path.join(output_point_dir, img_name.replace('.png','.json'))
            os.makedirs(os.path.dirname(save_point_path), exist_ok=True)
            json.dump(point_dict, open(save_point_path,'w'), indent=2)
            
            # generate the superpixels
            superpixel, sp_vis = gen_superpixel_adaptive(img, point_dict)
            save_superpixel_path = os.path.join(output_superpixel_dir, img_name.replace('.png','.tiff'))
            os.makedirs(os.path.dirname(save_superpixel_path), exist_ok=True)
            tif.imwrite(save_superpixel_path, superpixel)
            save_sp_vis_path = os.path.join(output_superpixel_dir, img_name.replace('.png','_vis.png'))
            os.makedirs(os.path.dirname(save_sp_vis_path), exist_ok=True)
            cv2.imwrite(save_sp_vis_path, sp_vis)
            
            # generate the voronoi images
            vor_img = gen_voronoi(semantic.copy(), point_dict)
            save_vor_path = os.path.join(output_voronoi_dir, img_name)
            os.makedirs(os.path.dirname(save_vor_path), exist_ok=True)
            cv2.imwrite(save_vor_path, vor_img)
            
            # generate the full distmap labels
            full_distmap = semantic
            full_distmap[full_distmap==255]=0
            full_distmap[full_distmap==128]=255
            save_full_distmap_path = os.path.join(output_full_dist_dir, img_name)
            os.makedirs(os.path.dirname(save_full_distmap_path),exist_ok=True)
            cv2.imwrite(save_full_distmap_path, full_distmap)
            
            weak_distmap = gen_dist_label(img, vor_img, point_dict)
            save_weak_distmap_path = os.path.join(output_weak_dist_dir, img_name)
            os.makedirs(os.path.dirname(save_weak_distmap_path),exist_ok=True)
            cv2.imwrite(save_weak_distmap_path, weak_distmap)
            
            # generate the full label and heatmap
            full_label, full_heat = gen_full_label_with_semantic(semantic)
            save_full_label_path = os.path.join(output_full_label_dir, img_name)
            os.makedirs(os.path.dirname(save_full_label_path), exist_ok=True)
            cv2.imwrite(save_full_label_path, full_label)
            save_full_heat_path = os.path.join(output_full_heatmap_dir, img_name)
            os.makedirs(os.path.dirname(save_full_heat_path), exist_ok=True)
            cv2.imwrite(save_full_heat_path, full_heat)
            
            # generate the weak label and heatmap
            weak_label, instance_dict, weak_label_valid, weak_heat = gen_weak_label_with_point_fb(img, point_dict, superpixel, vor_img.copy())
            save_weak_label_path = os.path.join(output_weak_label_dir, img_name)
            os.makedirs(os.path.dirname(save_weak_label_path),exist_ok=True)
            cv2.imwrite(save_weak_label_path, weak_label)
            save_weak_heat_path = os.path.join(output_weak_heatmap_dir, img_name)
            os.makedirs(os.path.dirname(save_weak_heat_path),exist_ok=True)
            cv2.imwrite(save_weak_heat_path, weak_heat)
    
def change_bg_point(preprocess_dir, output_dir):
    output_point_dir = os.path.join(output_dir, 'points_fuse')
    output_full_label_dir = os.path.join(output_dir,'labels_fuse/full')
    output_full_heatmap_dir = os.path.join(output_dir,'labels_fuse/full_heatmap')
    output_full_dist_dir = os.path.join(output_dir,'labels_fuse/full_distmap')
    output_weak_label_dir = os.path.join(output_dir,'labels_fuse/weak')
    output_weak_heatmap_dir = os.path.join(output_dir,'labels_fuse/weak_heatmap')
    output_weak_dist_dir = os.path.join(output_dir,'labels_fuse/weak_distmap')
    output_kmeans_dir = os.path.join(output_dir,'kmeans')
    print("Change the background points with fused methods...")
    img_dir = os.path.join(preprocess_dir, 'images')
    img_names = sorted(os.listdir(img_dir))
    
    kernel_size = 11
    conv_layer = nn.Conv2d(3,3, kernel_size, bias=False).eval()
    conv_layer.weight = torch.nn.Parameter(torch.ones([3,3,kernel_size,kernel_size]))
    
    for ii, img_name in enumerate(img_names):
        print("Generating the {}/{} labels....".format(ii, len(img_names)), end='\r')
        img_path = os.path.join(img_dir, img_name)
        point_path = os.path.join(preprocess_dir, 'points', img_name.replace('.png','.json'))
        superpixel_path = os.path.join(preprocess_dir, 'superpixels', img_name.replace('.png','.tiff'))
        semantic_path = os.path.join(preprocess_dir, 'semantics', img_name)
        vor_path = os.path.join(preprocess_dir, 'labels/weak_distmap', img_name)
        vor_img_path = os.path.join(preprocess_dir, 'voronois',img_name)
        img = cv2.imread(img_path)
        point_dict = json.load(open(point_path, 'r'))
        superpixel = tif.imread(superpixel_path)
        semantic = cv2.imread(semantic_path, flags=0)
        vor = cv2.imread(vor_path, flags=0)
        vor_img = cv2.imread(vor_img_path, flags=0)
        H,W,C = img.shape
        
        img_conv = conv_layer(torch.tensor(img,dtype=torch.float32).permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0).detach().numpy()
        
        pixels = img_conv.reshape((-1,3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 500, 0.1)
        K = 10
        cluster_list = [0,0,0,0,0,0,0,0,0,0]
        _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        
        labels = labels.reshape((H-(kernel_size-1),W-(kernel_size-1)))
        labels_mask = np.zeros((H,W),dtype=np.int16)
        labels_mask[int(kernel_size/2):H-int(kernel_size/2),int(kernel_size/2):W-int(kernel_size/2)] = labels
        labels = labels_mask
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
        mask = morphology.erosion(mask, morphology.disk(1))
        
        vor[vor==255]=1
        mask = mask*vor
        mask[:int(kernel_size/2),:] = 1
        mask[-int(kernel_size/2):,:] = 1
        mask[:,:int(kernel_size/2)] = 1
        mask[:,-int(kernel_size/2):] = 1
        vis = np.hstack((img, np.stack((mask,mask,mask),axis=2)))
        vis[vis==1]=255
        save_vis_path = os.path.join(output_kmeans_dir,img_name.replace('.png','_vis.png'))
        os.makedirs(os.path.dirname(save_vis_path),exist_ok=True)
        cv2.imwrite(save_vis_path, vis)
        
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
            
        save_point_path = os.path.join(output_point_dir, img_name.replace('.png','.json'))
        os.makedirs(os.path.dirname(save_point_path), exist_ok=True)
        json.dump(new_point_dict, open(save_point_path, 'w'), indent=2)
        
        # generate the labels
        # generate the full distmap labels
        full_distmap = semantic
        full_distmap[full_distmap==255]=0
        full_distmap[full_distmap==128]=255
        save_full_distmap_path = os.path.join(output_full_dist_dir, img_name)
        os.makedirs(os.path.dirname(save_full_distmap_path),exist_ok=True)
        cv2.imwrite(save_full_distmap_path, full_distmap)
        
        weak_distmap = gen_dist_label(img, vor_img, point_dict)
        save_weak_distmap_path = os.path.join(output_weak_dist_dir, img_name)
        os.makedirs(os.path.dirname(save_weak_distmap_path),exist_ok=True)
        cv2.imwrite(save_weak_distmap_path, weak_distmap)
        
        # generate the full label and heatmap
        full_label, full_heat = gen_full_label_with_semantic(semantic)
        save_full_label_path = os.path.join(output_full_label_dir, img_name)
        os.makedirs(os.path.dirname(save_full_label_path), exist_ok=True)
        cv2.imwrite(save_full_label_path, full_label)
        save_full_heat_path = os.path.join(output_full_heatmap_dir, img_name)
        os.makedirs(os.path.dirname(save_full_heat_path), exist_ok=True)
        cv2.imwrite(save_full_heat_path, full_heat)
        
        # generate the weak label and heatmap
        weak_label, instance_dict, weak_label_valid, weak_heat = gen_weak_label_with_point_fb(img, point_dict, superpixel, vor_img.copy())
        save_weak_label_path = os.path.join(output_weak_label_dir, img_name)
        os.makedirs(os.path.dirname(save_weak_label_path),exist_ok=True)
        cv2.imwrite(save_weak_label_path, weak_label)
        save_weak_heat_path = os.path.join(output_weak_heatmap_dir, img_name)
        os.makedirs(os.path.dirname(save_weak_heat_path),exist_ok=True)
        cv2.imwrite(save_weak_heat_path, weak_heat)
    pass

def gen_split_file(preprocess_dir, output_dir):
    # origin background points
    split_dict = {'train':[], 'val':[]}
    img_dir = os.path.join(preprocess_dir, 'images')
    img_names = sorted(os.listdir(img_dir))
    val_list = random.sample(img_names, 0)
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(preprocess_dir,'gts', img_name.replace('.png','.tiff'))
        full_label_path = os.path.join(preprocess_dir,'labels/full', img_name)
        full_heat_path = os.path.join(preprocess_dir,'labels/full_heatmap', img_name)
        full_dist_path = os.path.join(preprocess_dir,'labels/full_distmap', img_name)
        weak_label_path = os.path.join(preprocess_dir,'labels/weak', img_name)
        weak_heat_path = os.path.join(preprocess_dir,'labels/weak_heatmap', img_name)
        weak_dist_path = os.path.join(preprocess_dir,'labels/weak_distmap', img_name)
        semantic_path = os.path.join(preprocess_dir,'semantics', img_name)
        point_path = os.path.join(preprocess_dir,'points', img_name.replace('.png','.json'))
        train_item = {
            "img_path": img_path,
            "gt_path": gt_path,
            "full_label_path": full_label_path,
            "full_heat_path": full_heat_path,
            "full_dist_path": full_dist_path,
            "weak_label_path": weak_label_path,
            "weak_heat_path": weak_heat_path,
            "weak_dist_path": weak_dist_path,
            "semantic_path": semantic_path,
            "point_path": point_path,
        }
        if img_name not in val_list:
            split_dict['train'].append(train_item)
        else:
            split_dict['val'].append(train_item)
            
    save_path = os.path.join(output_dir,'train_tnbc_val_tnbc_iter0.json')
    json.dump(split_dict, open(save_path, 'w'), indent=2)
        
    # fuse background points
    split_dict = {'train':[], 'val':[]}
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(preprocess_dir,'gts', img_name.replace('.png','.tiff'))
        full_label_path = os.path.join(preprocess_dir,'labels_fuse/full', img_name)
        full_heat_path = os.path.join(preprocess_dir,'labels_fuse/full_heatmap', img_name)
        full_dist_path = os.path.join(preprocess_dir,'labels_fuse/full_distmap', img_name)
        weak_label_path = os.path.join(preprocess_dir,'labels_fuse/weak', img_name)
        weak_heat_path = os.path.join(preprocess_dir,'labels_fuse/weak_heatmap', img_name)
        weak_dist_path = os.path.join(preprocess_dir,'labels_fuse/weak_distmap', img_name)
        semantic_path = os.path.join(preprocess_dir,'semantics', img_name)
        point_path = os.path.join(preprocess_dir,'points_fuse', img_name.replace('.png','.json'))
        train_item = {
            "img_path": img_path,
            "gt_path": gt_path,
            "full_label_path": full_label_path,
            "full_heat_path": full_heat_path,
            "full_dist_path": full_dist_path,
            "weak_label_path": weak_label_path,
            "weak_heat_path": weak_heat_path,
            "weak_dist_path": weak_dist_path,
            "semantic_path": semantic_path,
            "point_path": point_path,
        }
        if img_name not in val_list:
            split_dict['train'].append(train_item)
        else:
            split_dict['val'].append(train_item)

    save_path = os.path.join(output_dir,'train_tnbc_val_tnbc_iter0_fuse.json')
    json.dump(split_dict, open(save_path, 'w'), indent=2)

def gen_split_file_fine_grained(preprocess_dir, output_dir, img_list, save_type='bf'):
    # origin background points
    split_dict = {'train':[], 'val':[]}
    img_dir = os.path.join(preprocess_dir, 'images')
    img_names = sorted(img_list)
    val_list = random.sample(img_names, round(0.2*len(img_names)))
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(preprocess_dir,'gts', img_name.replace('.png','_label.tiff'))
        full_label_path = os.path.join(preprocess_dir,'labels/full', img_name)
        full_heat_path = os.path.join(preprocess_dir,'labels/full_heatmap', img_name)
        full_dist_path = os.path.join(preprocess_dir,'labels/full_distmap', img_name)
        weak_label_path = os.path.join(preprocess_dir,'labels/weak', img_name)
        weak_heat_path = os.path.join(preprocess_dir,'labels/weak_heatmap', img_name)
        weak_dist_path = os.path.join(preprocess_dir,'labels/weak_distmap', img_name)
        semantic_path = os.path.join(preprocess_dir,'semantics', img_name)
        point_path = os.path.join(preprocess_dir,'points', img_name.replace('.png','.json'))
        train_item = {
            "img_path": img_path,
            "gt_path": gt_path,
            "full_label_path": full_label_path,
            "full_heat_path": full_heat_path,
            "full_dist_path": full_dist_path,
            "weak_label_path": weak_label_path,
            "weak_heat_path": weak_heat_path,
            "weak_dist_path": weak_dist_path,
            "semantic_path": semantic_path,
            "point_path": point_path,
        }
        if img_name not in val_list:
            split_dict['train'].append(train_item)
        else:
            split_dict['val'].append(train_item)
            
    save_path = os.path.join(output_dir,'train_{}_val_{}_iter0.json'.format(save_type, save_type))
    json.dump(split_dict, open(save_path, 'w'), indent=2)
        
    # fuse background points
    split_dict = {'train':[], 'val':[]}
    for img_name in img_names:
        img_path = os.path.join(img_dir, img_name)
        gt_path = os.path.join(preprocess_dir,'gts', img_name.replace('.png','_label.tiff'))
        full_label_path = os.path.join(preprocess_dir,'labels_fuse/full', img_name)
        full_heat_path = os.path.join(preprocess_dir,'labels_fuse/full_heatmap', img_name)
        full_dist_path = os.path.join(preprocess_dir,'labels_fuse/full_distmap', img_name)
        weak_label_path = os.path.join(preprocess_dir,'labels_fuse/weak', img_name)
        weak_heat_path = os.path.join(preprocess_dir,'labels_fuse/weak_heatmap', img_name)
        weak_dist_path = os.path.join(preprocess_dir,'labels_fuse/weak_distmap', img_name)
        semantic_path = os.path.join(preprocess_dir,'semantics', img_name)
        point_path = os.path.join(preprocess_dir,'points_fuse', img_name.replace('.png','.json'))
        train_item = {
            "img_path": img_path,
            "gt_path": gt_path,
            "full_label_path": full_label_path,
            "full_heat_path": full_heat_path,
            "full_dist_path": full_dist_path,
            "weak_label_path": weak_label_path,
            "weak_heat_path": weak_heat_path,
            "weak_dist_path": weak_dist_path,
            "semantic_path": semantic_path,
            "point_path": point_path,
        }
        if img_name not in val_list:
            split_dict['train'].append(train_item)
        else:
            split_dict['val'].append(train_item)

    save_path = os.path.join(output_dir,'train_{}_val_{}_iter0_fuse.json'.format(save_type, save_type))
    json.dump(split_dict, open(save_path, 'w'), indent=2)


if __name__=="__main__":
    data_root='./data/TNBC'
    img_dir = './data/TNBC/images'
    gt_dir = './data/TNBC/gts'
    # recollect_files(data_root, img_dir, gt_dir)
    output_dir = './data/TNBC'
    preprocess_TNBC(img_dir, gt_dir, output_dir, train=True)
    
    preprocess_dir = './data/TNBC'
    output_dir = './data/TNBC'
    change_bg_point(preprocess_dir, output_dir)

    # preprocess_dir = './data/TNBC'
    # output_dir = './data/splits'
    # gen_split_file(preprocess_dir, output_dir)
    