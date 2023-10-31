import cv2
from cv2 import ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE
import numpy as np
import random
import math
import torch
from skimage import segmentation, morphology, measure

def random_crop(img, anno, heat, vor, crop_size=(256,256), point_dict=None):
    H,W,C = img.shape
    H_range = H-crop_size[0]
    W_range = W-crop_size[1]
    h_start = random.randint(0,H_range)
    w_start = random.randint(0,W_range)
    img = img[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1], :]
    anno = anno[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    heat = heat[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    vor = vor[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    new_point_dict = None
    if point_dict is not None:
        new_point_dict = {"background":{}, "foreground":{}}
        for p_idx, p_prop in point_dict['background'].items():
            x = round(p_prop[0] - w_start)
            y = round(p_prop[1] - h_start)
            bg_valid = x>=0 and x<crop_size[0] and y>=0 and y<crop_size[1]
            if bg_valid:
                new_point_dict['background'][p_idx] = [x,y]
            else:
                continue
        for p_idx, p_prop in point_dict['foreground'].items():
            select_x = round(p_prop['select_point'][0] - w_start)
            select_y = round(p_prop['select_point'][1] - h_start)
            bnd_x = round(p_prop['boundary_point'][0] - w_start)
            bnd_y = round(p_prop['boundary_point'][1] -h_start)
            select_valid = select_x>=0 and select_x<crop_size[0] and select_y>=0 and select_y<crop_size[1]
            bnd_valid = bnd_x>=0 and bnd_x<crop_size[0] and bnd_y>=0 and bnd_y<crop_size[1]
            if select_valid and bnd_valid:
                new_point_dict['foreground'][p_idx] = {'select_point':[select_x, select_y], 'boundary_point':[bnd_x, bnd_y]}
            else:
                continue
            
    return img, anno, heat, vor, new_point_dict

def random_scale(img, anno, heat, vor, scale_range=(0.5,2.0), crop_size=(256,256), point_dict=None):
    H,W,C = img.shape
    # gen_scale_ratio
    scale = random.random()
    scale = scale*(scale_range[1]-scale_range[0]) + scale_range[0]
    # calculate the min scale_ratio
    h_dest = round(H*scale)
    w_dest = round(W*scale)
    img_temp = cv2.resize(img,(w_dest,h_dest),interpolation=cv2.INTER_LINEAR)
    anno_temp = cv2.resize(anno,(w_dest,h_dest),interpolation=cv2.INTER_NEAREST)
    heat_temp = cv2.resize(heat,(w_dest,h_dest),interpolation=cv2.INTER_LINEAR)
    vor_temp = cv2.resize(vor,(w_dest,h_dest),interpolation=cv2.INTER_NEAREST)
    h_temp,w_temp,_ = img_temp.shape
    h_pad = max(0,crop_size[0]-h_temp)
    w_pad = max(0,crop_size[1]-w_temp)
    img = cv2.copyMakeBorder(img_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
    anno = cv2.copyMakeBorder(anno_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
    heat = cv2.copyMakeBorder(heat_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
    vor = cv2.copyMakeBorder(vor_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
    if point_dict is not None:
        for p_idx, p_prop in point_dict['background'].items():
            p_prop = [p_prop[0]*scale,p_prop[1]*scale]
        for p_idx, p_prop in point_dict['foreground'].items():
            p_prop['select_point'] = [p_prop['select_point'][0]*scale, p_prop['select_point'][1]*scale]
            p_prop['boundary_point'] = [p_prop['boundary_point'][0]*scale, p_prop['boundary_point'][1]*scale]
    return img, anno, heat, vor, point_dict

def random_flip(img, anno, heat, vor, flip_ratio=0.5, point_dict=None):
    H,W,C = img.shape
    h_flip = random.random()
    w_flip = random.random()
    if h_flip < flip_ratio:
        img = cv2.flip(img, 1)
        anno = cv2.flip(anno, 1)
        heat = cv2.flip(heat, 1)
        vor = cv2.flip(vor, 1)
    if w_flip < flip_ratio:
        img = cv2.flip(img, 0)
        anno = cv2.flip(anno, 0)
        heat = cv2.flip(heat, 0)
        vor = cv2.flip(vor, 0)
    if point_dict is not None:
        if h_flip < flip_ratio:
            for p_idx, p_prop in point_dict['background'].items():
                p_prop = [W - p_prop[0] - 1, p_prop[1]]
            for p_idx, p_prop in point_dict['foreground'].items():
                p_prop['select_point'] = [W - p_prop['select_point'][0] - 1, p_prop['select_point'][1]]
                p_prop['boundary_point'] = [W - p_prop['boundary_point'][0] - 1, p_prop['boundary_point'][1]]
            
        if w_flip < flip_ratio:
            for p_idx, p_prop in point_dict['background'].items():
                p_prop = [p_prop[0], H - p_prop[1] - 1]
            for p_idx, p_prop in point_dict['foreground'].items():
                p_prop['select_point'] = [p_prop['select_point'][0], H - p_prop['select_point'][1] - 1]
                p_prop['boundary_point'] = [p_prop['boundary_point'][0], H - p_prop['boundary_point'][1] - 1]
    return img, anno, heat, vor, point_dict

def random_rotate(img, anno, heat, vor, point_dict=None):
    H,W,C = img.shape
    rot_id = random.randint(0,3)
    if rot_id == 0:
        pass
    elif rot_id == 1:
        img = cv2.rotate(img,ROTATE_90_CLOCKWISE)
        anno = cv2.rotate(anno,ROTATE_90_CLOCKWISE)
        heat = cv2.rotate(heat,ROTATE_90_CLOCKWISE)
        vor = cv2.rotate(vor,ROTATE_90_CLOCKWISE)
    elif rot_id == 2:
        img = cv2.rotate(img,ROTATE_180)
        anno = cv2.rotate(anno,ROTATE_180)
        heat = cv2.rotate(heat,ROTATE_180)
        vor = cv2.rotate(vor,ROTATE_180)
    elif rot_id == 3:
        img = cv2.rotate(img,ROTATE_90_COUNTERCLOCKWISE)
        anno = cv2.rotate(anno,ROTATE_90_COUNTERCLOCKWISE)
        heat = cv2.rotate(heat,ROTATE_90_COUNTERCLOCKWISE)
        vor = cv2.rotate(vor,ROTATE_180)
    if point_dict is not None:
        for p_idx, p_prop in point_dict['background'].items():
            if rot_id == 0:
                pass
            elif rot_id == 1:
                p_prop = [H - p_prop[1] - 1, p_prop[0]]
            elif rot_id == 2:
                p_prop = [W - p_prop[0] - 1, H - p_prop[1] - 1]
            elif rot_id == 3:
                p_prop = [p_prop[1], W - p_prop[0] - 1]
        for p_idx, p_prop in point_dict['foreground'].items():
            if rot_id == 0:
                pass
            elif rot_id == 1:
                p_prop['select_point'] = [H - p_prop['select_point'][1] - 1, p_prop['select_point'][0]]
                p_prop['boundary_point'] = [H - p_prop['boundary_point'][1] - 1, p_prop['boundary_point'][0]]
            elif rot_id == 2:
                p_prop['select_point'] = [W - p_prop['select_point'][0] - 1, H - p_prop['select_point'][1] - 1]
                p_prop['boundary_point'] = [W - p_prop['boundary_point'][0] - 1, H - p_prop['boundary_point'][1] - 1]
            elif rot_id == 3:
                p_prop['select_point'] = [p_prop['select_point'][1], W - p_prop['select_point'][0] - 1]
                p_prop['boundary_point'] = [p_prop['boundary_point'][1], W - p_prop['boundary_point'][0] - 1]
            
    return img, anno, heat, vor, point_dict

def multi_scale_test(img, anno, scale=[0.5,1.0,1.5], crop_size=(512,512)):
    img_list= []
    anno_list = []
    valid_list = []
    H,W,C = img.shape
    for s in scale:
        img_temp = cv2.resize(img,(int(W*s), int(H*s)), interpolation=cv2.INTER_LINEAR)
        anno_temp = cv2.resize(anno,(int(W*s), int(H*s)), interpolation=cv2.INTER_NEAREST)
        h_temp,w_temp,_ = img_temp.shape
        h_pad = max(0,crop_size[0]-h_temp)
        w_pad = max(0,crop_size[1]-w_temp)
        valid_region = (h_temp,w_temp)
        img_temp = cv2.copyMakeBorder(img_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=(0,0,0))
        anno_temp = cv2.copyMakeBorder(anno_temp, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0)
        img_list.append(img_temp)
        anno_list.append(anno_temp)
        valid_list.append(valid_region)
    return img_list, anno_list, valid_list

def random_subcrop(img, min_ratio=0.25): # for contrast learning
    # input type: Tensor
    C,H,W = img.shape
    crop_h = round(H/2)
    crop_w = round(W/2)
    overlap_ratio = 0
    while overlap_ratio < min_ratio:
        crop1_y = random.randint(0,H-crop_h)
        crop1_x = random.randint(0,W-crop_w)
        crop2_y = random.randint(0,H-crop_h)
        crop2_x = random.randint(0,W-crop_w)
        overlap_h = 2*crop_h - (max(crop1_y,crop2_y) - min(crop1_y,crop2_y))
        overlap_w = 2*crop_w - (max(crop1_x,crop2_x) - min(crop1_x,crop2_x))
        overlap_s = overlap_h*overlap_w
        overlap_ratio = overlap_s/(crop_h*crop_w)
    # crop the image
    crop1 = img[:, crop1_y:crop1_y+crop_h, crop1_x:crop1_x+crop_w]
    crop2 = img[:, crop2_y:crop2_y+crop_h, crop2_x:crop2_x+crop_w]
    # calculate the coordinate of the overlapping area
    overlap_y1 = max(crop1_y, crop2_y) - crop_h
    overlap_x1 = max(crop1_x, crop2_x) - crop_w
    overlap_y2 = min(crop1_y, crop2_y) + crop_h
    overlap_x2 = min(crop1_y, crop2_y) + crop_w
    # get relative coordinate
    overlap1_y1 = overlap_y1 - crop1_y
    overlap1_x1 = overlap_x1 - crop1_x
    overlap1_y2 = overlap_y2 - crop1_y
    overlap1_x2 = overlap_x2 - crop1_x
    
    overlap2_y1 = overlap_y1 - crop2_y
    overlap2_x1 = overlap_x1 - crop2_x
    overlap2_y2 = overlap_y2 - crop2_y
    overlap2_x2 = overlap_x2 - crop2_x
    
    return crop1, crop2, [(overlap1_y1,overlap1_x1),(overlap1_y2,overlap1_x2)], [(overlap2_y1,overlap2_x1),(overlap2_y2,overlap2_x2)]

def pre_point_dict(point_dict):
    if point_dict is not None:
        new_point_dict = {'background':{}, 'foreground':{}}
        for p_idx, p_prop in point_dict['background'].items():
            new_point_dict['background'][p_idx] = [p_prop['x'], p_prop['y']]
        for p_idx, p_prop in point_dict['foreground'].items():
            new_point_dict['foreground'][p_idx] = {'select_point': p_prop['select_point'], 'boundary_point':p_prop['boundary_point']}
        return new_point_dict
    else:
        return None
    
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

def gen_deg_n_dist(img, point_dict, neighbour=0):
    H,W = img.shape[:2]
    deg_map = np.random.rand(H,W)*360
    # deg_map = np.zeros(img.shape[1:])
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            bnd_point = inst_prop['boundary_point']
            deg, dist = get_degree_n_distance(fg_point, bnd_point)
            deg_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = deg
            dist_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = dist
            dist_mask[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            deg_map[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 0
            dist_map[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 0
            dist_mask[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 1

    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v2(img, point_dict, weak_label, neighbour = 0):
    H,W = img.shape[:2]
    deg_map = np.random.rand(H,W)*360
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])

    valid_mask = np.ones((H,W))
    valid_mask[weak_label==255] = 0
    temp_label = np.zeros((H,W))
    temp_label[weak_label==1] = 1
    temp_label = measure.label(temp_label, background=0)
    temp_label = morphology.dilation(temp_label, morphology.disk(1))
    dilated_label = morphology.dilation(temp_label, morphology.disk(1))
    boundary_label = dilated_label - temp_label
    boundary_label = boundary_label*valid_mask
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            rand_num = random.random()
            fg_point = inst_prop['select_point']
            cell_idx = temp_label[round(fg_point[1]), round(fg_point[0])]
            # assert cell_idx != 0
            if cell_idx == 0:
                bnd_point = inst_prop['boundary_point']
            else:
                # sample a pixel from the boundary or use the boundary
                bnd_points = np.array(np.where(boundary_label==cell_idx)).T
                bnd_points_num = bnd_points.shape[0]
                if rand_num < 0.5 and bnd_points_num>0:
                    bnd_point_idx = random.sample(range(bnd_points_num), 1)[0]
                    bnd_point = [bnd_points[bnd_point_idx][1], bnd_points[bnd_point_idx][0]]
                else:
                    bnd_point = inst_prop['boundary_point']
                
            deg, dist = get_degree_n_distance(fg_point, bnd_point)
            deg_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = deg
            dist_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = dist
            dist_mask[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            # deg_map[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour] = 0
            dist_map[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 0
            dist_mask[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v3(img, point_dict, weak_label, neighbour = 0):
    H,W = img.shape[:2]
    deg_map = np.random.rand(H,W)*360
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])

    valid_mask = np.ones((H,W))
    valid_mask[weak_label==255] = 0
    cell_label = np.zeros((H,W))
    cell_label[weak_label==1] = 1
    cell_label = measure.label(cell_label)
    dilated_label = morphology.dilation(cell_label, morphology.disk(1))
    boundary_label = dilated_label - cell_label
    boundary_label = boundary_label*valid_mask
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            cell_idx = cell_label[round(fg_point[1]), round(fg_point[0])]
            # assert cell_idx != 0
            if cell_idx == 0:
                bnd_point = inst_prop['boundary_point']
                deg, dist = get_degree_n_distance(fg_point, bnd_point)
                deg_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = deg
                dist_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = dist
                dist_mask[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = 1

            else:
                # sample N pairs of fg_points and bnd_points
                # sample fg point
                fg_points = np.array(np.where(cell_label==cell_idx)).T
                fg_points_num = fg_points.shape[0]
                bnd_points = np.array(np.where(boundary_label==cell_idx)).T
                bnd_points_num = bnd_points.shape[0]
                for i in range(fg_points_num):
                    sample_fg = [fg_points[i][1], fg_points[i][0]]
                    # sample bnd point
                    if bnd_points_num>0:
                        bnd_point_idx = random.sample(range(bnd_points_num), 1)[0]
                        sample_bnd = [bnd_points[bnd_point_idx][1], bnd_points[bnd_point_idx][0]]
                    else:
                        sample_bnd = inst_prop['boundary_point']
                    deg, dist = get_degree_n_distance(sample_fg, sample_bnd)
                    deg_map[int(fg_point[1]), int(fg_point[0])] = deg
                    dist_map[int(fg_point[1]), int(fg_point[0])] = dist
                    dist_mask[int(fg_point[1]), int(fg_point[0])] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            # deg_map[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour] = 0
            dist_map[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 0
            dist_mask[int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1, int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v5(img, point_dict, rand_init=True):
    H,W = img.shape[:2]
    inst_map = np.zeros((H,W))
    if rand_init==True:
        deg_map = np.random.rand(H,W)*360
    else:
        deg_map = np.ones((H,W))*(-360)
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            bnd_point = inst_prop['boundary_point']
            
            deg, dist = get_degree_n_distance(fg_point, bnd_point)
            start_point = [round(3/2*fg_point[0] - 1/2*bnd_point[0]), round(3/2*fg_point[1] - 1/2*bnd_point[1])]
            stop_point = [round(bnd_point[0]), round(bnd_point[1])]
            inst_map = cv2.line(inst_map, start_point, stop_point, thickness=2, color=int(inst_idx))
            inst_map[int(fg_point[1]), int(fg_point[0])] = int(inst_idx)
            
            deg_map[inst_map==int(inst_idx)] = int(deg)
            anno_points = np.array(np.where(inst_map==int(inst_idx))).T # get the points as (y,x)
            anno_points = anno_points[:,[1,0]]  # switch to (x,y)
            for anno_point in anno_points:
                deg, dist = get_degree_n_distance(anno_point, bnd_point)
                dist_map[anno_point[1], anno_point[0]] = dist
            dist_mask[inst_map==int(inst_idx)] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            deg_map[int(point_prop[0]), int(point_prop[1])] = random.randint(0,359)
            dist_map[int(point_prop[0]), int(point_prop[1])] = 0
            dist_mask[int(point_prop[0]), int(point_prop[1])] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v6(img, point_dict):
    H,W = img.shape[:2]
    inst_map = np.zeros((H,W))
    deg_map = np.ones((H,W))*-360
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            bnd_point = inst_prop['boundary_point']
            
            deg, dist = get_degree_n_distance(fg_point, bnd_point)
            start_point = [round(3/2*fg_point[0] - 1/2*bnd_point[0]), round(3/2*fg_point[1] - 1/2*bnd_point[1])]
            inst_map = cv2.line(inst_map, start_point, bnd_point, thickness=1, color=int(inst_idx))
            inst_map[int(fg_point[1]), int(fg_point[0])] = int(inst_idx)
            
            deg_map[inst_map==int(inst_idx)] = int(deg)
            anno_points = np.array(np.where(inst_map==int(inst_idx))).T # get the points as (y,x)
            anno_points = anno_points[:,[1,0]]  # switch to (x,y)
            for anno_point in anno_points:
                deg, dist = get_degree_n_distance(anno_point, bnd_point)
                dist_map[anno_point[1], anno_point[0]] = dist
            dist_mask[inst_map==int(inst_idx)] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            deg_map[int(point_prop[0]), int(point_prop[1])] = random.randint(0,359)
            dist_map[int(point_prop[0]), int(point_prop[1])] = 0
            dist_mask[int(point_prop[0]), int(point_prop[1])] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v7(img, point_dict, weak_label, use_pesudo, neighbour=0):
    H,W = img.shape[:2]
    deg_map = np.random.rand(H,W)*360
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])
    
    if point_dict is not None:
        valid_mask = np.ones((H,W))
        valid_mask[weak_label==255] = 0
        cell_label = np.zeros((H,W))
        cell_label[weak_label==1] = 1
        cell_label = measure.label(cell_label)
        dilated_label = morphology.dilation(cell_label, morphology.disk(1))
        boundary_label = dilated_label - cell_label
        boundary_label = boundary_label*valid_mask
        
        if use_pesudo == True:
            dist_mask = np.ones(img.shape[:2])
            cell_num = np.max(cell_label)
            for cell_idx in range(1, cell_num+1):
                fg_points = np.array(np.where(cell_label==cell_idx)).T
                fg_points_num = fg_points.shape[0]
                bnd_points = np.array(np.where(boundary_label==cell_idx)).T
                bnd_points_num = bnd_points.shape[0]
                for i in range(fg_points_num):
                    sample_fg = [fg_points[i][1], fg_points[i][0]]
                    # sample bnd point
                    bnd_point_idx = random.sample(range(bnd_points_num), 1)[0]
                    sample_bnd = [bnd_points[bnd_point_idx][1], bnd_points[bnd_point_idx][0]]
                    deg, dist = get_degree_n_distance(sample_fg, sample_bnd)
                    deg_map[int(fg_points[i][0]), int(fg_points[i][1])] = deg
                    dist_map[int(fg_points[i][0]), int(fg_points[i][1])] = dist
        else:
            fg_points_dict = point_dict['foreground']
            for inst_idx, inst_prop in fg_points_dict.items():
                fg_point = inst_prop['select_point']
                cell_idx = cell_label[round(fg_point[1]), round(fg_point[0])]
                if cell_idx == 0:
                    bnd_point = inst_prop['boundary_point']
                    deg, dist = get_degree_n_distance(fg_point, bnd_point)
                    deg_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = deg
                    dist_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = dist
                    dist_mask[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = 1
                else:
                    # sample N pairs of fg_points and bnd_points
                    # sample fg point
                    fg_points = np.array(np.where(cell_label==cell_idx)).T      #[y,x]
                    fg_points_num = fg_points.shape[0]
                    bnd_points = np.array(np.where(boundary_label==cell_idx)).T
                    bnd_points_num = bnd_points.shape[0]
                    for i in range(fg_points_num):
                        sample_fg = [fg_points[i][1], fg_points[i][0]]
                        # sample bnd point
                        if bnd_points_num>0:
                            bnd_point_idx = random.sample(range(bnd_points_num), 1)[0]
                            sample_bnd = [bnd_points[bnd_point_idx][1], bnd_points[bnd_point_idx][0]]
                        else:
                            sample_bnd = inst_prop['boundary_point']
                        deg, dist = get_degree_n_distance(sample_fg, sample_bnd)
                        deg_map[int(fg_points[i][0]), int(fg_points[i][1])] = deg
                        dist_map[int(fg_points[i][0]), int(fg_points[i][1])] = dist
                        dist_mask[int(fg_points[i][0]), int(fg_points[i][1])] = 1

            bg_points_dict = point_dict['background']
            for point_idx, point_prop in bg_points_dict.items():
                # deg_map[int(point_prop[1])-neighbour: int(point_prop[1])+neighbour, int(point_prop[0])-neighbour: int(point_prop[0])+neighbour] = 0
                dist_map[int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1, int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1] = 0
                dist_mask[int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1, int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v8(img, point_dict, weak_label, use_pesudo, neighbour=0):
    H,W = img.shape[:2]
    deg_map = np.random.rand(H,W)*360
    dist_map = np.zeros(img.shape[:2])
    dist_mask = np.zeros(img.shape[:2])

    valid_mask = np.ones((H,W))
    valid_mask[weak_label==255] = 0
    cell_label = np.zeros((H,W))
    cell_label[weak_label==1] = 1
    cell_label = measure.label(cell_label)
    dilated_label = morphology.dilation(cell_label, morphology.disk(1))
    boundary_label = dilated_label - cell_label
    boundary_label = boundary_label*valid_mask
    
    if use_pesudo == True:
        degree = random.randint(0,359)
        deg = np.ones((H,W))*degree
        dist_map = gen_distmap(weak_label, degree)
        dist_mask = np.ones(img.shape[:2])
    else:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            cell_idx = cell_label[round(fg_point[1]), round(fg_point[0])]
            if cell_idx == 0:
                bnd_point = inst_prop['boundary_point']
                deg, dist = get_degree_n_distance(fg_point, bnd_point)
                deg_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = deg
                dist_map[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = dist
                dist_mask[int(fg_point[1])-neighbour: int(fg_point[1])+neighbour+1, int(fg_point[0])-neighbour:int(fg_point[0])+neighbour+1] = 1
            else:
                # sample N pairs of fg_points and bnd_points
                # sample fg point
                fg_points = np.array(np.where(cell_label==cell_idx)).T      #[y,x]
                fg_points_num = fg_points.shape[0]
                bnd_points = np.array(np.where(boundary_label==cell_idx)).T
                bnd_points_num = bnd_points.shape[0]
                for i in range(fg_points_num):
                    sample_fg = [fg_points[i][1], fg_points[i][0]]
                    # sample bnd point
                    if bnd_points_num>0:
                        bnd_point_idx = random.sample(range(bnd_points_num), 1)[0]
                        sample_bnd = [bnd_points[bnd_point_idx][1], bnd_points[bnd_point_idx][0]]
                    else:
                        sample_bnd = inst_prop['boundary_point']
                    deg, dist = get_degree_n_distance(sample_fg, sample_bnd)
                    deg_map[int(fg_points[i][0]), int(fg_points[i][1])] = deg
                    dist_map[int(fg_points[i][0]), int(fg_points[i][1])] = dist
                    dist_mask[int(fg_points[i][0]), int(fg_points[i][1])] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            # deg_map[int(point_prop[1])-neighbour: int(point_prop[1])+neighbour, int(point_prop[0])-neighbour: int(point_prop[0])+neighbour] = 0
            dist_map[int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1, int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1] = 0
            dist_mask[int(point_prop[1])-neighbour: int(point_prop[1])+neighbour+1, int(point_prop[0])-neighbour: int(point_prop[0])+neighbour+1] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_deg_n_dist_v9(img, point_dict):
    H,W = img.shape[:2]
    inst_map = np.zeros((H,W))
    deg_map = np.zeros((H,W))
    dist_map = np.zeros((8,H,W))
    dist_mask = np.zeros((8,H,W))
    
    degree_list = np.array([0, 45, 90, 135, 180, 225, 270, 315, 360])
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            bnd_point = inst_prop['boundary_point']
            
            deg, dist = get_degree_n_distance(fg_point, bnd_point)
            sorted_deg_idx = np.argsort(np.abs(degree_list-deg))  # obtain the sorted index of the deg
            for deg_idx in sorted_deg_idx[:2]:
                if deg_idx == 8:
                    deg_idx = 0
                dist_map[deg_idx, round(fg_point[1]), round(fg_point[0])] = dist
                dist_mask[deg_idx, round(fg_point[1]), round(fg_point[0])] = 1

        bg_points_dict = point_dict['background']
        for point_idx, point_prop in bg_points_dict.items():
            dist_map[:, int(point_prop[0]), int(point_prop[1])] = 0
            dist_mask[:, int(point_prop[0]), int(point_prop[1])] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def get_8_coordinates(c_x, c_y, pos_mask_contour):
    ct = pos_mask_contour[:, :]
    x = torch.tensor(ct[:, 0] - c_x)
    y = torch.tensor(ct[:, 1] - c_y)
    angle = torch.atan2(x, y) * 180 / np.pi
    angle[angle < 0] += 360
    angle = angle.int()
    dist = torch.sqrt(x ** 2 + y ** 2)
    angle, idx = torch.sort(angle)
    dist = dist[idx]
    
    #生成8个角度
    new_coordinate = {}
    for i in range(0, 360, 45):
        if i in angle:
            d = dist[angle==i].max()
            new_coordinate[i] = max(1e-2, d)
        elif i + 1 in angle:
            d = dist[angle == i+1].max()
            new_coordinate[i] = max(1e-2, d)
        elif i - 1 in angle:
            d = dist[angle == i-1].max()
            new_coordinate[i] = max(1e-2, d)
        elif i + 2 in angle:
            d = dist[angle == i+2].max()
            new_coordinate[i] = max(1e-2, d)
        elif i - 2 in angle:
            d = dist[angle == i-2].max()
            new_coordinate[i] = max(1e-2, d)
        elif i + 3 in angle:
            d = dist[angle == i+3].max()
            new_coordinate[i] = max(1e-2, d)
        elif i - 3 in angle:
            d = dist[angle == i-3].max()
            new_coordinate[i] = max(1e-2, d)
        elif i + 4 in angle:
            d = dist[angle == i+4].max()
            new_coordinate[i] = max(1e-2, d)
        elif i - 4 in angle:
            d = dist[angle == i-4].max()
            new_coordinate[i] = max(1e-2, d)
        elif i + 5 in angle:
            d = dist[angle == i+5].max()
            new_coordinate[i] = max(1e-2, d)
        elif i - 5 in angle:
            d = dist[angle == i-5].max()
            new_coordinate[i] = max(1e-2, d)

    distances = np.zeros(8)

    for a in range(0, 360, 45):
        if a not in new_coordinate.keys():
            new_coordinate[a] = 1e-2
            distances[a//45] = 1e-2
        else:
            distances[a//45] = new_coordinate[a]

    return distances, new_coordinate

def gen_deg_n_dist_v10(img, point_dict, vor):
    H,W = img.shape[:2]
    inst_map = np.zeros((H,W))
    deg_map = np.zeros((H,W))
    dist_map = np.zeros((8,H,W))
    dist_mask = np.zeros((8,H,W))
    seg = measure.label(vor)
    bnd = segmentation.find_boundaries(vor, mode='inner')*1
    bnd = morphology.dilation(bnd, morphology.disk(1))
    bnd[0, :] = 1
    bnd[-1,:] = 1
    bnd[:, 0] = 1
    bnd[:,-1] = 1
    bnd = seg*bnd
    
    if point_dict is not None:
        fg_points_dict = point_dict['foreground']
        for inst_idx, inst_prop in fg_points_dict.items():
            fg_point = inst_prop['select_point']
            inst_value = seg[fg_point[1], fg_point[0]]
            bnd_points = np.array(np.where(bnd==inst_value)).T
            bnd_points = bnd_points[:,[1,0]]
            distances, new_coordinates = get_8_coordinates(fg_point[0], fg_point[1], bnd_points)
            dist_map[:, fg_point[0], fg_point[1]] = distances
            dist_mask[:, fg_point[0], fg_point[1]] = 1

        # bg_points_dict = point_dict['background']
        # for point_idx, point_prop in bg_points_dict.items():
        #     dist_map[:, int(point_prop[0]), int(point_prop[1])] = 0
        #     dist_mask[:, int(point_prop[0]), int(point_prop[1])] = 1
            
    degree_dict = {'deg_map': deg_map, 'dist_map':dist_map, 'dist_mask': dist_mask}
    return degree_dict

def gen_distmap(weak_label, degree=0):
    inst_map = measure.label(weak_label)
    cell_mask = np.zeros(inst_map.shape)
    cell_mask[inst_map>0] = 1
    radian = math.radians(degree)
    H,W = cell_mask.shape

    boundary = segmentation.find_boundaries(inst_map, mode='outer')*1
    boundary = morphology.dilation(boundary, morphology.disk(1))
    cell_mask[boundary==1]= 0
    cell_points = np.array(np.where(cell_mask==1)).T
    dist_map = np.zeros((H,W))
    for cell_point in cell_points:
        mask = np.zeros((H,W))
        if math.sin(radian)>=0 and math.cos(radian)>=0:
            margin_x = W-cell_point[1]
            margin_y = H-cell_point[0]
        elif math.sin(radian)>=0 and math.cos(radian)<0:
            margin_x = cell_point[1]
            margin_y = H-cell_point[0]
        elif math.sin(radian)<0 and math.cos(radian)<0:
            margin_x = cell_point[1]
            margin_y = cell_point[0]
        elif math.sin(radian)<0 and math.cos(radian)>=0:
            margin_x = W-cell_point[1]
            margin_y = cell_point[0]
        dx = math.cos(radian)
        dy = math.sin(radian)
        step_x = margin_x/(abs(dx)+1e-8)
        step_y = margin_y/(abs(dy)+1e-8)
        step = min(step_x,step_y)
        dest_x = min(max(0,round(cell_point[1]+ dx*step)),W)
        dest_y = min(max(0,round(cell_point[0]+ dy*step)),H)
        mask = cv2.line(mask,(cell_point[1],cell_point[0]), (dest_x, dest_y), color=1, thickness=1)
        mask = boundary*mask
        bnd_points = np.array(np.where(mask==1)).T
        if len(bnd_points)==0:
            bnd_points = np.array([[dest_y, dest_x]])
        min_dist = np.min(np.linalg.norm(cell_point[np.newaxis,:]-bnd_points, axis=1))
        # min_dist = min([np.linalg.norm(cell_point-bnd_point) for bnd_point in bnd_points])
        dist_map[cell_point[0], cell_point[1]] = min_dist
    return dist_map

def vis_point(img, anno, point_dict, save_path='./1.png'):
    anno [anno==1]=128
    anno_vis = np.stack((anno,anno,anno),axis=-1)
    fg_point_dict = point_dict['foreground']
    bg_point_dict = point_dict['background']
    for inst_idx, inst_prop in fg_point_dict.items():
        fg_point = inst_prop['select_point']
        bnd_point = inst_prop['boundary_point']
        img = cv2.circle(img, (round(fg_point[0]), round(fg_point[1])),radius=2, color=(255,0,0))
        anno_vis = cv2.circle(anno_vis, (round(fg_point[0]), round(fg_point[1])),radius=2, color=(255,0,0))
        img = cv2.circle(img, (round(bnd_point[0]), round(bnd_point[1])),radius=2, color=(0,255,0))
        anno_vis = cv2.circle(anno_vis, (round(bnd_point[0]), round(bnd_point[1])),radius=2, color=(0,255,0))
        
    for p_idx, p_prop in bg_point_dict.items():
        x,y = p_prop
        img = cv2.circle(img, (round(x), round(y)),radius=2, color=(0,0,255))
        anno_vis = cv2.circle(anno_vis, (round(x), round(y)),radius=2, color=(0,0,255))
        
    vis = np.hstack((img,anno_vis))
    cv2.imwrite(save_path, vis)

def gen_counting_map(img, point_dict, kernel_size):
    counting_map = np.zeros(img.shape[:2])
    fg_points_dict = point_dict['foreground']
    for inst_idx, inst_prop in fg_points_dict.items():
        fg_point = inst_prop['select_point']
        counting_map[round(fg_point[1]), round(fg_point[0])] = 1
    counting_map = cv2.GaussianBlur(counting_map, (kernel_size,kernel_size), sigmaX=-1)
    return counting_map

######################## iter 1 ###############################
def img_crop(img, anno, heat, point_dict, pt1, pt2):
    H,W = img.shape[:2]
    w = pt2[0]-pt1[0]
    h = pt2[1]-pt1[1]
    w_start = max(0,round(pt1[0]-w*0.2))
    h_start = max(0,round(pt1[1]-h*0.2))
    w_stop = min(W,round(pt2[0]+w*0.2))
    h_stop = min(H,round(pt2[1]+h*0.2))
    crop_size = [w_stop-w_start, h_stop-h_start]
    img_cropped = img[h_start:h_stop, w_start:w_stop, :]
    anno_cropped = anno[h_start:h_stop, w_start:w_stop]
    heat_cropped = heat[h_start:h_stop, w_start:w_stop]
    
    new_point_dict = None
    if point_dict is not None:
        new_point_dict = {"background":{}, "foreground":{}}
        for p_idx, p_prop in point_dict['background'].items():
            x = round(p_prop[0] - w_start)
            y = round(p_prop[1] - h_start)
            bg_valid = x>=0 and x<crop_size[0] and y>=0 and y<crop_size[1]
            if bg_valid:
                new_point_dict['background'][p_idx] = [x,y]
            else:
                continue
        for p_idx, p_prop in point_dict['foreground'].items():
            select_x = round(p_prop['select_point'][0] - w_start)
            select_y = round(p_prop['select_point'][1] - h_start)
            bnd_x = round(p_prop['boundary_point'][0] - w_start)
            bnd_y = round(p_prop['boundary_point'][1] - h_start)
            select_valid = select_x>=0 and select_x<crop_size[0] and select_y>=0 and select_y<crop_size[1]
            bnd_valid = bnd_x>=0 and bnd_x<crop_size[0] and bnd_y>=0 and bnd_y<crop_size[1]
            if select_valid and bnd_valid:
                new_point_dict['foreground'][p_idx] = {'select_point':[select_x, select_y], 'boundary_point':[bnd_x, bnd_y]}
            else:
                continue
    return img_cropped, anno_cropped, heat_cropped, new_point_dict
    
def img_resize(img, anno, heat, point_dict, crop_size):
    H,W,C = img.shape
    w_ratio = crop_size[0]/W
    h_ratio = crop_size[1]/H
    scale_ratio = min(w_ratio, h_ratio)
    dest_w = round(scale_ratio*W)
    dest_h = round(scale_ratio*H)
    
    img = cv2.resize(img, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
    anno = cv2.resize(anno, (dest_w, dest_h), interpolation=cv2.INTER_NEAREST)
    heat = cv2.resize(heat, (dest_w, dest_h), interpolation=cv2.INTER_LINEAR)
    valid_region = img.shape[:2]
    
    canvs = np.zeros((crop_size[1], crop_size[0], C)).astype(np.float32)
    anno_canvs = np.zeros((crop_size[1], crop_size[0])).astype(np.float32)
    heat_canvs = np.zeros((crop_size[1], crop_size[0])).astype(np.float32)
    canvs[0:dest_h, 0:dest_w, :] = img
    anno_canvs[0:dest_h, 0:dest_w] = anno
    heat_canvs[0:dest_h, 0:dest_w] = heat
    new_point_dict = {'background':{}, 'foreground':{}}
    if point_dict is not None:
        for p_idx, p_prop in point_dict['background'].items():
            new_point_dict['background'][p_idx] = [p_prop[0]*scale_ratio, p_prop[1]*scale_ratio]
            
        for p_idx, p_prop in point_dict['foreground'].items():
            new_point_dict['foreground'][p_idx]= {}
            new_point_dict['foreground'][p_idx]['select_point'] = [p_prop['select_point'][0]*scale_ratio, p_prop['select_point'][1]*scale_ratio]
            new_point_dict['foreground'][p_idx]['boundary_point'] = [p_prop['boundary_point'][0]*scale_ratio, p_prop['boundary_point'][1]*scale_ratio]
    
    return canvs, anno_canvs, heat_canvs, new_point_dict, valid_region