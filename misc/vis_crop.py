import cv2 
import json
import os
import tifffile as tif
from skimage import measure, segmentation, morphology
import numpy as np
import math
import torch

circle_color = [(128,128,0),(128,0,128),(0,128,128),(255,128,0),(128,255,0),(255,0,128),(128,0,255),(0,255,128),(0,128,255)]

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

def vis_circle(img_dir, point_dir, output_dir):
    img_names= os.listdir(img_dir)
    os.makedirs(output_dir,exist_ok=True)
    for ii, img_name in enumerate(sorted(img_names)):
        print('Generating the {}/{} images...'.format(ii, len(img_names)),end='\r')
        img_path = os.path.join(img_dir, img_name)
        point_path = os.path.join(point_dir, img_name.replace('.tif','.json'))
        img = tif.imread(img_path)
        point_dict = json.load(open(point_path,'r'))
        bg_point_dict = point_dict['background']
        fg_point_dict = point_dict['foreground']
        # for point_idx, point_prop in bg_point_dict.items():
        #     x = point_prop['x']
        #     y = point_prop['y']
        #     img = cv2.circle(img,(x,y), radius=2, color=(255,0,0), thickness=1)
        for item_idx, item_prop in fg_point_dict.items():
            select_point = item_prop['select_point']
            bnd_point = item_prop['boundary_point']
            img = cv2.circle(img, select_point, radius=2, color=(0,255,0), thickness=1)
            img = cv2.circle(img, bnd_point, radius=2, color=(0,0,255), thickness=1)
            color_idx = int(item_idx)%len(circle_color)
            # img = cv2.arrowedLine(img, select_point, bnd_point, circle_color[color_idx], thickness=1)
            # _, dist = get_degree_n_distance(select_point, bnd_point)
            # img = cv2.circle(img, select_point, round(dist), circle_color[color_idx], thickness=1)
            save_path = os.path.join(output_dir, img_name.replace('.tif','.png'))
            cv2.imwrite(save_path, img)


def vis_crop(img_dir, weak_dir, weak_heatmap_dir, weak_distmap_dir, point_dict_dir, superpixel_dir, vor_img_dir, circle_dir, kmeans_dir):
    img_name = 'TCGA-18-5592-01Z-00-DX1.png'
    img = cv2.imread(os.path.join(img_dir, img_name))
    weak = cv2.imread(os.path.join(weak_dir, img_name), flags=0)
    weak_heatmap = cv2.imread(os.path.join(weak_heatmap_dir, img_name), flags=0)
    weak_distmap = cv2.imread(os.path.join(weak_distmap_dir, img_name), flags=0)
    point_dict = json.load(open(os.path.join(point_dict_dir, img_name.replace('.png','.json')),'r'))
    superpixel = cv2.imread(os.path.join(superpixel_dir, img_name.replace('.png','_vis.png')))
    vor_img = cv2.imread(os.path.join(vor_img_dir, img_name).replace('.png','.png'),flags=0)
    circle_img = cv2.imread(os.path.join(circle_dir, img_name))
    kmeans_img = cv2.imread(os.path.join(kmeans_dir, img_name))
    
    h_start,w_start = 500,320
    crop_size = 100
    crop_img = img[h_start:h_start+crop_size, w_start:w_start+crop_size, :]
    weak[weak==1]=128
    crop_weak = weak[h_start:h_start+crop_size, w_start:w_start+crop_size]
    crop_weak_heatmap = weak_heatmap[h_start:h_start+crop_size, w_start:w_start+crop_size]
    weak_distmap[weak_distmap==255] = 128
    crop_weak_distmap = weak_distmap[h_start:h_start+crop_size, w_start:w_start+crop_size]
    weak_dist = np.stack((weak_distmap,weak_distmap,weak_distmap),axis=-1)
    fg_point_dict = point_dict['foreground']
    for point_idx, point_prop in fg_point_dict.items():
        select_point = point_prop['select_point']
        weak_dist = cv2.circle(weak_dist, select_point, radius=2, color=(0,255,0), thickness=1)
    crop_weak_dist = weak_dist[h_start:h_start+crop_size, w_start:w_start+crop_size,:]
    crop_sp = superpixel[h_start:h_start+crop_size, w_start:w_start+crop_size,:]
    crop_vor_img = vor_img[h_start:h_start+crop_size, w_start:w_start+crop_size]
    crop_vor_img = np.stack((crop_vor_img,crop_vor_img,crop_vor_img),axis=-1)
    green_canvs = np.zeros_like(crop_img)
    green_canvs[:,:,1]=255
    
    crop_vor_vis = crop_img.copy()
    crop_vor_vis = crop_vor_vis*(1-crop_vor_img/255)+green_canvs*(crop_vor_img/255)
    
    crop_vor_img[crop_vor_img==255]=1
    crop_vor_img[crop_vor_img==0]=255
    crop_vor_img[crop_vor_img==1]=0
    
    crop_circle_img = circle_img[h_start:h_start+crop_size, w_start:w_start+crop_size,:]
    
    H,W,C = kmeans_img.shape
    kmeans_img= kmeans_img[:,round(W/2):,:]
    crop_kmeans_img = kmeans_img[h_start:h_start+crop_size, w_start:w_start+crop_size,:]
    
    cv2.imwrite('1.png', crop_img)
    cv2.imwrite('2.png', crop_weak)
    cv2.imwrite('3.png', crop_weak_heatmap)
    cv2.imwrite('4.png', crop_weak_distmap)
    cv2.imwrite('5.png', crop_weak_dist)
    cv2.imwrite('6.png', crop_sp)
    cv2.imwrite('7.png', crop_vor_vis)
    cv2.imwrite('8.png', crop_vor_img)
    cv2.imwrite('9.png', crop_circle_img)
    cv2.imwrite('10.png', crop_kmeans_img)

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
    ct = ct[idx,:]
    
    #生成8个角度
    new_coordinate = {}
    new_point = {}
    for i in range(0, 360, 45):
        if i in angle:
            d = dist[angle==i].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i)*1
            valid_idx = np.argwhere(dist[angle==i].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i + 1 in angle:
            d = dist[angle == i+1].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i+1)*1
            valid_idx = np.argwhere(dist[angle==i+1].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
            
        elif i - 1 in angle:
            d = dist[angle == i-1].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i-1)*1
            valid_idx = np.argwhere(dist[angle==i-1].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i + 2 in angle:
            d = dist[angle == i+2].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i+2)*1
            valid_idx = np.argwhere(dist[angle==i+2].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i - 2 in angle:
            d = dist[angle == i-2].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i-2)*1
            valid_idx = np.argwhere(dist[angle==i-2].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i + 3 in angle:
            d = dist[angle == i+3].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i+3)*1
            valid_idx = np.argwhere(dist[angle==i+3].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i - 3 in angle:
            d = dist[angle == i-3].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i-3)*1
            valid_idx = np.argwhere(dist[angle==i-3].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i + 4 in angle:
            d = dist[angle == i+4].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i+4)*1
            valid_idx = np.argwhere(dist[angle==i+4].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i - 4 in angle:
            d = dist[angle == i-4].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i-4)*1
            valid_idx = np.argwhere(dist[angle==i-4].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i + 5 in angle:
            d = dist[angle == i+5].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i+5)*1
            valid_idx = np.argwhere(dist[angle==i+5].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]
        elif i - 5 in angle:
            d = dist[angle == i-5].max()
            new_coordinate[i] = max(1e-2, d)
            valid = (angle==i-5)*1
            valid_idx = np.argwhere(dist[angle==i-5].numpy()==d.numpy())
            count =0
            for ii, v in enumerate(valid):
                if v == 1:
                    count += 1
                if count == valid_idx+1:
                    ori_idx = ii
                    break
            new_point[i] = ct[ori_idx]

    distances = np.zeros(8)

    for a in range(0, 360, 45):
        if a not in new_coordinate.keys():
            new_coordinate[a] = 1e-2
            distances[a//45] = 1e-2
        else:
            distances[a//45] = new_coordinate[a]

    return distances, new_coordinate, new_point

def vis_whole(img_dir, weak_dir, weak_heatmap_dir, weak_distmap_dir, point_dict_dir, superpixel_dir, vor_img_dir):
    img_name = 'TCGA-18-5592-01Z-00-DX1.png'
    # img = cv2.imread(os.path.join(img_dir, img_name))
    weak = cv2.imread(os.path.join(weak_dir, img_name), flags=0)
    weak[weak==1]=128
    # weak_heatmap = cv2.imread(os.path.join(weak_heatmap_dir, img_name), flags=0)
    weak_distmap = cv2.imread(os.path.join(weak_distmap_dir, img_name), flags=0)
    vis = weak_distmap.copy()
    vis[vis==255]=128
    vis_ori = vis.copy()
    vis = np.stack((vis,vis,vis),axis=-1)
    
    # weak_dist = np.stack((weak_distmap,weak_distmap,weak_distmap),axis=-1)
    point_dict = json.load(open(os.path.join(point_dict_dir, img_name.replace('.png','.json')),'r'))
    weak_distmap = measure.label(weak_distmap)
    bnd = segmentation.find_boundaries(weak_distmap, mode='inner')*1
    bnd = morphology.dilation(bnd,morphology.disk(1))
    bnd[0, :] = 1
    bnd[-1,:] = 1
    bnd[:, 0] = 1
    bnd[:,-1] = 1
    bnd = weak_distmap*bnd
    
    fg_points_dict = point_dict['foreground']
    for inst_idx, inst_prop in fg_points_dict.items():
        fg_point = inst_prop['select_point']
        inst_value = weak_distmap[fg_point[1], fg_point[0]]
        bnd_points = np.array(np.where(bnd==inst_value)).T
        bnd_points = bnd_points[:,[1,0]]
        distances, new_coordinates, new_points = get_8_coordinates(fg_point[0], fg_point[1], bnd_points)
        for k,v in new_points.items():
            vis = cv2.arrowedLine(vis, fg_point, v, color=(0,255,0), thickness=2)
    cv2.imwrite('11.png', vis)
    cv2.imwrite('12.png', vis_ori)
    cv2.imwrite('13.png', weak)
    
    h_start,w_start = 500,320
    crop_size = 100
    crop_vis = vis[h_start:h_start+crop_size, w_start:w_start+crop_size, :]
    cv2.imwrite('14.png', crop_vis)

if __name__=="__main__":
    img_dir = './data/MoNuSeg/train/vis_two_points'
    weak_dir = './data/MoNuSeg/train/labels_v5_long_fuse/weak'
    weak_heatmap_dir = './data/MoNuSeg/train/labels_v5_long_fuse/weak_heatmap'
    weak_distmap_dir = './data/MoNuSeg/train/labels_v5_long_fuse/weak_distmap'
    point_dict_dir = './data/MoNuSeg/train/points_v5_long_fuse'
    superpixel_dir = './data/MoNuSeg/train/superpixels_v5'
    vor_img_dir = './data/MoNuSeg/train/voronois_v5'
    circle_dir = '././data/MoNuSeg/train/vis_two_points_circle'
    kmeans_dir = './data/MoNuSeg/train/kmeans'
    vis_crop(img_dir, weak_dir, weak_heatmap_dir, weak_distmap_dir, point_dict_dir, superpixel_dir, vor_img_dir, circle_dir, kmeans_dir)
    vis_whole(img_dir, weak_dir, weak_heatmap_dir, weak_distmap_dir, point_dict_dir, superpixel_dir, vor_img_dir,)
    
    img_dir = './data/MoNuSeg/train/images'
    point_dict_dir = './data/MoNuSeg/train/points_v5_long_fuse'
    output_dir = './data/MoNuSeg/train/vis_two_points'
    # vis_circle(img_dir, point_dict_dir, output_dir)
    